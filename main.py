import os
import sys
import asyncio # Added for asyncio
import time  # Add for performance monitoring
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from src.services.java_analyzer import JavaAnalyzer, find_java_files

async def process_files_in_batches(analyzer: JavaAnalyzer, java_files: list, batch_size: int = 10, max_concurrent: int = 5):
    """Process Java files in parallel batches with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    start_time = time.time()
    
    async def process_single_file(file_path):
        async with semaphore:
            try:
                file_start_time = time.time()
                await analyzer.process_java_file(file_path, flush_immediately=False)
                file_end_time = time.time()
                return True, file_end_time - file_start_time
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                return False, 0
    
    # Process files in batches
    total_files = len(java_files)
    processed_count = 0
    total_file_time = 0
    
    for i in range(0, total_files, batch_size):
        batch = java_files[i:i + batch_size]
        batch_start_time = time.time()
        print(f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size} ({len(batch)} files)")
        
        # Process current batch in parallel
        tasks = [process_single_file(file_path) for file_path in batch]
        results = await tqdm_asyncio.gather(*tasks, desc=f"Batch {i//batch_size + 1}")
        
        # Flush the batch to database
        flush_start_time = time.time()
        print("Flushing batch to database...")
        analyzer._flush_batch()
        flush_end_time = time.time()
        
        # Calculate performance metrics
        processed_count += len(batch)
        success_count = sum(1 for result, _ in results if result)
        batch_file_time = sum(file_time for _, file_time in results)
        total_file_time += batch_file_time
        
        batch_end_time = time.time()
        batch_total_time = batch_end_time - batch_start_time
        flush_time = flush_end_time - flush_start_time
        
        print(f"Batch completed: {success_count}/{len(batch)} files successful.")
        print(f"Batch timing - Processing: {batch_file_time:.2f}s, Flushing: {flush_time:.2f}s, Total: {batch_total_time:.2f}s")
        print(f"Total progress: {processed_count}/{total_files} ({processed_count/total_files*100:.1f}%)")
        
        # Estimate remaining time
        if processed_count > 0:
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time * total_files / processed_count
            remaining_time = estimated_total_time - elapsed_time
            print(f"Estimated remaining time: {remaining_time/60:.1f} minutes")
        print("-" * 50)

async def main_async(directory_path: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str):
    """Asynchronous function to handle the core analysis logic."""
    analyzer = JavaAnalyzer(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        openai_api_key=openai_api_key
    )
    
    # Initialize database schema with proper constraints
    print("Initializing database schema...")
    analyzer.initialize_database_schema()
    
    java_files = find_java_files(directory_path)
    print(f"Found {len(java_files)} Java files to process")
    
    if java_files:
        # Determine optimal batch size and concurrency based on file count
        if len(java_files) < 50:
            batch_size, max_concurrent = 10, 4
        elif len(java_files) < 200:
            batch_size, max_concurrent = 25, 6
        else:
            batch_size, max_concurrent = 40, 8
            
        print(f"Using batch_size={batch_size}, max_concurrent={max_concurrent} for {len(java_files)} files")
        
        # Process files in parallel batches
        await process_files_in_batches(
            analyzer, 
            java_files, 
            batch_size=batch_size,
            max_concurrent=max_concurrent
        )
    
    # Final flush for any remaining updates
    print("Final flush of remaining graph updates...")
    analyzer._flush_batch()
    
    print("Merging duplicate packages...")
    analyzer.merge_duplicate_packages()

    # Clean up incorrectly created package nodes (class names that were treated as packages)
    print("Cleaning up incorrectly created package nodes...")
    analyzer.clean_incorrect_package_nodes()

    # Collect all analyzed vertices
    print("Collecting final set of analyzed vertices from the graph...")
    all_vertices = analyzer.collect_all_analyzed_vertices()

    # Get explanations for each vertex and update them in the graph
    if all_vertices:
        print(f"\nStarting LLM explanation generation and graph update for {len(all_vertices)} vertices...")
        
        # Define the desired order of processing for vertex labels
        label_order = [
            "JavaMethod", "JavaEnum", "JavaInterface", "JavaClass",
            "JavaLeafPackage", "JavaInternalPackage" # Assuming JavaPackage meant these specific types
        ]
        
        # Custom sorting key function
        def sort_key(vertex):
            # Get the first label of the vertex for sorting, default to a high index if not in order
            primary_label = vertex.get("labels", [""])[0]
            try:
                return label_order.index(primary_label)
            except ValueError:
                return len(label_order) # Place items not in label_order at the end

        # Sort all_vertices based on the custom key
        sorted_vertices = sorted(all_vertices, key=sort_key)
        
        for vertex_info in tqdm(sorted_vertices, desc="Generating Explanations & Updating Graph"):
            if vertex_info.get("labels") == ["JavaInternalPackage"]:
                continue;

            # Skip if vertex_info is None or essential keys are missing (robustness)
            if not vertex_info or not vertex_info.get("name") or not vertex_info.get("labels"):
                print(f"Skipping invalid vertex_info: {vertex_info}")
                continue

            # Get description and summary from LLM
            description, summary = analyzer._get_explanation_code(vertex_info)
            
            # Update the vertex in Neo4j with the new description and summary
            if description and summary: # Ensure we have valid explanation parts to update
                update_success = analyzer.update_vertex_explanation_in_graph(vertex_info, description, summary)
                if not update_success:
                    print(f"Failed to update explanation for: {vertex_info.get('name')}")
            else:
                print(f"Skipping graph update for {vertex_info.get('name')} due to missing explanation parts.")
        print("LLM explanation generation and graph update complete.")
    else:
        print("No vertices were collected, so no explanations to generate.")

    print("Analysis complete.")



def main():
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY가 환경변수에 설정되어 있지 않습니다.")
    
    neo4j_uri = os.getenv("NEO4J_URI")
    if not neo4j_uri:
        raise ValueError("NEO4J_URI가 환경변수에 설정되어 있지 않습니다.")
    
    neo4j_user = os.getenv("NEO4J_USER")
    if not neo4j_user:
        raise ValueError("NEO4J_USER가 환경변수에 설정되어 있지 않습니다.")
    
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not neo4j_password:
        raise ValueError("NEO4J_PASSWORD가 환경변수에 설정되어 있지 않습니다.")

    # 명령행 인자로 디렉토리 경로를 받음
    if len(sys.argv) != 2:
        print("Usage: python main.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path): # Changed to isdir for clarity, though exists also works for dirs
        print(f"Error: Directory '{directory_path}'가 존재하지 않거나 디렉토리가 아닙니다.")
        sys.exit(1)
    
    # Run the asynchronous main function
    asyncio.run(main_async(directory_path, neo4j_uri, neo4j_user, neo4j_password, openai_api_key))

if __name__ == "__main__":
    main() 