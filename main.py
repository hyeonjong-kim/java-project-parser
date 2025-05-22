import os
import sys
import asyncio # Added for asyncio
from dotenv import load_dotenv
from tqdm import tqdm
from src.services.java_analyzer import JavaAnalyzer, find_java_files

async def main_async(directory_path: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str):
    """Asynchronous function to handle the core analysis logic."""
    analyzer = JavaAnalyzer(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        openai_api_key=openai_api_key
    )
    
    java_files = find_java_files(directory_path) # Get the list of files first
    for java_file in tqdm(java_files, desc="Processing Java files"):
        await analyzer.process_java_file(java_file) # Use await here
    
    # After processing all files, call batch explanation
    print("All files processed. Starting batch explanation of code snippets...")
    await analyzer._batch_explain_code_snippets()
    
    # Flush any remaining updates from batch explanations
    print("Flushing remaining graph updates after batch explanation...")
    analyzer._flush_batch() 
    
    print("Merging duplicate packages...")
    analyzer.merge_duplicate_packages()
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