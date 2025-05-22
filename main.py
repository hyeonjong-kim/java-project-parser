import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm
from src.services.java_analyzer import JavaAnalyzer, find_java_files

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
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}'가 존재하지 않습니다.")
        sys.exit(1)
    
    analyzer = JavaAnalyzer(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        openai_api_key=openai_api_key
    )
    
    for java_file in tqdm(find_java_files(directory_path)):
        analyzer.process_java_file(java_file)
    
    analyzer.merge_duplicate_packages()

if __name__ == "__main__":
    main() 