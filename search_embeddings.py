#!/usr/bin/env python3
"""
임베딩을 사용하여 유사한 코드를 검색하는 스크립트
Usage: python search_embeddings.py "검색어" [프로젝트명]
"""

import os
import sys
from dotenv import load_dotenv
from src.services.embedding_service import EmbeddingService
import json
from typing import Optional

def search_code(query: str, project_name: Optional[str] = None, 
                node_types: Optional[list] = None, limit: int = 10, 
                similarity_threshold: float = 0.7):
    """코드 검색 실행"""
    load_dotenv()
    
    upstage_api_key = os.getenv("UPSTAGE_API_KEY")
    postgres_url = os.getenv("POSTGRES_URL")
    
    if not upstage_api_key or not postgres_url:
        print("Error: UPSTAGE_API_KEY 또는 POSTGRES_URL이 환경변수에 설정되어 있지 않습니다.")
        return
    
    try:
        # 임베딩 서비스 초기화
        embedding_service = EmbeddingService(upstage_api_key, postgres_url)
        
        print(f"검색어: '{query}'")
        if project_name:
            print(f"프로젝트: {project_name}")
        if node_types:
            print(f"노드 타입 필터: {node_types}")
        print(f"유사도 임계값: {similarity_threshold}")
        print(f"최대 결과 수: {limit}")
        print("-" * 50)
        
        # 검색 실행
        results = embedding_service.search_similar_code(
            query=query,
            project_name=project_name,
            node_types=node_types,
            limit=limit,
            similarity_threshold=similarity_threshold
        )
        
        if not results:
            print("검색 결과가 없습니다.")
            return
        
        print(f"검색 결과: {len(results)}개")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['node_type']}: {result['node_name']}")
            print(f"    프로젝트: {result['project_name']}")
            print(f"    패키지: {result.get('package_name', 'N/A')}")
            print(f"    유사도: {result['similarity']:.3f}")
            print(f"    내부 코드: {'예' if result.get('is_internal') else '아니오'}")
            
            if result.get('summary'):
                print(f"    요약: {result['summary']}")
            
            if result.get('description') and len(result['description']) > 200:
                print(f"    설명: {result['description'][:200]}...")
            elif result.get('description'):
                print(f"    설명: {result['description']}")
            
            print("-" * 40)
    
    except Exception as e:
        print(f"검색 중 오류가 발생했습니다: {e}")

def print_statistics(project_name: Optional[str] = None):
    """임베딩 통계 출력"""
    load_dotenv()
    
    upstage_api_key = os.getenv("UPSTAGE_API_KEY")
    postgres_url = os.getenv("POSTGRES_URL")
    
    if not upstage_api_key or not postgres_url:
        print("Error: UPSTAGE_API_KEY 또는 POSTGRES_URL이 환경변수에 설정되어 있지 않습니다.")
        return
    
    try:
        embedding_service = EmbeddingService(upstage_api_key, postgres_url)
        stats = embedding_service.get_embedding_statistics(project_name)
        
        print("=== 임베딩 통계 ===")
        if project_name:
            print(f"프로젝트: {project_name}")
        
        print(f"총 임베딩 수: {stats.get('total_embeddings', 0)}")
        print("\n노드 타입별 분포:")
        
        for node_type, count in stats.get('by_node_type', {}).items():
            print(f"  {node_type}: {count}개")
        
        print("=" * 20)
        
    except Exception as e:
        print(f"통계 조회 중 오류가 발생했습니다: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python search_embeddings.py \"검색어\" [프로젝트명]")
        print("  python search_embeddings.py --stats [프로젝트명]")
        print("\nExamples:")
        print("  python search_embeddings.py \"사용자 인증\"")
        print("  python search_embeddings.py \"데이터베이스 연결\" my-project")
        print("  python search_embeddings.py --stats")
        print("  python search_embeddings.py --stats my-project")
        sys.exit(1)
    
    if sys.argv[1] == "--stats":
        project_name = sys.argv[2] if len(sys.argv) > 2 else None
        print_statistics(project_name)
    else:
        query = sys.argv[1]
        project_name = sys.argv[2] if len(sys.argv) > 2 else None
        
        # 고급 검색 옵션 (환경변수로 설정 가능)
        node_types = None
        if os.getenv("SEARCH_NODE_TYPES"):
            node_types = os.getenv("SEARCH_NODE_TYPES").split(",")
        
        similarity_threshold = float(os.getenv("SEARCH_SIMILARITY_THRESHOLD", "0.7"))
        limit = int(os.getenv("SEARCH_LIMIT", "10"))
        
        search_code(query, project_name, node_types, limit, similarity_threshold)

if __name__ == "__main__":
    main() 