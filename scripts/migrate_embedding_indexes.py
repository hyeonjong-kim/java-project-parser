#!/usr/bin/env python3
"""
임베딩 인덱스 마이그레이션 스크립트
IVFFlat 인덱스에서 HNSW 인덱스로 마이그레이션합니다.

사용법:
    python scripts/migrate_embedding_indexes.py --postgres-url "postgresql://user:pass@localhost:5432/dbname"
"""

import sys
import os
import argparse
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def migrate_indexes(postgres_url: str):
    """IVFFlat 인덱스를 HNSW 인덱스로 마이그레이션"""
    print("PostgreSQL 임베딩 인덱스 마이그레이션을 시작합니다...")
    
    try:
        engine = create_engine(postgres_url, pool_pre_ping=True)
        
        with engine.connect() as conn:
            # 먼저 테이블 존재 여부 확인
            print("0. code_embeddings 테이블 존재 여부 확인 중...")
            
            check_table_query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'code_embeddings'
                );
            """)
            
            result = conn.execute(check_table_query)
            table_exists = list(result)[0][0]
            
            if not table_exists:
                print("   ℹ code_embeddings 테이블이 존재하지 않습니다.")
                print("   이는 정상적인 상황입니다. (아직 임베딩 기능을 사용하지 않았음)")
                print("\n📝 다음 단계:")
                print("   1. Java 분석 코드에서 upstage_api_key와 postgres_url을 제공하세요")
                print("   2. 임베딩 기능이 첫 실행될 때 자동으로 HNSW 인덱스가 생성됩니다")
                print("\n✓ 마이그레이션 준비가 완료되었습니다!")
                return
            
            print("   ✓ code_embeddings 테이블이 존재합니다.")
            
            print("\n1. 기존 IVFFlat 인덱스 확인 중...")
            
            # 기존 IVFFlat 인덱스 확인
            check_ivfflat_query = text("""
                SELECT indexname, indexdef FROM pg_indexes 
                WHERE tablename = 'code_embeddings' 
                AND indexdef LIKE '%ivfflat%'
            """)
            
            result = conn.execute(check_ivfflat_query)
            ivfflat_indexes = [(row[0], row[1]) for row in result]
            
            if not ivfflat_indexes:
                print("   ℹ IVFFlat 인덱스가 발견되지 않았습니다.")
                
                # HNSW 인덱스가 이미 있는지 확인
                check_hnsw_query = text("""
                    SELECT indexname FROM pg_indexes 
                    WHERE tablename = 'code_embeddings' 
                    AND indexdef LIKE '%hnsw%'
                """)
                
                result = conn.execute(check_hnsw_query)
                hnsw_indexes = [row[0] for row in result]
                
                if hnsw_indexes:
                    print(f"   ✓ HNSW 인덱스가 이미 존재합니다: {', '.join(hnsw_indexes)}")
                    print("\n✓ 마이그레이션이 이미 완료되었습니다!")
                    return
            else:
                print(f"   ⚠ {len(ivfflat_indexes)}개의 IVFFlat 인덱스를 발견했습니다:")
                for index_name, index_def in ivfflat_indexes:
                    print(f"   - {index_name}")
            
            print("\n2. IVFFlat 인덱스 삭제 중...")
            for index_name, _ in ivfflat_indexes:
                print(f"   삭제 중: {index_name}")
                conn.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
            
            print("\n3. HNSW 인덱스 생성 중...")
            
            # HNSW 인덱스 생성
            hnsw_indexes = [
                ("idx_embedding_cosine", "CREATE INDEX IF NOT EXISTS idx_embedding_cosine ON code_embeddings USING hnsw (embedding vector_cosine_ops)"),
                ("idx_embedding_l2", "CREATE INDEX IF NOT EXISTS idx_embedding_l2 ON code_embeddings USING hnsw (embedding vector_l2_ops)")
            ]
            
            for index_name, index_sql in hnsw_indexes:
                try:
                    print(f"   생성 중: {index_name}")
                    conn.execute(text(index_sql))
                    print(f"   ✓ {index_name} 생성 완료")
                except Exception as e:
                    if "already exists" in str(e):
                        print(f"   ℹ {index_name} 이미 존재함")
                    else:
                        print(f"   ✗ {index_name} 생성 실패: {e}")
                        raise
            
            # 변경사항 커밋
            conn.commit()
            
            print("\n4. 마이그레이션 검증 중...")
            
            # 결과 확인
            check_hnsw_query = text("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'code_embeddings' 
                AND indexdef LIKE '%hnsw%'
            """)
            
            result = conn.execute(check_hnsw_query)
            hnsw_indexes_created = [row[0] for row in result]
            
            print(f"   생성된 HNSW 인덱스: {len(hnsw_indexes_created)}개")
            for index_name in hnsw_indexes_created:
                print(f"   - {index_name}")
            
            print("\n✓ 마이그레이션이 성공적으로 완료되었습니다!")
            
    except Exception as e:
        print(f"\n✗ 마이그레이션 중 오류 발생: {e}")
        print("\n문제 해결 방법:")
        print("1. PostgreSQL 서버가 실행 중인지 확인하세요")
        print("2. 연결 URL이 올바른지 확인하세요")
        print("3. pgvector 확장이 설치되어 있는지 확인하세요")
        print("4. 데이터베이스 사용자에게 인덱스 생성 권한이 있는지 확인하세요")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="임베딩 인덱스 마이그레이션 (IVFFlat → HNSW)")
    parser.add_argument("--postgres-url", 
                       help="PostgreSQL 연결 URL (예: postgresql://user:pass@localhost:5432/dbname)")
    
    args = parser.parse_args()
    
    # 환경변수에서 URL 로드 시도
    if not args.postgres_url:
        load_dotenv()
        postgres_url = os.getenv("POSTGRES_URL")
        if not postgres_url:
            print("오류: PostgreSQL URL이 제공되지 않았습니다.")
            print("다음 중 하나를 사용하세요:")
            print("  1. --postgres-url 옵션 사용")
            print("  2. .env 파일에 POSTGRES_URL 설정")
            sys.exit(1)
    else:
        postgres_url = args.postgres_url
    
    print(f"PostgreSQL URL: {postgres_url.replace(postgres_url.split('@')[0].split('//')[1], '***')}")
    
    migrate_indexes(postgres_url)

if __name__ == "__main__":
    main() 