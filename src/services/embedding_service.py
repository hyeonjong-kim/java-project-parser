from typing import List, Dict, Optional, Tuple
import logging
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from langchain_upstage import UpstageEmbeddings
import numpy as np
from tqdm import tqdm

# 프로젝트 root에서 import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.embedding_models import Base, CodeEmbedding, EmbeddingSearchLog

logger = logging.getLogger(__name__)

class EmbeddingService:
    """코드 임베딩 생성 및 PostgreSQL 저장을 담당하는 서비스"""
    
    def __init__(self, 
                 upstage_api_key: str,
                 postgres_url: str):
        """
        Args:
            upstage_api_key: Upstage API 키
            postgres_url: PostgreSQL 연결 URL (예: postgresql://user:pass@localhost:5432/dbname)
        """
        self.upstage_api_key = upstage_api_key
        self.postgres_url = postgres_url
        
        # Upstage Embeddings 초기화
        self.embeddings = UpstageEmbeddings(
            api_key=upstage_api_key,
            model="embedding-query"
        )
        
        # SQLAlchemy 엔진 및 세션 설정
        self.engine = create_engine(postgres_url, pool_pre_ping=True, pool_recycle=3600)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # 데이터베이스 초기화
        self._initialize_database()
    
    def _initialize_database(self):
        """데이터베이스 테이블 생성 및 pgvector 확장 활성화"""
        try:
            with self.engine.connect() as conn:
                # pgvector 확장 활성화
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            
            # 테이블 생성
            Base.metadata.create_all(bind=self.engine)
            
            # 기존 IVFFlat 인덱스가 있다면 삭제하고 HNSW로 교체
            self._migrate_indexes()
            
            logger.info("Database initialized successfully with pgvector extension")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _migrate_indexes(self):
        """IVFFlat 인덱스를 HNSW 인덱스로 마이그레이션"""
        try:
            with self.engine.connect() as conn:
                # 기존 IVFFlat 인덱스 확인 및 삭제
                check_ivfflat_query = text("""
                    SELECT indexname FROM pg_indexes 
                    WHERE tablename = 'code_embeddings' 
                    AND indexdef LIKE '%ivfflat%'
                """)
                
                result = conn.execute(check_ivfflat_query)
                ivfflat_indexes = [row[0] for row in result]
                
                for index_name in ivfflat_indexes:
                    logger.info(f"Dropping IVFFlat index: {index_name}")
                    conn.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
                
                # HNSW 인덱스 생성 (테이블이 이미 존재하는 경우)
                hnsw_indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_embedding_cosine ON code_embeddings USING hnsw (embedding vector_cosine_ops)",
                    "CREATE INDEX IF NOT EXISTS idx_embedding_l2 ON code_embeddings USING hnsw (embedding vector_l2_ops)"
                ]
                
                for index_sql in hnsw_indexes:
                    try:
                        logger.info(f"Creating HNSW index...")
                        conn.execute(text(index_sql))
                    except Exception as e:
                        if "already exists" not in str(e):
                            logger.warning(f"Could not create HNSW index: {e}")
                
                conn.commit()
                logger.info("Index migration completed successfully")
                
        except Exception as e:
            logger.warning(f"Index migration failed (this may be normal for new databases): {e}")
    
    def create_embedding(self, text: str) -> List[float]:
        """단일 텍스트에 대한 임베딩 생성"""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트에 대한 임베딩 배치 생성"""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}")
            raise
    
    def prepare_embedding_text(self, vertex_info: Dict) -> str:
        """정점 정보로부터 임베딩용 텍스트 준비 (description과 summary만 사용)"""
        description = vertex_info.get("description", "")
        summary = vertex_info.get("summary", "")
        
        # 임베딩용 텍스트 구성 (description과 summary만 사용)
        text_parts = []
        
        # 요약이 있으면 추가 (가장 중요한 정보)
        if summary and summary.strip():
            text_parts.append(summary)
        
        # 설명이 있으면 추가
        if description and description.strip():
            text_parts.append(description)
        
        # Type, Name, Code Body 모두 제외하고 순수 의미 정보만 사용
        # Note: 메타데이터(type, name, code_body)는 데이터베이스에는 저장되지만 임베딩 생성에는 사용하지 않음
        
        return " | ".join(text_parts) if text_parts else ""
    
    def save_embedding(self, 
                      vertex_info: Dict, 
                      embedding: List[float],
                      project_name: str,
                      package_name: Optional[str] = None,
                      file_path: Optional[str] = None) -> bool:
        """단일 임베딩을 데이터베이스에 저장"""
        db = self.SessionLocal()
        try:
            node_type = vertex_info.get("labels", ["Unknown"])[0] if vertex_info.get("labels") else "Unknown"
            node_name = vertex_info.get("name", "")
            
            # 기존 레코드 확인
            existing = db.query(CodeEmbedding).filter(
                CodeEmbedding.node_type == node_type,
                CodeEmbedding.node_name == node_name,
                CodeEmbedding.project_name == project_name
            ).first()
            
            if existing:
                # 업데이트
                existing.code_body = vertex_info.get("body", "")
                existing.description = vertex_info.get("description", "")
                existing.summary = vertex_info.get("summary", "")
                existing.embedding = embedding
                existing.package_name = package_name
                existing.file_path = file_path
                logger.info(f"Updated embedding for {node_type}: {node_name}")
            else:
                # 새로 생성
                code_embedding = CodeEmbedding(
                    node_type=node_type,
                    node_name=node_name,
                    project_name=project_name,
                    code_body=vertex_info.get("body", ""),
                    description=vertex_info.get("description", ""),
                    summary=vertex_info.get("summary", ""),
                    embedding=embedding,
                    package_name=package_name,
                    is_internal=self._is_internal_node(vertex_info),
                    file_path=file_path
                )
                db.add(code_embedding)
                logger.info(f"Created new embedding for {node_type}: {node_name}")
            
            db.commit()
            return True
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error saving embedding for {vertex_info.get('name', 'Unknown')}: {e}")
            return False
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving embedding for {vertex_info.get('name', 'Unknown')}: {e}")
            return False
        finally:
            db.close()
    
    def save_embeddings_batch(self, 
                             vertex_infos: List[Dict], 
                             embeddings: List[List[float]],
                             project_name: str,
                             batch_size: int = 100) -> Tuple[int, int]:
        """임베딩 배치를 데이터베이스에 저장"""
        if len(vertex_infos) != len(embeddings):
            raise ValueError("vertex_infos와 embeddings의 길이가 일치하지 않습니다")
        
        success_count = 0
        error_count = 0
        
        # 배치 단위로 처리
        for i in range(0, len(vertex_infos), batch_size):
            batch_vertices = vertex_infos[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            db = self.SessionLocal()
            try:
                for vertex_info, embedding in zip(batch_vertices, batch_embeddings):
                    try:
                        node_type = vertex_info.get("labels", ["Unknown"])[0] if vertex_info.get("labels") else "Unknown"
                        node_name = vertex_info.get("name", "")
                        
                        # 기존 레코드 확인
                        existing = db.query(CodeEmbedding).filter(
                            CodeEmbedding.node_type == node_type,
                            CodeEmbedding.node_name == node_name,
                            CodeEmbedding.project_name == project_name
                        ).first()
                        
                        if existing:
                            # 업데이트
                            existing.code_body = vertex_info.get("body", "")
                            existing.description = vertex_info.get("description", "")
                            existing.summary = vertex_info.get("summary", "")
                            existing.embedding = embedding
                        else:
                            # 새로 생성
                            code_embedding = CodeEmbedding(
                                node_type=node_type,
                                node_name=node_name,
                                project_name=project_name,
                                code_body=vertex_info.get("body", ""),
                                description=vertex_info.get("description", ""),
                                summary=vertex_info.get("summary", ""),
                                embedding=embedding,
                                package_name=self._extract_package_name(vertex_info),
                                is_internal=self._is_internal_node(vertex_info)
                            )
                            db.add(code_embedding)
                        
                        success_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing vertex {vertex_info.get('name', 'Unknown')}: {e}")
                        error_count += 1
                        continue
                
                db.commit()
                logger.info(f"Batch {i//batch_size + 1} saved: {len(batch_vertices)} items")
                
            except Exception as e:
                db.rollback()
                logger.error(f"Batch save error: {e}")
                error_count += len(batch_vertices)
            finally:
                db.close()
        
        return success_count, error_count
    
    def search_similar_code(self, 
                           query: str, 
                           project_name: Optional[str] = None,
                           node_types: Optional[List[str]] = None,
                           limit: int = 10,
                           similarity_threshold: float = 0.7,
                           search_type: str = "cosine") -> List[Dict]:
        """유사한 코드 검색"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.create_embedding(query)
            
            db = self.SessionLocal()
            try:
                # 기본 쿼리 구성
                if search_type == "cosine":
                    distance_expr = CodeEmbedding.embedding.cosine_distance(query_embedding)
                    order_expr = distance_expr
                else:  # L2 distance
                    distance_expr = CodeEmbedding.embedding.l2_distance(query_embedding)
                    order_expr = distance_expr
                
                query_obj = db.query(
                    CodeEmbedding,
                    distance_expr.label('distance')
                )
                
                # 필터 적용
                if project_name:
                    query_obj = query_obj.filter(CodeEmbedding.project_name == project_name)
                
                if node_types:
                    query_obj = query_obj.filter(CodeEmbedding.node_type.in_(node_types))
                
                # 유사도 임계값 적용
                if search_type == "cosine":
                    query_obj = query_obj.filter(distance_expr < (1 - similarity_threshold))
                
                # 정렬 및 제한
                results = query_obj.order_by(order_expr).limit(limit).all()
                
                # 결과 포맷팅
                formatted_results = []
                for code_embedding, distance in results:
                    similarity = 1 - distance if search_type == "cosine" else distance
                    formatted_results.append({
                        'id': str(code_embedding.id),
                        'node_type': code_embedding.node_type,
                        'node_name': code_embedding.node_name,
                        'project_name': code_embedding.project_name,
                        'description': code_embedding.description,
                        'summary': code_embedding.summary,
                        'package_name': code_embedding.package_name,
                        'is_internal': code_embedding.is_internal,
                        'similarity': float(similarity),
                        'distance': float(distance)
                    })
                
                # 검색 로그 저장
                self._log_search(db, query, query_embedding, project_name, len(formatted_results), search_type, str(similarity_threshold))
                
                return formatted_results
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error searching similar code: {e}")
            return []
    
    def _log_search(self, db: Session, query_text: str, query_embedding: List[float], 
                   project_name: Optional[str], num_results: int, search_type: str, threshold: str):
        """검색 로그 저장"""
        try:
            search_log = EmbeddingSearchLog(
                query_text=query_text,
                query_embedding=query_embedding,
                project_name=project_name,
                num_results=num_results,
                search_type=search_type,
                threshold=threshold
            )
            db.add(search_log)
            db.commit()
        except Exception as e:
            logger.error(f"Error logging search: {e}")
    
    def _extract_package_name(self, vertex_info: Dict) -> Optional[str]:
        """정점 정보에서 패키지 이름 추출"""
        # 이는 JavaAnalyzer에서 설정되는 패키지 정보에 따라 달라질 수 있음
        return vertex_info.get("package_name")
    
    def _is_internal_node(self, vertex_info: Dict) -> bool:
        """노드가 내부 코드인지 판단"""
        # JavaPackage의 경우 is_internal 속성 확인
        if "JavaPackage" in vertex_info.get("labels", []):
            return vertex_info.get("is_internal", True)
        # 다른 타입의 경우 기본적으로 내부로 간주
        return True
    
    def get_embedding_statistics(self, project_name: Optional[str] = None) -> Dict:
        """임베딩 통계 조회"""
        db = self.SessionLocal()
        try:
            query = db.query(CodeEmbedding)
            if project_name:
                query = query.filter(CodeEmbedding.project_name == project_name)
            
            total_count = query.count()
            
            # 노드 타입별 통계
            type_stats = {}
            for node_type, count in db.query(CodeEmbedding.node_type, func.count(CodeEmbedding.id)).group_by(CodeEmbedding.node_type).all():
                type_stats[node_type] = count
            
            return {
                'total_embeddings': total_count,
                'by_node_type': type_stats,
                'project_name': project_name
            }
        finally:
            db.close() 