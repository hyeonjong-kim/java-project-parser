from sqlalchemy import Column, String, Text, Integer, DateTime, Boolean, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime

Base = declarative_base()

class CodeEmbedding(Base):
    """Java 코드 분석 결과의 임베딩을 저장하는 테이블"""
    __tablename__ = 'code_embeddings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_type = Column(String(50), nullable=False)  # JavaClass, JavaMethod, JavaPackage 등
    node_name = Column(String(500), nullable=False)  # 노드의 이름 또는 signature
    project_name = Column(String(200), nullable=False)  # 프로젝트 이름
    
    # 텍스트 내용
    code_body = Column(Text)  # 원본 코드
    description = Column(Text)  # LLM이 생성한 설명
    summary = Column(Text)  # LLM이 생성한 요약
    
    # 임베딩 벡터 (1536 차원은 OpenAI ada-002 기준, Upstage 모델에 맞게 조정 필요)
    embedding = Column(Vector(4096))  # Upstage embedding-query 모델은 4096 차원
    
    # 메타데이터
    package_name = Column(String(500))  # 패키지 이름
    is_internal = Column(Boolean, default=True)  # 내부 코드인지 여부
    file_path = Column(String(1000))  # 파일 경로
    
    # 타임스탬프
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 인덱스 정의 (벡터 인덱스 제거 - 4096차원 지원)
    __table_args__ = (
        Index('idx_node_type', 'node_type'),
        Index('idx_project_name', 'project_name'),
        Index('idx_package_name', 'package_name'),
        Index('idx_is_internal', 'is_internal'),
        Index('idx_node_name', 'node_name'),
        # 벡터 인덱스는 2000차원 제한으로 인해 제거
        # 필요시 차원을 줄이거나 pgvector 업그레이드 후 추가 가능
    )

    def __repr__(self):
        return f"<CodeEmbedding(id={self.id}, node_type={self.node_type}, node_name={self.node_name[:50]}...)>"

class EmbeddingSearchLog(Base):
    """임베딩 검색 로그를 저장하는 테이블"""
    __tablename__ = 'embedding_search_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text = Column(Text, nullable=False)  # 검색 쿼리
    query_embedding = Column(Vector(4096))  # 쿼리 임베딩
    project_name = Column(String(200))  # 검색 대상 프로젝트
    
    # 검색 결과 메타데이터
    num_results = Column(Integer)  # 반환된 결과 수
    search_type = Column(String(50))  # cosine, l2 등
    threshold = Column(String(20))  # 유사도 임계값
    
    # 타임스탬프
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 인덱스
    __table_args__ = (
        Index('idx_search_project', 'project_name'),
        Index('idx_search_created', 'created_at'),
    ) 