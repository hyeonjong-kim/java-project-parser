# PostgreSQL 임베딩 인덱스 차원 제한 문제 해결 가이드

## 문제 상황

PostgreSQL의 pgvector 확장에서 IVFFlat 인덱스는 최대 2000차원까지만 지원합니다. 하지만 Upstage의 `embedding-query` 모델은 4096차원의 벡터를 생성하므로, 다음과 같은 오류가 발생합니다:

```
Warning: Failed to initialize embedding service: (psycopg2.errors.ProgramLimitExceeded) column cannot have more than 2000 dimensions for ivfflat index
```

## 해결 방법

### 1. HNSW 인덱스 사용 (권장)

HNSW(Hierarchical Navigable Small World) 인덱스는 차원 제한이 없으며, 일반적으로 IVFFlat보다 더 나은 성능을 제공합니다.

#### 자동 마이그레이션

코드가 이미 업데이트되어 자동으로 HNSW 인덱스를 사용하도록 변경되었습니다. 새로운 데이터베이스에서는 자동으로 HNSW 인덱스가 생성됩니다.

#### 기존 데이터베이스 마이그레이션

기존에 IVFFlat 인덱스를 사용하던 데이터베이스가 있다면, 다음 스크립트를 실행하여 마이그레이션하세요:

```bash
# 환경변수 설정 (선택사항)
export POSTGRES_URL="postgresql://username:password@localhost:5432/database_name"

# 마이그레이션 실행
python scripts/migrate_embedding_indexes.py

# 또는 직접 URL 지정
python scripts/migrate_embedding_indexes.py --postgres-url "postgresql://username:password@localhost:5432/database_name"
```

### 2. 수동 SQL 마이그레이션

만약 스크립트를 사용할 수 없다면, 다음 SQL을 직접 실행하세요:

```sql
-- 1. 기존 IVFFlat 인덱스 삭제
DROP INDEX IF EXISTS idx_embedding_cosine;
DROP INDEX IF EXISTS idx_embedding_l2;

-- 2. HNSW 인덱스 생성
CREATE INDEX idx_embedding_cosine ON code_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_embedding_l2 ON code_embeddings USING hnsw (embedding vector_l2_ops);
```

## 변경 사항 요약

### 1. 모델 변경 (`src/models/embedding_models.py`)

```python
# 이전 (IVFFlat)
Index('idx_embedding_cosine', 'embedding', postgresql_using='ivfflat', postgresql_ops={'embedding': 'vector_cosine_ops'}),

# 현재 (HNSW)
Index('idx_embedding_cosine', 'embedding', postgresql_using='hnsw', postgresql_ops={'embedding': 'vector_cosine_ops'}),
```

### 2. 서비스 업데이트 (`src/services/embedding_service.py`)

- 자동 인덱스 마이그레이션 기능 추가
- 기존 IVFFlat 인덱스 감지 및 자동 삭제
- HNSW 인덱스 자동 생성

### 3. 마이그레이션 스크립트 (`scripts/migrate_embedding_indexes.py`)

- 기존 데이터베이스의 안전한 마이그레이션
- 진행 상황 표시
- 오류 처리 및 롤백 기능

## HNSW vs IVFFlat 비교

| 특성 | IVFFlat | HNSW |
|------|---------|------|
| 차원 제한 | 2000차원 | 제한 없음 |
| 검색 속도 | 빠름 | 매우 빠름 |
| 정확도 | 좋음 | 우수함 |
| 메모리 사용량 | 적음 | 중간 |
| 구축 시간 | 중간 | 김 |

## 확인 방법

마이그레이션 완료 후 다음 SQL로 인덱스를 확인할 수 있습니다:

```sql
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'code_embeddings' 
AND indexdef LIKE '%hnsw%';
```

예상 출력:
```
     indexname      |                            indexdef                             
--------------------+-----------------------------------------------------------------
 idx_embedding_cosine | CREATE INDEX idx_embedding_cosine ON public.code_embeddings USING hnsw (embedding vector_cosine_ops)
 idx_embedding_l2   | CREATE INDEX idx_embedding_l2 ON public.code_embeddings USING hnsw (embedding vector_l2_ops)
```

## 문제 해결

### pgvector 버전 확인

HNSW를 사용하려면 pgvector 0.5.0 이상이 필요합니다:

```sql
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

### 권한 확인

인덱스 생성에 필요한 권한이 있는지 확인:

```sql
-- 현재 사용자 권한 확인
SELECT current_user;

-- 테이블 소유자 확인
SELECT tableowner FROM pg_tables WHERE tablename = 'code_embeddings';
```

### 로그 확인

애플리케이션 실행 시 다음과 같은 로그가 나타나면 성공:

```
Database initialized successfully with pgvector extension
Index migration completed successfully
Embedding service initialized successfully
```

## 추가 최적화

### HNSW 파라미터 튜닝

더 나은 성능을 위해 HNSW 파라미터를 조정할 수 있습니다:

```sql
-- 기본 인덱스 삭제 후 파라미터 조정한 인덱스 생성
DROP INDEX idx_embedding_cosine;
CREATE INDEX idx_embedding_cosine ON code_embeddings 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);
```

- `m`: 그래프 연결성 (기본값: 16, 범위: 2-100)
- `ef_construction`: 구축 품질 (기본값: 64, 범위: 4-1000)

## 성능 모니터링

인덱스 사용량 및 성능 모니터링:

```sql
-- 인덱스 사용 통계
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE indexname LIKE 'idx_embedding%';

-- 인덱스 크기
SELECT 
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as size
FROM pg_indexes 
WHERE tablename = 'code_embeddings';
``` 