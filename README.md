# Java Code Analysis Tool

## 주요 기능

- Java 소스 코드의 AST(Abstract Syntax Tree) 분석
- 패키지, 클래스, 인터페이스, 메서드 등의 구조적 분석
- 코드 간의 관계(상속, 구현, 의존성 등) 분석
- GPT-4를 활용한 코드 설명 및 요약 생성
- Neo4j 그래프 데이터베이스에 분석 결과 저장
- **NEW**: Upstage Embeddings를 사용한 코드 임베딩 생성 및 PostgreSQL 저장
- **NEW**: 의미적 코드 검색 기능 (pgvector 기반)

## 요구사항

- Python 3.12 이상
- Neo4j 데이터베이스
- OpenAI API 키
- PostgreSQL with pgvector extension (임베딩 기능 사용 시)
- Upstage API 키 (임베딩 기능 사용 시)

## 설치

1. 저장소를 클론합니다:
```bash
git clone [repository-url]
cd graphrag-test
```

2. 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

3. PostgreSQL에 pgvector 확장을 설치합니다 (임베딩 기능 사용 시):
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

4. 환경 변수 설정:
`.env` 파일을 생성하고 다음 환경 변수들을 설정합니다:
```
# 필수 환경변수
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=your_neo4j_uri
NEO4J_USER=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password

# 임베딩 기능 사용 시 (선택사항)
UPSTAGE_API_KEY=your_upstage_api_key
POSTGRES_URL=postgresql://user:password@localhost:5432/dbname

# 검색 기능 고급 옵션 (선택사항)
SEARCH_NODE_TYPES=JavaClass,JavaMethod  # 검색할 노드 타입 제한
SEARCH_SIMILARITY_THRESHOLD=0.7          # 유사도 임계값
SEARCH_LIMIT=10                          # 최대 검색 결과 수
```

## 사용 방법

### 기본 분석

Java 소스 코드가 있는 디렉토리를 분석하려면:

```bash
python main.py <project_name> <directory_path>
```

예시:
```bash
python main.py my-project ./src/main/java
```

임베딩 서비스가 설정되어 있으면 자동으로 코드 임베딩이 생성되고 PostgreSQL에 저장됩니다.

### 코드 검색

임베딩이 생성된 후 의미적 코드 검색을 할 수 있습니다:

```bash
# 전체 프로젝트에서 검색
python search_embeddings.py "사용자 인증"

# 특정 프로젝트에서 검색
python search_embeddings.py "데이터베이스 연결" my-project

# 임베딩 통계 확인
python search_embeddings.py --stats

# 특정 프로젝트의 임베딩 통계 확인
python search_embeddings.py --stats my-project
```

## 데이터 모델

### Neo4j (그래프 데이터베이스)

프로젝트는 다음과 같은 주요 엔티티들을 포함합니다:

- JavaProject: 프로젝트 정보
- JavaPackage: 패키지 정보
- JavaClass: 클래스 정보
- JavaInterface: 인터페이스 정보
- JavaMethod: 메서드 정보
- JavaEnum: 열거형 정보

각 엔티티는 Neo4j 그래프 데이터베이스에 노드로 저장되며, 엔티티 간의 관계는 엣지로 표현됩니다.

### PostgreSQL (임베딩 데이터베이스)

임베딩 기능을 위한 테이블:

- `code_embeddings`: 코드 임베딩 벡터와 메타데이터 저장
- `embedding_search_logs`: 검색 로그 기록

## 임베딩 기능 특징

- **Upstage embedding-query 모델**: 4096차원 벡터 생성
- **pgvector 기반 유사도 검색**: 코사인 유사도와 L2 거리 지원
- **배치 처리**: 대량의 코드를 효율적으로 처리
- **검색 로그**: 검색 이력 추적 및 분석
- **메타데이터 필터링**: 프로젝트, 노드 타입, 패키지별 필터링 지원

## 성능 최적화

- 배치 처리를 통한 임베딩 생성 최적화
- pgvector 인덱스를 통한 고속 유사도 검색
- 트랜잭션 기반 안전한 데이터 저장
- 병렬 처리를 통한 대용량 코드베이스 지원
