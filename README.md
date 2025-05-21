# GraphRAG - Java Code Analysis Tool

GraphRAG는 Java 소스 코드를 분석하고 Neo4j 그래프 데이터베이스에 저장하는 도구입니다. 이 도구는 Java 코드의 구조를 분석하고, 코드 간의 관계를 파악하며, GPT-4를 활용하여 코드에 대한 설명과 요약을 생성합니다.

## 주요 기능

- Java 소스 코드의 AST(Abstract Syntax Tree) 분석
- 패키지, 클래스, 인터페이스, 메서드 등의 구조적 분석
- 코드 간의 관계(상속, 구현, 의존성 등) 분석
- GPT-4를 활용한 코드 설명 및 요약 생성
- Neo4j 그래프 데이터베이스에 분석 결과 저장

## 요구사항

- Python 3.12 이상
- Neo4j 데이터베이스
- OpenAI API 키

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

3. 환경 변수 설정:
`.env` 파일을 생성하고 다음 환경 변수들을 설정합니다:
```
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=your_neo4j_uri
NEO4J_USER=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
```

## 사용 방법

Java 소스 코드가 있는 디렉토리를 분석하려면:

```bash
python main.py <directory_path>
```

예시:
```bash
python main.py ./src/main/java
```

## 데이터 모델

프로젝트는 다음과 같은 주요 엔티티들을 포함합니다:

- JavaPackage: 패키지 정보
- JavaClass: 클래스 정보
- JavaInterface: 인터페이스 정보
- JavaMethod: 메서드 정보
- JavaField: 필드 정보
- JavaParameter: 파라미터 정보
- JavaLocalVariable: 지역 변수 정보
- JavaEnum: 열거형 정보

각 엔티티는 Neo4j 그래프 데이터베이스에 노드로 저장되며, 엔티티 간의 관계는 엣지로 표현됩니다.