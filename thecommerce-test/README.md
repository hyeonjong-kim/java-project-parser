# 더커머스 과제 테스트

## 1. 환경

OS:  Ubuntu 20.0.4

Java: 1.8

Spring boot: 2.7.8

Docker: 26.0.2

PostgreSQL: 12.18

## 2. 배포

### 2.1  On-premise

```bash
git clone https://github.com/hyeonjong-kim/thecommerce-test.git

cd thecommerce-test

chmod 755 gradlew

./gradlew build --exclude-task test

//front ground
java -jar build/libs/test-0.0.1-SNAPSHOT.jar --server.port=${서버 포트} --spring.profiles.active=prod --spring.datasource.url=${DB 접속 정보} --spring.datasource.username=${DB 접속 아이디} --spring.datasource.password=${DB 접속 비밀번호} --jpa.properties.hibernate.default_schema=${PostgreSQL 스키마}

//back ground
nohup java -jar build/libs/test-0.0.1-SNAPSHOT.jar --server.port=${서버 포트} --spring.profiles.active=prod --spring.datasource.url=${DB 접속 정보} --spring.datasource.username=${DB 접속 아이디} --spring.datasource.password=${DB 접속 비밀번호} --jpa.properties.hibernate.default_schema=${PostgreSQL 스키마} &
```

### 2.2  Docker

```bash
git clone https://github.com/hyeonjong-kim/thecommerce-test.git

cd thecommerce-test

chmod 755 gradlew

./gradlew build --exclude-task test

//도커파일 수정
vi Dockerfile

FROM openjdk:8
ARG JAR_FILE=build/libs/*.jar
COPY ${JAR_FILE} app.jar
ENTRYPOINT ["java","-jar","/app.jar"]
// 환경에 맞게 수정
CMD ["--spring.profiles.active=prod", "--server.port=8080", "--spring.datasource.url=${DB 접속 정보}", "--spring.datasource.username=${DB 접속 아이디}", "--spring.datasource.password=${DB 접속 비밀번호}", "--jpa.properties.hibernate.default_schema=${PostgreSQL 스키마}"]

docker build -t ${이미지 이름} ./
 
docker run -d -p ${사용 포트}:8080 ${이미지 이름}
```

## 3. API 명세 접속 방법

→ http://${host}:${port}/swagger-ui/index.html
