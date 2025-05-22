FROM openjdk:8
ARG JAR_FILE=build/libs/*.jar
COPY ${JAR_FILE} app.jar
ENTRYPOINT ["java","-jar","/app.jar"]
CMD ["--spring.profiles.active=prod", "--server.port=8080", "--spring.datasource.url=${DB 접속 정보}", "--spring.datasource.username=${DB 접속 아이디}", "--spring.datasource.password=${DB 접속 비밀번호}", "--jpa.properties.hibernate.default_schema=${PostgreSQL 스키마}"]