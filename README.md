# Spring Boot Application

A Spring Boot 3.4.0 application with Java 21.

## Features

- Spring Boot 3.4.0 (latest version)
- Java 21
- Spring Web (REST API)
- Maven build system

## Project Structure

```
├── src/
│   ├── main/
│   │   ├── java/com/example/springbootapp/
│   │   │   ├── Application.java
│   │   │   └── controller/
│   │   │       └── HelloController.java
│   │   └── resources/
│   │       └── application.properties
│   └── test/
│       └── java/com/example/springbootapp/
│           └── ApplicationTests.java
└── pom.xml
```

## Getting Started

### Prerequisites

- Java 21 or higher
- Maven 3.6+

### Running the Application

```bash
mvn spring-boot:run
```

The application will start on `http://localhost:8080`

### Build the Application

```bash
mvn clean install
```

### Run Tests

```bash
mvn test
```

## API Endpoints

- `GET /api/hello` - Returns a greeting message
- `GET /api/health` - Returns application health status

### Example Requests

```bash
curl http://localhost:8080/api/hello
curl http://localhost:8080/api/health
```

## Configuration

Application configuration can be found in `src/main/resources/application.properties`

- Server port: 8080
- Logging level: INFO (root), DEBUG (com.example)
