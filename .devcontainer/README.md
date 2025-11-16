# Java 21 Development Container

This devcontainer provides a complete Java 21 development environment with:

- **Java 21** (Microsoft OpenJDK)
- **Maven** (latest)
- **Gradle 8.5**
- **PostgreSQL 16** database

## Database Connection

The PostgreSQL database is automatically started and accessible at:
- Host: `localhost`
- Port: `5432`
- Database: `devdb`
- Username: `postgres`
- Password: `postgres`

## Environment Variables

The following environment variables are pre-configured:
- `POSTGRES_HOST=localhost`
- `POSTGRES_PORT=5432`
- `POSTGRES_DB=devdb`
- `POSTGRES_USER=postgres`
- `POSTGRES_PASSWORD=postgres`

## Getting Started

1. Open this folder in VS Code
2. When prompted, click "Reopen in Container"
3. Wait for the container to build and start
4. You're ready to develop!

## PostgreSQL Client

You can connect to the database using:
```bash
psql -h localhost -U postgres -d devdb
```

Password: `postgres`
