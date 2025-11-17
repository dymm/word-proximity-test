# Performance Tests - Match Words API

Gatling-based performance tests for the `/api/embedding/match-words` endpoint, designed to measure latency and throughput under load with varying sentence lengths and target list sizes.

## Overview

This module uses **Gatling 3.10.3** to execute a ramp-to-peak load profile:

1. **Warm-up Phase**: Low constant load to prime the ONNX model cache
2. **Ramp Phase**: Gradual increase from warm-up to peak concurrency
3. **Steady Phase**: Sustained peak load to measure stable performance

### Test Data Breadth

- **Sentences**: Short (~10 words), medium (~20-30 words), long (~50+ words)
- **Target Lists**: Small (2-3 words), medium (5-8 words), large (10+ words)
- **Parameters**: Varied `threshold` (0.3-0.45) and `topK` (2-5)

### Performance Assertions

- **p95 Latency**: < 1000ms
- **Failure Rate**: < 1%

## Prerequisites

1. **Java 21** (or compatible JDK)
2. **Maven 3.6+**
3. **Running Application**: The Spring Boot app must be running at `http://localhost:8080` (or override with `BASE_URL`)

## Usage

### Start the Application

From the workspace root:

```bash
mvn clean package
java -jar target/spring-boot-app-0.0.1-SNAPSHOT.jar
```

Or if using Spring Boot plugin:

```bash
mvn spring-boot:run
```

### Run Performance Tests (Default Configuration)

From the workspace root:

```bash
mvn -f performance-tests/pom.xml io.gatling:gatling-maven-plugin:test
```

Default profile:
- Warm-up: 5 users for 30 seconds
- Ramp: 5 → 20 users over 60 seconds
- Steady: 20 users for 120 seconds

### Customize Load Profile

Override via system properties:

```bash
mvn -pl performance-tests io.gatling:gatling-maven-plugin:test \
  -DBASE_URL=http://localhost:8080 \
  -DWARMUP_USERS=5 \
  -DWARMUP_SEC=20 \
  -DUSERS_MAX=20 \
  -DRAMP_SEC=90 \
  -DSTEADY_SEC=180
```

### Configuration Parameters

| Parameter      | Default               | Description                          |
|----------------|-----------------------|--------------------------------------|
| `BASE_URL`     | `http://localhost:8080` | Base URL of the Spring Boot app     |
| `WARMUP_USERS` | `5`                   | Concurrent users during warm-up      |
| `WARMUP_SEC`   | `30`                  | Warm-up phase duration (seconds)     |
| `USERS_MAX`    | `20`                  | Peak concurrent users                |
| `RAMP_SEC`     | `60`                  | Ramp-up duration (seconds)           |
| `STEADY_SEC`   | `120`                 | Steady state duration (seconds)      |

### Example: Quick Smoke Test

```bash
mvn -pl performance-tests io.gatling:gatling-maven-plugin:test \
  -DWARMUP_USERS=2 \
  -DWARMUP_SEC=10 \
  -DUSERS_MAX=10 \
  -DRAMP_SEC=20 \
  -DSTEADY_SEC=30
```

### Example: High Load Scenario

```bash
mvn -pl performance-tests io.gatling:gatling-maven-plugin:test \
  -DUSERS_MAX=20 \
  -DRAMP_SEC=120 \
  -DSTEADY_SEC=300
```

## Interpreting Results

After execution, Gatling generates an HTML report in:

```
performance-tests/target/gatling/matchwords-<timestamp>/index.html
```

### Key Metrics

1. **Response Time Percentiles**
   - p50 (median): Expected typical latency
   - p95: 95% of requests faster than this (assertion: < 1000ms)
   - p99: Slowest 1% threshold

2. **Requests per Second (RPS)**
   - Throughput achieved during steady state
   - Correlate with CPU/memory usage on server

3. **Success Rate**
   - Should be ≥ 99% (assertion: failures < 1%)
   - 4xx: Client errors (malformed requests)
   - 5xx: Server errors (ONNX inference failures, OOM, etc.)

4. **Response Time Distribution**
   - Histogram showing latency spread
   - Look for bimodal distributions (cold vs warm cache)

### Common Bottlenecks

- **High p95/p99 latency**: ONNX inference time scales with sentence length and number of targets (1 + N inferences per request)
- **Increasing errors at high load**: Thread pool exhaustion, heap pressure, GC pauses
- **Throughput plateau**: CPU saturation (ONNX runs on CPU by default)

## Optimizing Performance

If tests fail assertions:

1. **Reduce Load**: Lower `USERS_MAX` or increase `RAMP_SEC`
2. **Server Tuning**: Increase JVM heap (`-Xmx`), adjust thread pool size
3. **Model Optimization**: Use ONNX optimization, quantization, or GPU acceleration
4. **Caching**: Implement target embedding cache in `EmbeddingService`
5. **Batching**: Batch target inferences instead of sequential execution

## CI Integration

Example GitLab CI / GitHub Actions step:

```yaml
performance-test:
  stage: test
  script:
    - mvn spring-boot:start -Dspring-boot.run.fork=true
    - sleep 10
    - mvn -pl performance-tests io.gatling:gatling-maven-plugin:test -DUSERS_MAX=20 -DSTEADY_SEC=60
    - mvn spring-boot:stop
  artifacts:
    paths:
      - performance-tests/target/gatling/
    when: always
```

## Troubleshooting

### "Connection refused" errors

Ensure the Spring Boot app is running and accessible at `BASE_URL`.

### OutOfMemoryError during test

Increase Maven memory:

```bash
export MAVEN_OPTS="-Xmx2g"
mvn -pl performance-tests io.gatling:gatling-maven-plugin:test
```

### Assertions fail consistently

Review the HTML report to identify bottlenecks. Consider:
- Is the server under-resourced?
- Are sentence/target combinations too complex?
- Is the warm-up phase long enough to prime caches?

## Test Data

- **Sentences**: `src/test/resources/data/sentences.csv` (13 entries, mixed lengths)
- **Targets**: `src/test/resources/data/targets.csv` (13 entries, small/medium/large lists)

Feeders use **random** selection, so each request gets a varied combination.

## Further Reading

- [Gatling Documentation](https://gatling.io/docs/gatling/)
- [Load Profile Injection](https://gatling.io/docs/gatling/reference/current/core/injection/)
- [HTTP Protocol Reference](https://gatling.io/docs/gatling/reference/current/http/protocol/)

---

**Maintained by**: Performance Engineering Team  
**Last Updated**: November 16, 2025
