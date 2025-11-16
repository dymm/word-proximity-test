package simulations

import io.gatling.core.Predef._
import io.gatling.http.Predef._
import scala.concurrent.duration._

class MatchWordsSimulation extends Simulation {

  // Configuration from system properties with defaults
  val baseUrl = sys.props.getOrElse("BASE_URL", "http://localhost:8080")
  val warmupUsers = sys.props.getOrElse("WARMUP_USERS", "5").toInt
  val warmupDuration = sys.props.getOrElse("WARMUP_SEC", "30").toInt
  val rampUsers = sys.props.getOrElse("USERS_MAX", "50").toInt
  val rampDuration = sys.props.getOrElse("RAMP_SEC", "60").toInt
  val steadyDuration = sys.props.getOrElse("STEADY_SEC", "120").toInt

  // HTTP protocol configuration
  val httpProtocol = http
    .baseUrl(baseUrl)
    .acceptHeader("application/json")
    .contentTypeHeader("application/json")
    .userAgentHeader("Gatling Performance Test")

  // Feeders for test data breadth
  val sentencesFeeder = csv("data/sentences.csv").random
  val targetsFeeder = csv("data/targets.csv").random

  // Scenario: Match Words Load Test
  val matchWordsScenario = scenario("Match Words - Ramp to Peak")
    .feed(sentencesFeeder)
    .feed(targetsFeeder)
    .exec(
      http("Match Words Request")
        .post("/api/embedding/match-words")
        .body(StringBody(session => {
          val sentence = session("sentence").as[String]
          val targets = session("targets").as[String]
          val threshold = session("threshold").as[String]
          val topK = session("topK").as[String]
          
          s"""{
            |  "sentence": "$sentence",
            |  "targets": $targets,
            |  "threshold": $threshold,
            |  "topK": $topK
            |}""".stripMargin
        }))
        .check(status.is(200))
        .check(jsonPath("$.sentence").exists)
        .check(jsonPath("$.words").exists)
    )
    .pause(500.milliseconds, 2.seconds)

  // Load profile: warmup -> ramp -> steady
  setUp(
    matchWordsScenario.inject(
      // Phase 1: Warm-up to prime the ONNX model cache
      constantUsersPerSec(warmupUsers) during(warmupDuration.seconds),
      
      // Phase 2: Ramp to peak load
      rampUsersPerSec(warmupUsers) to(rampUsers) during(rampDuration.seconds),
      
      // Phase 3: Steady state at peak
      constantUsersPerSec(rampUsers) during(steadyDuration.seconds)
    )
  )
  .protocols(httpProtocol)
  .assertions(
    // Global assertions: p95 latency < 1000ms, failure rate < 1%
    global.responseTime.percentile3.lt(1000),
    global.failedRequests.percent.lt(1.0)
  )
}
