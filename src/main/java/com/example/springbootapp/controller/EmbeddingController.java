package com.example.springbootapp.controller;

import com.example.springbootapp.dto.EmbeddingRequest;
import com.example.springbootapp.dto.EmbeddingResponse;
import com.example.springbootapp.dto.WordMatchRequest;
import com.example.springbootapp.dto.WordMatchResponse;
import com.example.springbootapp.service.CachedEmbeddingService;
import com.example.springbootapp.service.EmbeddingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/embedding")
@ConditionalOnBean(EmbeddingService.class)
public class EmbeddingController {

    private final EmbeddingService embeddingService;
    private final CachedEmbeddingService cachedService;

    @Autowired
    public EmbeddingController(
            @Qualifier("cachedEmbeddingService") EmbeddingService embeddingService) {
        this.embeddingService = embeddingService;
        // Cast to CachedEmbeddingService for cache management methods
        this.cachedService = embeddingService instanceof CachedEmbeddingService 
                ? (CachedEmbeddingService) embeddingService 
                : null;
    }

    /**
     * Generate an embedding vector for a given sentence.
     * Example request: POST /api/embedding
     * Body: { "sentence": "Hello, how are you?" }
     * 
     * @param request contains the sentence to embed
     * @return embedding response with vector and metadata
     */
    @PostMapping
    public ResponseEntity<EmbeddingResponse> getEmbedding(@RequestBody EmbeddingRequest request) {
        try {
            String sentence = request.getSentence();
            
            // Generate embedding vector using the service
            float[] embedding = embeddingService.getSentenceEmbedding(sentence);
            
            // Create response
            EmbeddingResponse response = new EmbeddingResponse();
            response.setSentence(sentence);
            response.setEmbedding(embedding);
            response.setDimension(embedding.length);
            response.setSuccess(true);
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            // Handle errors gracefully
            EmbeddingResponse errorResponse = new EmbeddingResponse();
            errorResponse.setSentence(request.getSentence());
            errorResponse.setSuccess(false);
            errorResponse.setError(e.getMessage());
            
            return ResponseEntity.internalServerError().body(errorResponse);
        }
    }

    /**
     * Test endpoint to demonstrate tokenization encode/decode.
     * Example request: POST /api/embedding/test
     * Body: { "sentence": "Hello, how are you?" }
     * 
     * @param request contains the sentence to test
     * @return simple success message
     */
    @PostMapping("/test")
    public ResponseEntity<String> testEncodeDecode(@RequestBody EmbeddingRequest request) {
        try {
            embeddingService.exampleEncodeDecode(request.getSentence());
            return ResponseEntity.ok("Encode/decode test completed. Check console logs for output.");
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error: " + e.getMessage());
        }
    }

    /**
     * Match sentence words to nearest targets using semantic similarity.
     * Example: POST /api/embedding/match-words
     * Body: { "sentence": "The quick brown fox", "targets": ["animal", "speed", "color"], "threshold": 0.6, "topK": 2 }
     */
    @PostMapping("/match-words")
    public ResponseEntity<?> matchWords(@RequestBody WordMatchRequest request) {
        try {
            if (request.getSentence() == null || request.getSentence().isBlank()) {
                return ResponseEntity.badRequest().body("Sentence must not be empty");
            }
            if (request.getTargets() == null || request.getTargets().isEmpty()) {
                return ResponseEntity.badRequest().body("Targets must not be empty");
            }

            float threshold = request.getThreshold() != null ? request.getThreshold() : 0.6f;
            int topK = request.getTopK() != null ? request.getTopK() : 3;
            boolean aggregateSubwords = request.getAggregateSubwords() != null ? request.getAggregateSubwords() : true;

            var matches = cachedService.matchWords(
                    request.getSentence(), request.getTargets(), threshold, topK, aggregateSubwords,
                    request.getExcludeStopwords() != null ? request.getExcludeStopwords() : true,
                    request.getMinTokenLength() != null ? request.getMinTokenLength() : 3);

            WordMatchResponse resp = new WordMatchResponse();
            resp.setSentence(request.getSentence());
            resp.setWords(matches);
            return ResponseEntity.ok(resp);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error: " + e.getMessage());
        }
    }

    /**
     * Get cache statistics for monitoring performance.
     * Example: GET /api/embedding/cache-stats
     * Returns: { "maxSize": 1000, "currentSize": 42, "hitRate": "87.50%", ... }
     */
    @GetMapping("/cache-stats")
    public ResponseEntity<?> getCacheStats() {
        if (cachedService == null) {
            return ResponseEntity.ok(Map.of("message", "Caching not enabled"));
        }
        return ResponseEntity.ok(cachedService.getCacheStats());
    }

    /**
     * Get access frequency statistics for cached sentences.
     * Example: GET /api/embedding/access-frequencies
     * Returns: { "hello world": 15, "test sentence": 8, ... }
     */
    @GetMapping("/access-frequencies")
    public ResponseEntity<?> getAccessFrequencies() {
        if (cachedService == null) {
            return ResponseEntity.ok(Map.of("message", "Caching not enabled"));
        }
        return ResponseEntity.ok(cachedService.getAccessFrequencies());
    }

    /**
     * Clear all caches (sentence and target embeddings).
     * Example: POST /api/embedding/clear-cache
     */
    @PostMapping("/clear-cache")
    public ResponseEntity<?> clearCache() {
        if (cachedService != null) {
            cachedService.clearAllCaches();
            return ResponseEntity.ok(Map.of("message", "All caches cleared"));
        }
        embeddingService.clearTargetCache();
        return ResponseEntity.ok(Map.of("message", "Target cache cleared"));
    }

    /**
     * Clear only the sentence cache, keeping target cache.
     * Example: POST /api/embedding/clear-sentence-cache
     */
    @PostMapping("/clear-sentence-cache")
    public ResponseEntity<?> clearSentenceCache() {
        if (cachedService == null) {
            return ResponseEntity.ok(Map.of("message", "Sentence caching not enabled"));
        }
        cachedService.clearSentenceCache();
        return ResponseEntity.ok(Map.of("message", "Sentence cache cleared"));
    }
}
