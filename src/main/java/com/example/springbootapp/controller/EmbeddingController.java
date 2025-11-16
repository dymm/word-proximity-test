package com.example.springbootapp.controller;

import com.example.springbootapp.dto.EmbeddingRequest;
import com.example.springbootapp.dto.EmbeddingResponse;
import com.example.springbootapp.dto.WordMatchRequest;
import com.example.springbootapp.dto.WordMatchResponse;
import com.example.springbootapp.service.EmbeddingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/embedding")
@ConditionalOnBean(EmbeddingService.class)
public class EmbeddingController {

    private final EmbeddingService embeddingService;

    @Autowired
    public EmbeddingController(EmbeddingService embeddingService) {
        this.embeddingService = embeddingService;
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

            var matches = embeddingService.matchWords(
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
}
