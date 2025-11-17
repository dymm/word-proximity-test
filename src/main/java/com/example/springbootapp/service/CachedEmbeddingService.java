package com.example.springbootapp.service;

import ai.onnxruntime.OrtException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Decorator for EmbeddingService that adds LRU caching for sentence embeddings.
 * 
 * Features:
 * - Thread-safe LRU cache with configurable maximum size
 * - Automatic eviction of least recently used entries when cache is full
 * - Access frequency tracking for cache analytics
 * - Cache statistics (hit rate, miss rate, eviction count)
 * - Case-insensitive cache keys for better hit rate
 * 
 * Memory estimation:
 * - Each embedding: ~384 floats * 4 bytes = ~1.5KB
 * - Default cache size: 1000 entries = ~1.5MB
 * - Adjust maxCacheSize via application.properties if needed
 * 
 * Performance impact:
 * - Cache hit: O(1) lookup, no ONNX inference (~100x faster)
 * - Cache miss: O(1) lookup + ONNX inference + cache store
 * 
 * @see EmbeddingService
 */
@Service("cachedEmbeddingService")
@ConditionalOnProperty(name = "embedding.enabled", havingValue = "true", matchIfMissing = false)
public class CachedEmbeddingService extends EmbeddingService {
    
    private final Map<String, float[]> sentenceCache;
    private final Map<String, AtomicLong> accessFrequency;
    private final int maxCacheSize;
    
    // Cache statistics
    private final AtomicLong cacheHits = new AtomicLong(0);
    private final AtomicLong cacheMisses = new AtomicLong(0);
    private final AtomicLong evictions = new AtomicLong(0);
    
    /**
     * Constructor with configurable cache size.
     * 
     * @param maxCacheSize maximum number of sentence embeddings to cache (default: 1000)
     */
    @Autowired
    public CachedEmbeddingService(
            @Value("${embedding.cache.max-size:1000}") int maxCacheSize) throws Exception {
        super(); // Initialize parent EmbeddingService (ONNX session, tokenizer)
        
        this.maxCacheSize = maxCacheSize;
        this.accessFrequency = new ConcurrentHashMap<>();
        
        // Create LRU cache with access-order LinkedHashMap
        // When removeEldestEntry returns true, the eldest (LRU) entry is automatically removed
        this.sentenceCache = new LinkedHashMap<String, float[]>(maxCacheSize, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, float[]> eldest) {
                boolean shouldRemove = size() > maxCacheSize;
                if (shouldRemove) {
                    evictions.incrementAndGet();
                    // Remove from frequency map as well
                    accessFrequency.remove(eldest.getKey());
                }
                return shouldRemove;
            }
        };
    }
    
    /**
     * Get sentence embedding with LRU caching.
     * Cache keys are case-insensitive and trimmed for consistency.
     * 
     * @param sentence input text to embed
     * @return L2 normalized embedding vector
     * @throws OrtException if ONNX runtime encounters an error
     */
    @Override
    public synchronized float[] getSentenceEmbedding(String sentence) throws OrtException {
        // Normalize cache key: trim and lowercase for consistency
        String cacheKey = sentence.trim().toLowerCase();
        
        // Check cache first
        float[] cached = sentenceCache.get(cacheKey);
        
        if (cached != null) {
            // Cache hit: update statistics and frequency
            cacheHits.incrementAndGet();
            accessFrequency.computeIfAbsent(cacheKey, k -> new AtomicLong(0)).incrementAndGet();
            
            // Note: LinkedHashMap.get() automatically moves entry to end (most recently used)
            return cached;
        }
        
        // Cache miss: compute embedding via parent class
        cacheMisses.incrementAndGet();
        float[] embedding = super.getSentenceEmbedding(sentence);
        
        // Store in cache (LRU eviction happens automatically if size > maxCacheSize)
        sentenceCache.put(cacheKey, embedding);
        accessFrequency.computeIfAbsent(cacheKey, k -> new AtomicLong(0)).incrementAndGet();
        
        return embedding;
    }
    
    /**
     * Get cache statistics for monitoring and debugging.
     * 
     * @return map containing cache metrics
     */
    public Map<String, Object> getCacheStats() {
        long hits = cacheHits.get();
        long misses = cacheMisses.get();
        long total = hits + misses;
        double hitRate = total > 0 ? (double) hits / total : 0.0;
        
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("maxSize", maxCacheSize);
        stats.put("currentSize", sentenceCache.size());
        stats.put("cacheHits", hits);
        stats.put("cacheMisses", misses);
        stats.put("hitRate", String.format("%.2f%%", hitRate * 100));
        stats.put("evictions", evictions.get());
        stats.put("totalRequests", total);
        
        return stats;
    }
    
    /**
     * Get access frequency for each cached sentence.
     * Useful for understanding cache effectiveness and usage patterns.
     * 
     * @return map of sentence -> access count, sorted by frequency descending
     */
    public Map<String, Long> getAccessFrequencies() {
        Map<String, Long> frequencies = new LinkedHashMap<>();
        
        accessFrequency.entrySet().stream()
                .sorted((e1, e2) -> Long.compare(e2.getValue().get(), e1.getValue().get()))
                .forEach(entry -> frequencies.put(entry.getKey(), entry.getValue().get()));
        
        return frequencies;
    }
    
    /**
     * Clear both sentence cache and target cache.
     * Resets all statistics to zero.
     */
    public synchronized void clearAllCaches() {
        sentenceCache.clear();
        accessFrequency.clear();
        super.clearTargetCache();
        
        cacheHits.set(0);
        cacheMisses.set(0);
        evictions.set(0);
    }
    
    /**
     * Clear only the sentence cache, keeping target cache intact.
     * Resets sentence-related statistics to zero.
     */
    public synchronized void clearSentenceCache() {
        sentenceCache.clear();
        accessFrequency.clear();
        
        cacheHits.set(0);
        cacheMisses.set(0);
        evictions.set(0);
    }
    
    /**
     * Get the current size of the sentence cache.
     * 
     * @return number of cached sentence embeddings
     */
    public int getSentenceCacheSize() {
        return sentenceCache.size();
    }
}
