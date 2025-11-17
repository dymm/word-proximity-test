package com.example.springbootapp.service;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

@Service
@ConditionalOnProperty(name = "embedding.enabled", havingValue = "true", matchIfMissing = false)
public class EmbeddingService {
    private static final String MODELS_ALL_MINI_LM_L6_V2_TOKENIZER_JSON = "models/all-MiniLM-L6-v2-onnx/tokenizer.json";
    private static final String MODELS_ALL_MINI_LM_L6_V2_MODEL_ONNX = "models/all-MiniLM-L6-v2-onnx/model.onnx";
    private final OrtEnvironment env;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;
    
    /**
     * Cache for target word embeddings to avoid recomputation across requests.
     * Key: target word (lowercase), Value: L2-normalized embedding vector
     * Thread-safe for concurrent access.
     */
    private final Map<String, float[]> targetEmbeddingCache = new ConcurrentHashMap<>();

    private static final Set<String> STOPWORDS = Set.of(
            "a","an","the","and","or","but","if","then","else","when",
            "at","by","for","with","about","against","between","into","through","during",
            "before","after","above","below","to","from","up","down","in","out","on","off",
            "over","under","again","further","here","there","all","any","both","each","few",
            "more","most","other","some","such","no","nor","not","only","own","same","so",
            "than","too","very","can","will","just","don","should","now","is","are","was",
            "were","be","been","being","have","has","had","do","does","did","of","as","it",
            "its","you","your","yours","me","my","mine","we","our","ours","they","their",
            "theirs","he","him","his","she","her","hers","them","us","this","that","these","those"
    );

    /**
     * Constructor initializes ONNX runtime environment, loads model and tokenizer from classpath resources.
     * 
     * @throws OrtException ONNX runtime exception
     * @throws IOException tokenizer file loading exception
     */
    public EmbeddingService() throws OrtException, IOException {
        // Initialize ONNX Runtime environment
        this.env = OrtEnvironment.getEnvironment();
        
        // Load model from classpath as byte array
        byte[] modelBytes;
        try (var modelStream = getClass().getClassLoader().getResourceAsStream(MODELS_ALL_MINI_LM_L6_V2_MODEL_ONNX)) {
            if (modelStream == null) {
                throw new IOException("Model file not found in classpath: " + MODELS_ALL_MINI_LM_L6_V2_MODEL_ONNX);
            }
            modelBytes = modelStream.readAllBytes();
        }
        
        // Create ONNX session from byte array
        this.session = env.createSession(modelBytes, new OrtSession.SessionOptions());

        // Load tokenizer from classpath
        try (var tokenizerStream = getClass().getClassLoader().getResourceAsStream(MODELS_ALL_MINI_LM_L6_V2_TOKENIZER_JSON)) {
            if (tokenizerStream == null) {
                throw new IOException("Tokenizer file not found in classpath: " + MODELS_ALL_MINI_LM_L6_V2_TOKENIZER_JSON);
            }
            // HuggingFaceTokenizer needs a file path, so we need to extract to temp file
            var tempFile = java.nio.file.Files.createTempFile("tokenizer", ".json");
            java.nio.file.Files.copy(tokenizerStream, tempFile, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            // Use the Path object to ensure it's treated as a local file
            this.tokenizer = HuggingFaceTokenizer.newInstance(tempFile);
            tempFile.toFile().deleteOnExit();
        }
    }

    /**
     * Generate an embedding vector for a given sentence.
     * Steps:
     * - Tokenize input sentence to token IDs with attention masks.
     * - Pad or truncate sequences to fixed length 256.
     * - Convert inputs to ONNX tensors with shape [1, 256].
     * - Run inference to get token embeddings (shape [1, 256, embedding_dim]).
     * - Reshape flat output buffer to 2D array.
     * - Apply mean pooling using attention mask.
     * - L2 normalize the pooled embedding vector.
     * 
     * @param sentence input text to embed
     * @return L2 normalized embedding vector as float array
     * @throws OrtException ONNX runtime exception
     */
    public float[] getSentenceEmbedding(String sentence) throws OrtException {
        // Step 1: Tokenize the input sentence using HuggingFace tokenizer
        Encoding encoding = tokenizer.encode(sentence);

        // Step 2: Prepare input arrays with fixed sequence length
        long[] inputIds = padOrTruncate(encoding.getIds(), 256);
        long[] attentionMask = padOrTruncate(encoding.getAttentionMask(), 256);
        long[] tokenTypeIds = new long[256]; // All zeros for single sentence

        // Step 3: Run ONNX inference to get raw token embeddings
        float[] flatEmbeddings = runInference(inputIds, attentionMask, tokenTypeIds);

        // Step 4: Reshape flat array to 2D token embeddings matrix
        int sequenceLength = 256;
        int embeddingDim = flatEmbeddings.length / sequenceLength;
        float[][] tokenEmbeddings = extractTokenEmbeddings(flatEmbeddings, sequenceLength, embeddingDim);

        // Step 5: Apply mean pooling over tokens (considers attention mask)
        float[] pooled = meanPooling(tokenEmbeddings, attentionMask);

        // Step 6: L2 normalize to unit vector for cosine similarity comparisons
        return l2Normalize(pooled);
    }

    /**
     * Compute word-level embeddings for a sentence by aggregating subword token embeddings (WordPiece).
     * Skips special tokens ([CLS], [SEP]) and padding. Returns a list of word vectors aligned to token spans.
     */
    public List<WordEmbedding> getWordEmbeddings(String sentence, boolean aggregateSubwords) throws OrtException {
        // Step 1: Tokenize the input sentence
        Encoding encoding = tokenizer.encode(sentence);

        // Step 2: Prepare input arrays with fixed sequence length
        long[] attentionMask = padOrTruncate(encoding.getAttentionMask(), 256);
        long[] inputIds = padOrTruncate(encoding.getIds(), 256);
        long[] tokenTypeIds = new long[256];

        // Step 3: Run ONNX inference to get raw token embeddings
        float[] flatArray = runInference(inputIds, attentionMask, tokenTypeIds);

        // Step 4: Reshape flat array to 2D token embeddings matrix
        int sequenceLength = 256;
        int embeddingDim = flatArray.length / sequenceLength;
        float[][] tokenEmbeddings = extractTokenEmbeddings(flatArray, sequenceLength, embeddingDim);

        // Step 5: Get token strings and determine valid token range
        String[] tokens = encoding.getTokens();
        int validCount = 0;
        for (long v : attentionMask) if (v == 1) validCount++;
        int lastValidIndex = Math.max(0, validCount - 1); // Position of [SEP] token

        // Step 6: Group tokens into words based on aggregation strategy
        return groupTokensIntoWords(tokens, tokenEmbeddings, embeddingDim, lastValidIndex, aggregateSubwords);
    }

    public static class WordEmbedding {
        public final String word;
        public final int tokenStart;
        public final int tokenEnd;
        public final float[] vector;

        public WordEmbedding(String word, int tokenStart, int tokenEnd, float[] vector) {
            this.word = word;
            this.tokenStart = tokenStart;
            this.tokenEnd = tokenEnd;
            this.vector = vector;
        }
    }

    private float[] mean(float[][] tokenEmbeddings, int start, int end) {
        int dim = tokenEmbeddings[0].length;
        float[] out = new float[dim];
        int count = 0;
        for (int i = start; i <= end; i++) {
            float[] t = tokenEmbeddings[i];
            for (int j = 0; j < dim; j++) out[j] += t[j];
            count++;
        }
        if (count == 0) count = 1;
        for (int j = 0; j < dim; j++) out[j] /= count;
        return out;
    }

    private String reconstructWord(String[] tokens, int start, int end) {
        StringBuilder sb = new StringBuilder();
        for (int i = start; i <= end && i < tokens.length; i++) {
            String t = tokens[i];
            if (t.startsWith("##")) sb.append(t.substring(2));
            else sb.append(t);
        }
        return sb.toString();
    }

    public float cosine(float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length && i < b.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        double denom = Math.sqrt(na) * Math.sqrt(nb);
        if (denom == 0) return 0f;
        return (float) (dot / denom);
    }

    /**
     * High-level API: Find semantically similar target words for each word in the sentence.
     * This is the main entry point for word-level semantic matching.
     * 
     * Algorithm:
     * 1. Extract word-level embeddings from the input sentence
     * 2. Compute embeddings for all target words (cached for the request)
     * 3. For each sentence word, compute cosine similarity to all targets
     * 4. Filter results by threshold and apply stopword/length filters
     * 5. Return top-K matches per word, sorted by similarity score descending
     * 
     * Use cases:
     * - Topic detection: Find words related to specific themes
     * - Semantic highlighting: Identify words similar to search terms
     * - Content categorization: Match document words to category keywords
     * 
     * @param sentence input text to analyze
     * @param targets list of target words/concepts to match against
     * @param threshold minimum cosine similarity score (0.0-1.0) to include a match
     *                  Recommended: 0.3-0.4 for word-level matching
     * @param topK maximum number of target matches to return per word
     * @param aggregateSubwords if true, merge WordPiece subwords into complete words
     * @param excludeStopwords if true, skip common stopwords ("the", "and", etc.)
     * @param minTokenLength minimum word length to consider (e.g., 3 to skip "a", "in")
     * @return list of WordMatch objects containing words and their nearest target matches
     * @throws OrtException if ONNX runtime encounters an error during inference
     */
    public List<com.example.springbootapp.dto.WordMatch> matchWords(
            String sentence, List<String> targets, float threshold, int topK, boolean aggregateSubwords,
            boolean excludeStopwords, int minTokenLength) throws OrtException {

        // Step 1: Extract word-level embeddings from the sentence
        List<WordEmbedding> words = getWordEmbeddings(sentence, aggregateSubwords);

        // Step 2: Precompute target embeddings once for all words (optimization)
        List<float[]> targetVecs = computeTargetEmbeddings(targets);

        // Step 3: Match each word to targets and build result list
        List<com.example.springbootapp.dto.WordMatch> results = new ArrayList<>();
        for (WordEmbedding wordEmb : words) {
            // Step 4: Apply filtering rules (stopwords, min length)
            if (shouldSkipWord(wordEmb.word, excludeStopwords, minTokenLength)) {
                continue;
            }

            // Step 5: Compute similarity scores and filter by threshold
            List<com.example.springbootapp.dto.TargetScore> scores = 
                computeTargetScores(wordEmb.vector, targets, targetVecs, threshold, topK);

            // Step 6: Build response DTO for this word
            com.example.springbootapp.dto.WordMatch match = new com.example.springbootapp.dto.WordMatch();
            match.setWord(wordEmb.word);
            match.setTokenStart(wordEmb.tokenStart);
            match.setTokenEnd(wordEmb.tokenEnd);
            match.setMatches(scores);
            results.add(match);
        }

        return results;
    }

    /**
     * Example method demonstrating tokenization and detokenization (decode).
     * Prints token IDs and decoded string to console.
     * 
     * @param text string to encode and decode
     * @throws IOException tokenizer loading exception
     */
    public void exampleEncodeDecode(String text) throws IOException {
        // Encode text to token IDs including special tokens
        Encoding encoding = tokenizer.encode(text);

        // Retrieve token IDs as int array
        long[] tokenIds = encoding.getIds();
        System.out.println("Token IDs: " + Arrays.toString(tokenIds));

        // Decode token IDs back to string, skipping special tokens
        String decoded = tokenizer.decode(tokenIds, true);
        System.out.println("Decoded text: " + decoded);
    }

    /**
     * Mean pooling: averages token embeddings over all tokens where attentionMask == 1.
     * 
     * @param tokenEmbeddings 2D array [sequenceLength][embeddingDim]
     * @param attentionMask long array indicating valid tokens (1 = valid, 0 = pad)
     * @return pooled embedding vector
     */
    private float[] meanPooling(float[][] tokenEmbeddings, long[] attentionMask) {
        int dim = tokenEmbeddings[0].length;
        float[] pooled = new float[dim];
        int validTokens = 0;

        for (int i = 0; i < tokenEmbeddings.length; i++) {
            if (attentionMask[i] == 1) {
                for (int j = 0; j < dim; j++) {
                    pooled[j] += tokenEmbeddings[i][j];
                }
                validTokens++;
            }
        }

        // Avoid division by zero
        if (validTokens == 0) {
            validTokens = 1;
        }

        for (int j = 0; j < dim; j++) {
            pooled[j] /= validTokens;
        }
        return pooled;
    }

    /**
     * Normalize vector with L2 norm (Euclidean norm).
     * 
     * @param vector input float vector
     * @return L2 normalized vector
     */
    private float[] l2Normalize(float[] vector) {
        double sumSquares = 0.0;
        for (float v : vector) {
            sumSquares += v * v;
        }
        double norm = Math.sqrt(sumSquares);

        // Avoid division by zero
        if (norm == 0.0) {
            norm = 1.0;
        }

        float[] normalized = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = (float) (vector[i] / norm);
        }
        return normalized;
    }

    /**
     * Pad or truncate a long[] to fixed length.
     * Pads with zeros if shorter than length.
     * 
     * @param array input long array
     * @param length desired fixed length
     * @return padded or truncated long array
     */
    private long[] padOrTruncate(long[] array, int length) {
        long[] result = Arrays.copyOf(array, length);
        if (array.length < length) {
            Arrays.fill(result, array.length, length, 0L);
        }
        return result;
    }

    // ============================================================================
    // NEW EXTRACTED HELPER METHODS - WELL-DOCUMENTED AND FOCUSED
    // ============================================================================

    /**
     * Run ONNX inference with the MiniLM model to get raw token embeddings.
     * This method encapsulates all ONNX tensor creation, session execution, and output extraction.
     * 
     * @param inputIds padded token ID array [256]
     * @param attentionMask padded attention mask array [256]
     * @param tokenTypeIds token type IDs array [256], typically all zeros for single sentences
     * @return flat float array containing all token embeddings [256 * 384 = 98304 elements]
     * @throws OrtException if ONNX runtime encounters an error
     */
    private float[] runInference(long[] inputIds, long[] attentionMask, long[] tokenTypeIds) throws OrtException {
        // Define input tensor shape: batch size 1, sequence length 256
        long[] inputShape = new long[]{1, 256};

        // Create ONNX tensors using LongBuffer for efficient memory usage
        try (OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), inputShape);
             OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), inputShape);
             OnnxTensor tokenTypeIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(tokenTypeIds), inputShape)) {

            // Map input names to tensors as expected by the BERT/MiniLM ONNX model
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIdsTensor);
            inputs.put("attention_mask", attentionMaskTensor);
            inputs.put("token_type_ids", tokenTypeIdsTensor);

            // Run inference and get output tensor
            try (OrtSession.Result result = session.run(inputs)) {
                OnnxValue output = result.get(0); // First output contains token embeddings
                
                // Extract flat float array from the output tensor's FloatBuffer
                return ((OnnxTensor) output).getFloatBuffer().array();
            }
        }
    }

    /**
     * Extract and reshape token embeddings from flat ONNX output array to 2D matrix.
     * ONNX returns embeddings as a flat 1D array; this method converts it to [tokens][dimensions].
     * 
     * @param flatArray flat array from ONNX output [sequenceLength * embeddingDim]
     * @param sequenceLength number of tokens (256)
     * @param embeddingDim embedding dimension per token (384 for MiniLM)
     * @return 2D array of token embeddings [sequenceLength][embeddingDim]
     */
    private float[][] extractTokenEmbeddings(float[] flatArray, int sequenceLength, int embeddingDim) {
        float[][] tokenEmbeddings = new float[sequenceLength][embeddingDim];
        
        // Reshape: flat[i*dim : (i+1)*dim] -> tokenEmbeddings[i][:]
        for (int i = 0; i < sequenceLength; i++) {
            System.arraycopy(flatArray, i * embeddingDim, tokenEmbeddings[i], 0, embeddingDim);
        }
        
        return tokenEmbeddings;
    }

    /**
     * Group WordPiece tokens into complete words and compute word embeddings.
     * This method implements the core logic for reconstructing words from subword tokens.
     * 
     * WordPiece tokenization strategy:
     * - Tokens starting with \"##\" are continuations of the previous token
     * - Example: \"playing\" -> [\"play\", \"##ing\"]
     * 
     * @param tokens array of token strings from the tokenizer
     * @param tokenEmbeddings 2D array of token-level embeddings
     * @param embeddingDim dimension of embedding vectors
     * @param lastValidIndex index of the [SEP] token (exclusive upper bound)
     * @param aggregateSubwords if true, merge subwords; if false, one word per token
     * @return list of WordEmbedding objects with reconstructed words and their embeddings
     */
    private List<WordEmbedding> groupTokensIntoWords(String[] tokens, float[][] tokenEmbeddings, 
                                                     int embeddingDim, int lastValidIndex, 
                                                     boolean aggregateSubwords) {
        List<WordEmbedding> words = new ArrayList<>();

        if (!aggregateSubwords) {
            // Simple mode: treat each token as a separate word
            for (int i = 1; i < lastValidIndex && i < tokens.length; i++) { // Skip [CLS] at index 0
                String tok = tokens[i];
                String piece = tok.startsWith("##") ? tok.substring(2) : tok;
                float[] vec = l2Normalize(Arrays.copyOf(tokenEmbeddings[i], embeddingDim));
                words.add(new WordEmbedding(piece, i, i, vec));
            }
            return words;
        }

        // Complex mode: aggregate WordPiece subwords into complete words
        List<int[]> spans = new ArrayList<>(); // Track [start, end] ranges for each word
        StringBuilder currentWord = new StringBuilder();
        int wordStart = -1;
        int i = 1; // Start after [CLS] at index 0

        // Scan tokens and group into words based on "##" prefix
        while (i < lastValidIndex && i < tokens.length) {
            String tok = tokens[i];
            boolean isContinuation = tok.startsWith("##");
            String piece = isContinuation ? tok.substring(2) : tok;

            if (currentWord.length() == 0) {
                // Start a new word
                currentWord.append(piece);
                wordStart = i;
            } else if (isContinuation) {
                // Continue current word
                currentWord.append(piece);
            } else {
                // Flush previous word and start new one
                spans.add(new int[]{wordStart, i - 1});
                currentWord.setLength(0);
                currentWord.append(piece);
                wordStart = i;
            }
            i++;
        }

        // Flush final word if any
        if (currentWord.length() > 0 && wordStart >= 1) {
            spans.add(new int[]{wordStart, Math.min(lastValidIndex - 1, tokens.length - 1)});
        }

        // Compute word embeddings by averaging token embeddings in each span
        for (int[] span : spans) {
            int start = span[0];
            int end = span[1];
            
            // Average token embeddings across the span
            float[] wordVec = mean(tokenEmbeddings, start, end);
            wordVec = l2Normalize(wordVec);
            
            // Reconstruct word text from tokens
            String word = reconstructWord(tokens, start, end);
            
            words.add(new WordEmbedding(word, start, end, wordVec));
        }

        return words;
    }

    /**
     * Compute embeddings for all target words with caching optimization.
     * Uses a thread-safe cache to avoid recomputing embeddings for frequently used targets.
     * Cache keys are case-insensitive (lowercase) for better hit rate.
     * 
     * Performance impact:
     * - Cache hit: O(1) lookup, no ONNX inference
     * - Cache miss: O(1) lookup + ONNX inference + cache store
     * 
     * @param targets list of target word strings
     * @return list of L2-normalized embedding vectors, one per target
     * @throws OrtException if ONNX runtime encounters an error
     */
    private List<float[]> computeTargetEmbeddings(List<String> targets) throws OrtException {
        List<float[]> targetVecs = new ArrayList<>();
        
        // Compute or retrieve cached embedding for each target word
        for (String target : targets) {
            // Normalize to lowercase for cache key consistency
            String cacheKey = target.toLowerCase();
            
            // Try cache first
            float[] vec = targetEmbeddingCache.get(cacheKey);
            
            if (vec == null) {
                // Cache miss: compute embedding and store in cache
                vec = getSentenceEmbedding(target);
                targetEmbeddingCache.put(cacheKey, vec);
            }
            
            targetVecs.add(vec);
        }
        
        return targetVecs;
    }
    
    /**
     * Clear the target embedding cache. Useful for testing or memory management.
     * Thread-safe operation.
     */
    public void clearTargetCache() {
        targetEmbeddingCache.clear();
    }
    
    /**
     * Get the current size of the target embedding cache.
     * Useful for monitoring and debugging cache effectiveness.
     * 
     * @return number of cached target embeddings
     */
    public int getTargetCacheSize() {
        return targetEmbeddingCache.size();
    }

    /**
     * Determine if a word should be skipped based on filtering criteria.
     * 
     * @param word the word text to check
     * @param excludeStopwords if true, filter out common stopwords
     * @param minTokenLength minimum word length threshold (0 = no minimum)
     * @return true if word should be skipped, false if it should be included
     */
    private boolean shouldSkipWord(String word, boolean excludeStopwords, int minTokenLength) {
        String lowerWord = word.toLowerCase();
        
        // Check stopword filter
        if (excludeStopwords && STOPWORDS.contains(lowerWord)) {
            return true;
        }
        
        // Check minimum length filter
        if (minTokenLength > 0 && lowerWord.length() < minTokenLength) {
            return true;
        }
        
        return false;
    }

    /**
     * Compute similarity scores between a word embedding and all target embeddings.
     * Filters by threshold and returns top-K matches sorted by score descending.
     * 
     * @param wordVector the word's embedding vector
     * @param targets list of target word strings (for labels)
     * @param targetVecs list of target embedding vectors (parallel to targets)
     * @param threshold minimum cosine similarity to include in results
     * @param topK maximum number of results to return
     * @return list of TargetScore objects, sorted by score descending, limited to topK
     */
    private List<com.example.springbootapp.dto.TargetScore> computeTargetScores(
            float[] wordVector, List<String> targets, List<float[]> targetVecs, 
            float threshold, int topK) {
        
        List<com.example.springbootapp.dto.TargetScore> scores = new ArrayList<>();
        
        // Compute cosine similarity to each target
        for (int i = 0; i < targets.size(); i++) {
            float similarity = cosine(wordVector, targetVecs.get(i));
            
            // Filter by threshold
            if (similarity >= threshold) {
                scores.add(new com.example.springbootapp.dto.TargetScore(targets.get(i), similarity));
            }
        }
        
        // Sort by score descending (highest similarity first)
        scores.sort(Comparator.comparing(com.example.springbootapp.dto.TargetScore::getScore).reversed());
        
        // Limit to top-K results
        if (scores.size() > topK) {
            scores = new ArrayList<>(scores.subList(0, topK));
        }
        
        return scores;
    }
}
