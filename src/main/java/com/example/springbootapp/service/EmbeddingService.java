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

@Service
@ConditionalOnProperty(name = "embedding.enabled", havingValue = "true", matchIfMissing = false)
public class EmbeddingService {
    private static final String MODELS_ALL_MINI_LM_L6_V2_TOKENIZER_JSON = "models/all-MiniLM-L6-v2-onnx/tokenizer.json";
    private static final String MODELS_ALL_MINI_LM_L6_V2_MODEL_ONNX = "models/all-MiniLM-L6-v2-onnx/model.onnx";
    private final OrtEnvironment env;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;

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
        // Tokenize input sentence, returning token IDs and masks
        Encoding encoding = tokenizer.encode(sentence);

        // Pad or truncate token arrays to length 256
        long[] inputIds = padOrTruncate(encoding.getIds(), 256);
        long[] attentionMask = padOrTruncate(encoding.getAttentionMask(), 256);
        // tokenTypeIds usually zero for single sentence inputs
        long[] tokenTypeIds = new long[256];

        // Define input tensor shape: batch size 1, sequence length 256
        long[] inputShape = new long[]{1, 256};

        // Create OnnxTensor inputs using LongBuffer for efficient memory usage
        try (OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), inputShape);
             OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), inputShape);
             OnnxTensor tokenTypeIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(tokenTypeIds), inputShape)) {

            // Map input names to tensors as expected by ONNX model
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIdsTensor);
            inputs.put("attention_mask", attentionMaskTensor);
            inputs.put("token_type_ids", tokenTypeIdsTensor);

            // Run inference session
            try (OrtSession.Result result = session.run(inputs)) {
                // Get output tensor (usually first output)
                OnnxValue output = result.get(0);

                // Retrieve output as flat float array from FloatBuffer
                float[] flatArray = ((OnnxTensor) output).getFloatBuffer().array();

                // Calculate sequence length and embedding dimension
                int sequenceLength = 256;
                int embeddingDim = flatArray.length / sequenceLength;

                // Reshape flat output [sequenceLength * embeddingDim] -> 2D [sequenceLength][embeddingDim]
                float[][] tokenEmbeddings = new float[sequenceLength][embeddingDim];
                for (int i = 0; i < sequenceLength; i++) {
                    System.arraycopy(flatArray, i * embeddingDim, tokenEmbeddings[i], 0, embeddingDim);
                }

                // Apply mean pooling over tokens, considering attention mask
                float[] pooled = meanPooling(tokenEmbeddings, attentionMask);

                // Return L2 normalized pooled vector
                return l2Normalize(pooled);
            }
        }
    }

    /**
     * Compute word-level embeddings for a sentence by aggregating subword token embeddings (WordPiece).
     * Skips special tokens ([CLS], [SEP]) and padding. Returns a list of word vectors aligned to token spans.
     */
    public List<WordEmbedding> getWordEmbeddings(String sentence, boolean aggregateSubwords) throws OrtException {
        Encoding encoding = tokenizer.encode(sentence);

        long[] attentionMask = padOrTruncate(encoding.getAttentionMask(), 256);
        long[] inputIds = padOrTruncate(encoding.getIds(), 256);
        long[] tokenTypeIds = new long[256];

        long[] inputShape = new long[]{1, 256};

        try (OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), inputShape);
             OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), inputShape);
             OnnxTensor tokenTypeIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(tokenTypeIds), inputShape)) {

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIdsTensor);
            inputs.put("attention_mask", attentionMaskTensor);
            inputs.put("token_type_ids", tokenTypeIdsTensor);

            try (OrtSession.Result result = session.run(inputs)) {
                OnnxValue output = result.get(0);
                float[] flatArray = ((OnnxTensor) output).getFloatBuffer().array();

                int sequenceLength = 256;
                int embeddingDim = flatArray.length / sequenceLength;

                float[][] tokenEmbeddings = new float[sequenceLength][embeddingDim];
                for (int i = 0; i < sequenceLength; i++) {
                    System.arraycopy(flatArray, i * embeddingDim, tokenEmbeddings[i], 0, embeddingDim);
                }

                // tokens including special tokens
                String[] tokens = encoding.getTokens();
                int validCount = 0;
                for (long v : attentionMask) if (v == 1) validCount++;
                int lastValidIndex = Math.max(0, validCount - 1); // likely SEP

                List<WordEmbedding> words = new ArrayList<>();

                int i = 1; // skip CLS at 0
                StringBuilder currentWord = new StringBuilder();
                List<int[]> spans = new ArrayList<>(); // temp spans for ranges
                int wordStart = -1;

                while (i < lastValidIndex && i < tokens.length) {
                    String tok = tokens[i];
                    boolean isSub = tok.startsWith("##");

                    String piece = isSub ? tok.substring(2) : tok;

                    if (!aggregateSubwords) {
                        // create one word per token
                        String w = piece;
                        float[] vec = l2Normalize(Arrays.copyOf(tokenEmbeddings[i], embeddingDim));
                        words.add(new WordEmbedding(w, i, i, vec));
                        i++;
                        continue;
                    }

                    if (currentWord.length() == 0) {
                        currentWord.append(piece);
                        wordStart = i;
                    } else if (isSub) {
                        currentWord.append(piece);
                    } else {
                        // flush previous word
                        spans.add(new int[]{wordStart, i - 1});
                        currentWord.setLength(0);
                        currentWord.append(piece);
                        wordStart = i;
                    }
                    i++;
                }

                // flush tail word if any
                if (aggregateSubwords && currentWord.length() > 0 && wordStart >= 1) {
                    spans.add(new int[]{wordStart, Math.min(lastValidIndex - 1, tokens.length - 1)});
                }

                if (aggregateSubwords) {
                    for (int[] span : spans) {
                        int s = span[0];
                        int e = span[1];
                        float[] vec = mean(tokenEmbeddings, s, e);
                        vec = l2Normalize(vec);
                        String word = reconstructWord(tokens, s, e);
                        words.add(new WordEmbedding(word, s, e, vec));
                    }
                }

                return words;
            }
        }
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
     * High-level API: find nearest target words for each sentence word.
     */
    public List<com.example.springbootapp.dto.WordMatch> matchWords(
            String sentence, List<String> targets, float threshold, int topK, boolean aggregateSubwords,
            boolean excludeStopwords, int minTokenLength) throws OrtException {

        List<WordEmbedding> words = getWordEmbeddings(sentence, aggregateSubwords);

        // Precompute target embeddings (pooled sentence embeddings for single words)
        List<float[]> targetVecs = new ArrayList<>();
        for (String t : targets) {
            float[] v = getSentenceEmbedding(t);
            targetVecs.add(v);
        }

        List<com.example.springbootapp.dto.WordMatch> out = new ArrayList<>();
        for (WordEmbedding w : words) {
            String lw = w.word.toLowerCase();
            if (excludeStopwords && STOPWORDS.contains(lw)) {
                continue;
            }
            if (minTokenLength > 0 && lw.length() < minTokenLength) {
                continue;
            }

            List<com.example.springbootapp.dto.TargetScore> scores = new ArrayList<>();
            for (int i = 0; i < targets.size(); i++) {
                float s = cosine(w.vector, targetVecs.get(i));
                if (s >= threshold) {
                    scores.add(new com.example.springbootapp.dto.TargetScore(targets.get(i), s));
                }
            }
            scores.sort(Comparator.comparing(com.example.springbootapp.dto.TargetScore::getScore).reversed());
            if (scores.size() > topK) {
                scores = new ArrayList<>(scores.subList(0, topK));
            }

            com.example.springbootapp.dto.WordMatch wm = new com.example.springbootapp.dto.WordMatch();
            wm.setWord(w.word);
            wm.setTokenStart(w.tokenStart);
            wm.setTokenEnd(w.tokenEnd);
            wm.setMatches(scores);
            out.add(wm);
        }

        return out;
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

    /**
     * Main method to test encoding and embedding generation.
     */
    public static void main(String[] args) throws Exception {
        EmbeddingService service = new EmbeddingService();

        String sentence = "Hello, how are you?";
        // Generate embedding vector
        float[] embedding = service.getSentenceEmbedding(sentence);
        System.out.println("Embedding vector length: " + embedding.length);

        // Demonstrate encode-decode
        service.exampleEncodeDecode(sentence);
    }
}
