# Spring Boot Semantic Word Matching Application

A Spring Boot 3.4.0 application with Java 21 that provides semantic word matching using sentence transformers (MiniLM) and ONNX Runtime.

## Features

- **Semantic Word Matching** - Find semantically similar words in text using deep learning embeddings
- **Sentence Embeddings** - Generate 384-dimensional vector representations of text
- **ONNX Runtime Integration** - Fast inference using optimized ONNX models
- **WordPiece Tokenization** - HuggingFace tokenizer for handling subword tokens
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

## How Semantic Word Matching Works

This application uses **sentence transformers** (specifically the `all-MiniLM-L6-v2` model) to perform semantic word matching. Here's how it works:

### Architecture Overview

```
Input Sentence → Tokenization → ONNX Model → Token Embeddings → Word Grouping → Similarity Matching
```

### Step-by-Step Process

1. **Tokenization**: The input sentence is tokenized using a WordPiece tokenizer, which may split words into subword tokens (e.g., "playing" → ["play", "##ing"])

2. **Embedding Generation**: The ONNX MiniLM model processes the tokens and generates 384-dimensional embedding vectors for each token

3. **Word Reconstruction**: Subword tokens are grouped back into complete words, and their embeddings are averaged to create word-level embeddings

4. **Target Embedding**: Target words (concepts to match against) are also converted to embeddings using the same model

5. **Similarity Computation**: Cosine similarity is calculated between each word embedding and target embeddings:
   - Similarity ranges from 0 (unrelated) to 1 (identical meaning)
   - Words above a threshold (e.g., 0.3-0.4) are considered matches

6. **Filtering & Ranking**: Results are filtered by:
   - Stopword exclusion (optional, default: true)
   - Minimum word length (optional, default: 3 characters)
   - Top-K nearest targets per word (configurable)

### Example

**Input:**
```json
{
  "sentence": "The quick brown fox jumps",
  "targets": ["animal", "speed", "color"],
  "threshold": 0.3
}
```

**Process:**
- "brown" embedding compared to ["animal", "speed", "color"]
  - "color" similarity: 0.32 ✓ (above threshold)
- "fox" embedding compared to targets
  - "animal" similarity: 0.38 ✓ (above threshold)

**Output:**
```json
{
  "words": [
    {"word": "brown", "matches": [{"target": "color", "score": 0.32}]},
    {"word": "fox", "matches": [{"target": "animal", "score": 0.38}]}
  ]
}
```

## API Endpoints

### Basic Endpoints

- `GET /api/hello` - Returns a greeting message
- `GET /api/health` - Returns application health status

### Embedding Endpoints

#### Generate Sentence Embedding
`POST /api/embedding`

Generate a 384-dimensional embedding vector for a sentence.

```bash
curl -X POST http://localhost:8080/api/embedding \
  -H 'Content-Type: application/json' \
  -d '{"sentence": "Hello, how are you?"}'
```

#### Match Words to Targets
`POST /api/embedding/match-words`

Find semantically similar target words for each word in a sentence.

**Request Parameters:**
- `sentence` (required): Input text to analyze
- `targets` (required): Array of target words/concepts to match against
- `threshold` (optional, default: 0.6): Minimum similarity score (0.0-1.0). Recommended: 0.3-0.4 for word-level matching
- `topK` (optional, default: 3): Maximum number of matches per word
- `aggregateSubwords` (optional, default: true): Merge WordPiece subwords into complete words
- `excludeStopwords` (optional, default: true): Skip common stopwords ("the", "and", etc.)
- `minTokenLength` (optional, default: 3): Minimum word length to consider

**Example Request:**
```bash
curl -X POST http://localhost:8080/api/embedding/match-words \
  -H 'Content-Type: application/json' \
  -d '{
    "sentence": "The quick brown fox jumps over the lazy dog",
    "targets": ["animal", "speed", "color", "sleepy"],
    "threshold": 0.3,
    "topK": 2
  }'
```

**Example Response:**
```json
{
  "sentence": "The quick brown fox jumps over the lazy dog",
  "words": [
    {
      "word": "quick",
      "tokenStart": 2,
      "tokenEnd": 2,
      "matches": [
        {"target": "speed", "score": 0.42}
      ]
    },
    {
      "word": "brown",
      "tokenStart": 3,
      "tokenEnd": 3,
      "matches": [
        {"target": "color", "score": 0.32}
      ]
    },
    {
      "word": "fox",
      "tokenStart": 4,
      "tokenEnd": 4,
      "matches": [
        {"target": "animal", "score": 0.38}
      ]
    },
    {
      "word": "dog",
      "tokenStart": 9,
      "tokenEnd": 9,
      "matches": [
        {"target": "animal", "score": 0.44}
      ]
    }
  ]
}
```

### Use Cases

- **Topic Detection**: Identify words related to specific themes in documents
- **Content Categorization**: Match document words to category keywords
- **Semantic Highlighting**: Highlight words similar to search terms
- **Keyword Extraction**: Find words relevant to given concepts
- **Text Analysis**: Understand which words align with target concepts

## Configuration

Application configuration can be found in `src/main/resources/application.properties`

- Server port: 8080
- Logging level: INFO (root), DEBUG (com.example)
- Embedding service: `embedding.enabled=true` (enables semantic matching endpoints)

## Technical Implementation

### Dependencies

- **ONNX Runtime** (1.23.2): Runs the MiniLM transformer model for fast inference
- **HuggingFace Tokenizers** (0.25.0): WordPiece tokenization compatible with the model

### Model Details

- **Model**: `all-MiniLM-L6-v2` (ONNX format)
- **Architecture**: 6-layer MiniLM (distilled BERT)
- **Embedding Dimension**: 384
- **Max Sequence Length**: 256 tokens
- **Pooling Strategy**: Mean pooling with attention mask
- **Normalization**: L2 normalization for cosine similarity

### Code Structure

The semantic matching implementation is organized in the `EmbeddingService` class:

#### Core Methods

- **`getSentenceEmbedding()`**: Generates sentence-level embeddings
  - Tokenizes input → Runs ONNX inference → Mean pools tokens → L2 normalizes

- **`getWordEmbeddings()`**: Generates word-level embeddings
  - Extracts token embeddings → Groups WordPiece subwords → Creates word vectors

- **`matchWords()`**: Main API for semantic word matching
  - Computes word embeddings → Compares to target embeddings → Filters and ranks results

#### Helper Methods

- **`runInference()`**: Executes ONNX model inference
- **`extractTokenEmbeddings()`**: Reshapes ONNX output to token matrix
- **`groupTokensIntoWords()`**: Reconstructs words from WordPiece tokens
- **`computeTargetEmbeddings()`**: Batch computes target word embeddings
- **`shouldSkipWord()`**: Applies stopword and length filters
- **`computeTargetScores()`**: Calculates cosine similarities and ranks matches
- **`cosine()`**: Computes cosine similarity between two vectors
- **`meanPooling()`**: Averages token embeddings (attention-masked)
- **`l2Normalize()`**: Normalizes vectors to unit length

### Cosine Similarity Formula

The similarity between two embedding vectors **A** and **B** is computed as:

```
similarity = (A · B) / (||A|| × ||B||)
```

Where:
- `A · B` is the dot product of vectors A and B
- `||A||` is the L2 norm (Euclidean length) of vector A
- For L2-normalized vectors, this simplifies to just the dot product

**Interpretation:**
- `1.0`: Identical semantic meaning
- `0.7-0.9`: Very similar concepts
- `0.4-0.6`: Related concepts
- `0.3-0.4`: Weakly related (recommended threshold for word matching)
- `< 0.3`: Mostly unrelated

### Performance Considerations

- **ONNX Session Reuse**: Single model instance shared across requests
- **Target Caching**: Target embeddings computed once per request
- **Sequence Padding**: Fixed 256-token length (truncated if longer)
- **CPU Inference**: Model runs on CPU (GPU optional with CUDA provider)

## Model Attribution

This application uses the `all-MiniLM-L6-v2` model from Sentence Transformers:
- **Source**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **License**: Apache 2.0
- **Paper**: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

Implementation based on the article: [Bringing Sentence Transformers to Java: Run all-MiniLM-L6-v2 with ONNX Runtime](https://medium.com/@nil.joshi860/bringing-sentence-transformers-to-java-run-all-minilm-l6-v2-with-onnx-runtime-73938447342b)

````
