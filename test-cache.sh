#!/bin/bash

# Test script to demonstrate sentence embedding caching with LRU eviction
# This script sends multiple requests and monitors cache performance

BASE_URL="http://localhost:8080"

echo "=== Testing Sentence Embedding Cache with LRU Eviction ==="
echo ""

# Function to make a request
make_request() {
    local sentence="$1"
    echo "Request: \"$sentence\""
    curl -s -X POST "$BASE_URL/api/embedding/match-words" \
        -H 'Content-Type: application/json' \
        -d "{\"sentence\": \"$sentence\", \"targets\": [\"animal\", \"speed\"], \"threshold\": 0.3}" \
        > /dev/null
    echo "✓ Completed"
}

# Function to show cache stats
show_stats() {
    echo ""
    echo "--- Cache Statistics ---"
    curl -s "$BASE_URL/api/embedding/cache-stats" | jq '.'
    echo ""
}

# Clear cache to start fresh
echo "Clearing cache..."
curl -s -X POST "$BASE_URL/api/embedding/clear-cache" > /dev/null
echo "✓ Cache cleared"
echo ""

# Test 1: First request (cache miss)
echo "Test 1: First request - should be CACHE MISS"
make_request "The quick brown fox jumps"
show_stats

# Test 2: Same request (cache hit)
echo "Test 2: Same request - should be CACHE HIT"
make_request "The quick brown fox jumps"
show_stats

# Test 3: Different requests
echo "Test 3: Adding different sentences to cache"
make_request "Hello world"
make_request "Machine learning is fascinating"
make_request "Spring Boot is awesome"
show_stats

# Test 4: Repeat previous sentence (cache hit)
echo "Test 4: Repeating 'Hello world' - should be CACHE HIT"
make_request "Hello world"
make_request "Hello world"
make_request "Hello world"
show_stats

# Test 5: Access frequencies
echo "Test 5: Access frequency tracking"
echo "--- Access Frequencies ---"
curl -s "$BASE_URL/api/embedding/access-frequencies" | jq '.'
echo ""

# Test 6: Multiple cache hits
echo "Test 6: Testing cache hit performance"
echo "Sending 5 identical requests..."
for i in {1..5}; do
    make_request "The quick brown fox jumps"
done
show_stats

echo "=== Test Complete ==="
echo ""
echo "Expected results:"
echo "- Hit rate should increase with repeated requests"
echo "- 'hello world' and 'the quick brown fox jumps' should have highest access count"
echo "- Cache size should grow up to max-size (1000 by default)"
