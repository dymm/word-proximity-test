package com.example.springbootapp.dto;

import java.util.Arrays;

public class EmbeddingResponse {
    private String sentence;
    private float[] embedding;
    private int dimension;
    private boolean success;
    private String error;

    public EmbeddingResponse() {
    }

    public String getSentence() {
        return sentence;
    }

    public void setSentence(String sentence) {
        this.sentence = sentence;
    }

    public float[] getEmbedding() {
        return embedding;
    }

    public void setEmbedding(float[] embedding) {
        this.embedding = embedding;
    }

    public int getDimension() {
        return dimension;
    }

    public void setDimension(int dimension) {
        this.dimension = dimension;
    }

    public boolean isSuccess() {
        return success;
    }

    public void setSuccess(boolean success) {
        this.success = success;
    }

    public String getError() {
        return error;
    }

    public void setError(String error) {
        this.error = error;
    }

    @Override
    public String toString() {
        return "EmbeddingResponse{" +
                "sentence='" + sentence + '\'' +
                ", dimension=" + dimension +
                ", success=" + success +
                ", error='" + error + '\'' +
                ", embedding=" + (embedding != null ? Arrays.toString(Arrays.copyOf(embedding, Math.min(5, embedding.length))) + "..." : "null") +
                '}';
    }
}
