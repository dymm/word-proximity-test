package com.example.springbootapp.dto;

public class EmbeddingRequest {
    private String sentence;

    public EmbeddingRequest() {
    }

    public EmbeddingRequest(String sentence) {
        this.sentence = sentence;
    }

    public String getSentence() {
        return sentence;
    }

    public void setSentence(String sentence) {
        this.sentence = sentence;
    }
}
