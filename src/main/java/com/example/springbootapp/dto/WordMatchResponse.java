package com.example.springbootapp.dto;

import java.util.List;

public class WordMatchResponse {
    private String sentence;
    private List<WordMatch> words;

    public String getSentence() {
        return sentence;
    }

    public void setSentence(String sentence) {
        this.sentence = sentence;
    }

    public List<WordMatch> getWords() {
        return words;
    }

    public void setWords(List<WordMatch> words) {
        this.words = words;
    }
}
