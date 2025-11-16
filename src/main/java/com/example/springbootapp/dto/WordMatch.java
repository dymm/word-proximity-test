package com.example.springbootapp.dto;

import java.util.List;

public class WordMatch {
    private String word;
    private int tokenStart; // inclusive
    private int tokenEnd;   // inclusive
    private List<TargetScore> matches; // sorted by score desc

    public String getWord() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }

    public int getTokenStart() {
        return tokenStart;
    }

    public void setTokenStart(int tokenStart) {
        this.tokenStart = tokenStart;
    }

    public int getTokenEnd() {
        return tokenEnd;
    }

    public void setTokenEnd(int tokenEnd) {
        this.tokenEnd = tokenEnd;
    }

    public List<TargetScore> getMatches() {
        return matches;
    }

    public void setMatches(List<TargetScore> matches) {
        this.matches = matches;
    }
}
