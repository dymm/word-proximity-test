package com.example.springbootapp.dto;

import java.util.List;

public class WordMatchRequest {
    private String sentence;
    private List<String> targets;
    private Float threshold; // optional, default 0.6
    private Integer topK;    // optional, default 3
    private Boolean aggregateSubwords; // optional, default true
    private Boolean excludeStopwords; // optional, default true
    private Integer minTokenLength;   // optional, default 3

    public String getSentence() {
        return sentence;
    }

    public void setSentence(String sentence) {
        this.sentence = sentence;
    }

    public List<String> getTargets() {
        return targets;
    }

    public void setTargets(List<String> targets) {
        this.targets = targets;
    }

    public Float getThreshold() {
        return threshold;
    }

    public void setThreshold(Float threshold) {
        this.threshold = threshold;
    }

    public Integer getTopK() {
        return topK;
    }

    public void setTopK(Integer topK) {
        this.topK = topK;
    }

    public Boolean getAggregateSubwords() {
        return aggregateSubwords;
    }

    public void setAggregateSubwords(Boolean aggregateSubwords) {
        this.aggregateSubwords = aggregateSubwords;
    }

    public Boolean getExcludeStopwords() {
        return excludeStopwords;
    }

    public void setExcludeStopwords(Boolean excludeStopwords) {
        this.excludeStopwords = excludeStopwords;
    }

    public Integer getMinTokenLength() {
        return minTokenLength;
    }

    public void setMinTokenLength(Integer minTokenLength) {
        this.minTokenLength = minTokenLength;
    }
}
