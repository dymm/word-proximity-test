package com.example.springbootapp.dto;

public class TargetScore {
    private String target;
    private float score;

    public TargetScore() {}

    public TargetScore(String target, float score) {
        this.target = target;
        this.score = score;
    }

    public String getTarget() {
        return target;
    }

    public void setTarget(String target) {
        this.target = target;
    }

    public float getScore() {
        return score;
    }

    public void setScore(float score) {
        this.score = score;
    }
}
