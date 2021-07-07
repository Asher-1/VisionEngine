package com.asher.faceengine;

public class LivingInfo {
    public float score;
    public boolean isLiving;

    public LivingInfo(float score, boolean isLiving) {
        this.score = score;
        this.isLiving = isLiving;
    }

    public boolean isLiving() {
        return isLiving;
    }

    public void setLiving(boolean living) {
        isLiving = living;
    }

    public float getScore() {
        return score;
    }

    public void setScore(float score) {
        this.score = score;
    }
}
