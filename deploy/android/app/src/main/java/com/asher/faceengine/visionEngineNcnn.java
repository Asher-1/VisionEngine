// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.asher.faceengine;

import java.util.List;

import android.view.Surface;
import android.graphics.Bitmap;
import android.content.res.AssetManager;


public class visionEngineNcnn {

    public static class FaceInfo {
        public float x;
        public float y;
        public float w;
        public float h;
        public float faceProb;
        public String name;
        public float liveProb;
        public float similarity;
        public boolean isLive;
        public boolean isSame;
    }

    public interface LivingCallBack {
        void callback(FaceInfo face);
    }

    public interface VerifyCallBack {
        void callback(FaceInfo face);
    }

    public native boolean addCallback(final VerifyCallBack verifyCallBack,
                                      final LivingCallBack livingCallBack);

    public native boolean verifyFacesWithCallback(Bitmap bitmap, boolean inPlace);

    public native boolean verifyFace(Bitmap bitmap, KeyPoint keyPoint);

    public native boolean clearFaces();

    public native boolean findFaces(List<String> names);

    public native boolean deleteFace(String name);

    public native boolean registerFace(Bitmap bitmap, String name);

    public native KeyPoint[] detectFaces(Bitmap bitmap, float threshold, float nmsThreshold);

    public native FaceKeyPoint[] extractKeypoints(Bitmap bitmap, float threshold, float nmsThreshold);

    public native LivingInfo detectLiving(Bitmap bitmap, KeyPoint keyPoint, float threshold);
    public LivingInfo detectLiving(Bitmap bitmap, KeyPoint keyPoint) {
        return detectLiving(bitmap, keyPoint, 0.915f);
    }

    public native boolean loadModel(AssetManager assetManager, String db_path, int det_model_id,
                                    int rec_model_id, int rec_enabled,
                                    int live_enabled, int show_kps, int cpugpu);

    public boolean loadModel(AssetManager assetManager, String dbPath, int detModelId,
                             boolean liveEnabled, boolean useGpu) {
        return loadModel(assetManager, dbPath, detModelId, 0, 1,
                liveEnabled ? 1 : 0, 0, useGpu ? 1 : 0);
    }

    public boolean loadModel(AssetManager assetManager, String dbPath,
                             boolean liveEnabled, boolean useGpu) {
        return loadModel(assetManager, dbPath, 0, 0,
                1, liveEnabled ? 1 : 0, 0, useGpu ? 1 : 0);
    }

    public native boolean loadLandmarkModel(AssetManager assetManager, int landmark_model_id, int cpugpu);

    public boolean loadLandmarkModel(AssetManager assetManager, int landmarkModelId, boolean useGpu) {
        return loadLandmarkModel(assetManager, landmarkModelId, useGpu ? 1 : 0);
    }

    public native boolean openCamera(int facing);

    public native boolean closeCamera();

    public native boolean setOutputWindow(Surface surface);

    static {
        System.loadLibrary("visionEngineNcnn");
    }
}
