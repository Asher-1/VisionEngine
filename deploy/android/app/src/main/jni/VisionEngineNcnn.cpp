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

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "FaceEngine.h"
#include "VisionTools.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON


#define ASSERT(status, ret)     if (!(status)) { return ret; }
#define ASSERT_FALSE(status)    ASSERT(status, false)

using namespace mirror;

static FaceEngineParams faceParams;
static ncnn::Mutex lock;
static bool showKps = false;
static bool recEnabled = false;
static bool liveEnabled = false;
static const int keypointsNum = 5;
static const float IDScoreThreshold = 0.5;
static const float LivingThreshold = 0.915;
static const cv::Scalar boxColor(0, 255, 0);


static int draw_unsupported(cv::Mat &rgb) {
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                          1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat &img_src) {
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f) {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--) {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f) {
            return 0;
        }

        for (int i = 0; i < 10; i++) {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int thickness = 1;
    float fontScale = 0.5;
    int orientation = 1; // top right
    cv::Point2i position;
    utility::GetTextCornerPosition(img_src, text, orientation, fontScale, thickness, position);
    utility::DrawText(img_src, position, text, fontScale, thickness);
    return 0;
}

static bool BitmapToMatrix(JNIEnv *env, jobject obj_bitmap, cv::Mat &matrix) {
    if (!obj_bitmap || !env) {
        return false;
    }

    void *bitmapPixels; // Save picture pixel data
    AndroidBitmapInfo bitmapInfo; // Save picture parameters

    // Get picture parameters
    ASSERT_FALSE(AndroidBitmap_getInfo(env, obj_bitmap, &bitmapInfo) >= 0);
    // Only ARGB? 8888 and RGB? 565 are supported
    ASSERT_FALSE(bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888
                 || bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGB_565);
    // Get picture pixels (lock memory block)
    ASSERT_FALSE(AndroidBitmap_lockPixels(env, obj_bitmap, &bitmapPixels) >= 0);
    ASSERT_FALSE(bitmapPixels);

    // Establish temporary mat
    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);
        tmp.copyTo(matrix); // Copy to target matrix
    } else {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC2, bitmapPixels);
        cv::cvtColor(tmp, matrix, cv::COLOR_BGR5652RGB);
    }

    //convert RGB to BGR
    cv::cvtColor(matrix, matrix, cv::COLOR_RGB2BGR);

    AndroidBitmap_unlockPixels(env, obj_bitmap); // Unlock
    return true;
}

static bool MatrixToBitmap(JNIEnv *env, const cv::Mat &matrix, jobject obj_bitmap) {
    if (!obj_bitmap || !env) {
        return false;
    }

    void *bitmapPixels; // Save picture pixel data
    AndroidBitmapInfo bitmapInfo; // Save picture parameters

    ASSERT_FALSE(AndroidBitmap_getInfo(env, obj_bitmap, &bitmapInfo) >=
                 0);        // Get picture parameters
    ASSERT_FALSE(bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888
                 || bitmapInfo.format ==
                    ANDROID_BITMAP_FORMAT_RGB_565); // Only ARGB? 8888 and RGB? 565 are supported
    // It must be a 2-dimensional matrix with the same length and width
    ASSERT_FALSE(matrix.dims == 2
                 && bitmapInfo.height == (uint32_t) matrix.rows
                 && bitmapInfo.width == (uint32_t) matrix.cols);
    ASSERT_FALSE(matrix.type() == CV_8UC1 || matrix.type() == CV_8UC3 || matrix.type() == CV_8UC4);
    // Get picture pixels (lock memory block)
    ASSERT_FALSE(AndroidBitmap_lockPixels(env, obj_bitmap, &bitmapPixels) >= 0);
    ASSERT_FALSE(bitmapPixels);

    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);
        switch (matrix.type()) {
            case CV_8UC1:
                cv::cvtColor(matrix, tmp, cv::COLOR_GRAY2RGBA);
                break;
            case CV_8UC3:
                cv::cvtColor(matrix, tmp, cv::COLOR_RGB2RGBA);
                break;
            case CV_8UC4:
                matrix.copyTo(tmp);
                break;
            default:
                AndroidBitmap_unlockPixels(env, obj_bitmap);
                return false;
        }
    } else {
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC2, bitmapPixels);
        switch (matrix.type()) {
            case CV_8UC1:
                cv::cvtColor(matrix, tmp, cv::COLOR_GRAY2BGR565);
                break;
            case CV_8UC3:
                cv::cvtColor(matrix, tmp, cv::COLOR_RGB2BGR565);
                break;
            case CV_8UC4:
                cv::cvtColor(matrix, tmp, cv::COLOR_RGBA2BGR565);
                break;
            default:
                AndroidBitmap_unlockPixels(env, obj_bitmap);
                return false;
        }
    }
    AndroidBitmap_unlockPixels(env, obj_bitmap); // Unlock
    return true;
}

static void ProcessImage(cv::Mat &img_src) {
    ncnn::MutexLockGuard g(lock);
    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    if (face_engine) {
        std::vector<FaceInfo> faces;
        // face detection
        face_engine->detectFace(img_src, faces);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                            "detect %s faces", std::to_string(faces.size()).c_str());
        // draw faces
        for (const FaceInfo &face: faces) {
            cv::rectangle(img_src, face.location_, boxColor);

            if (showKps) {
                std::vector<cv::Point2f> keyPoints;
                ConvertKeyPoints(face.keypoints_, keypointsNum, keyPoints);
                utility::DrawKeyPoints(img_src, keyPoints, 2, boxColor);
            }

            char text[64];
            sprintf(text, "%s %.1f%%", "face score", face.score_ * 100);
            int next_y = utility::DrawText(img_src, face.location_.tl(), text);

            // living detection
            bool is_living = true;
            if (liveEnabled) {
                float livingScore;
                cv::Scalar color;
                is_living = face_engine->detectLivingFace(
                        img_src, face.location_, livingScore);
                if (is_living) {
                    color = boxColor;
                    sprintf(text, "%s %.1f%%", "real face", livingScore * 100);
                } else {
                    color = cv::Scalar(0, 0, 255);
                    sprintf(text, "%s %.1f%%", "fake face", livingScore * 100);
                }
                cv::rectangle(img_src, face.location_, color, 2);
                cv::Point2i position(face.location_.tl().x, next_y);
                next_y = utility::DrawText(img_src, position, text);
            }

            // face recognition
            if (is_living && recEnabled) {

                // align face
                cv::Mat faceAligned;
                std::vector<cv::Point2f> keyPoints;
                ConvertKeyPoints(face.keypoints_, keypointsNum, keyPoints);
                face_engine->alignFace(img_src, keyPoints, faceAligned);

                // extract feature
                std::vector<float> feat;
                face_engine->extractFeature(faceAligned, feat);

                // compare feature
                QueryResult query_result;
                int flag = face_engine->QueryTop(feat, query_result);
                if (flag == ErrorCode::EMPTY_DATA_ERROR) {
                    __android_log_print(ANDROID_LOG_WARN, "ncnn",
                                        "face database is empty, please register first!");
                }

                if (query_result.sim_ > IDScoreThreshold) {
                    sprintf(text, "%s %.1f%%", query_result.name_.c_str(),
                            query_result.sim_ * 100);
                } else {
                    sprintf(text, "%s %.1f%%", "stranger",
                            query_result.sim_ * 100);
                }
                utility::DrawText(img_src,
                                  cv::Point2i(face.location_.tl().x, next_y), text);
            }
        }
    } else {
        draw_unsupported(img_src);
    }
}

class MyNdkCamera : public NdkCameraWindow {
public:
    void on_image_render(cv::Mat &img_src) const override;
};

void MyNdkCamera::on_image_render(cv::Mat &img_src) const {
    ProcessImage(img_src);
    draw_fps(img_src);
}

static MyNdkCamera *g_camera = nullptr;

const char *NATIVE_VISION_ENGINE_CLASS_NAME = "(Lcom/asher/faceengine/visionEngineNcnn;)V";
const char *NATIVE_FACE_OBJECT_CLASS_NAME = "com/asher/faceengine/visionEngineNcnn$FaceInfo";
const char *NATIVE_FACEINFO_CLASS_NAME = "(Lcom/asher/faceengine/visionEngineNcnn$FaceInfo;)V";
const char *NATIVE_CALLBACK_METHOD_NAME = "callback";
const char *NATIVE_LIVING_OBJECT_CLASS_NAME = "com/asher/faceengine/visionEngineNcnn$LivingCallBack";
const char *NATIVE_LIVING_INTERFACE_CLASS_NAME = "(Lcom/asher/faceengine/visionEngineNcnn$LivingCallBack;)V";
const char *NATIVE_VERIFY_OBJECT_CLASS_NAME = "com/asher/faceengine/visionEngineNcnn$VerifyCallBack";
const char *NATIVE_VERIFY_INTERFACE_CLASS_NAME = "(Lcom/asher/faceengine/visionEngineNcnn$VerifyCallBack;)V";

struct CallBack {
    jclass objCls = nullptr;
    jobject interfaceCallback;
    jmethodID callback;

    bool isValid() const { return objCls && interfaceCallback && callback; }
};

struct FaceObject {
    jclass objCls = nullptr;
    jmethodID constructortorId;
    jfieldID xId;
    jfieldID yId;
    jfieldID wId;
    jfieldID hId;
    jfieldID nameId;
    jfieldID faceProbId;
    jfieldID liveProbId;
    jfieldID similarityId;
    jfieldID isLiveId;
    jfieldID isSameId;
};

static FaceObject faceObject;
static CallBack liveCallback;
static CallBack verifyCallback;

static void RegisterFaceObject(JNIEnv *env) {
    // init jni glue
    jclass localObjCls = env->FindClass(NATIVE_FACE_OBJECT_CLASS_NAME);
    faceObject.objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));
    faceObject.constructortorId = env->GetMethodID(faceObject.objCls, "<init>",
                                                   "()V");

    faceObject.xId = env->GetFieldID(faceObject.objCls, "x", "F");
    faceObject.yId = env->GetFieldID(faceObject.objCls, "y", "F");
    faceObject.wId = env->GetFieldID(faceObject.objCls, "w", "F");
    faceObject.hId = env->GetFieldID(faceObject.objCls, "h", "F");
    faceObject.faceProbId = env->GetFieldID(faceObject.objCls, "faceProb", "F");
    faceObject.nameId = env->GetFieldID(faceObject.objCls, "name", "Ljava/lang/String;");
    faceObject.liveProbId = env->GetFieldID(faceObject.objCls, "liveProb", "F");
    faceObject.similarityId = env->GetFieldID(faceObject.objCls, "similarity", "F");
    faceObject.isLiveId = env->GetFieldID(faceObject.objCls, "isLive", "Z");
    faceObject.isSameId = env->GetFieldID(faceObject.objCls, "isSame", "Z");

}

static void RegisterLiveCallback(JNIEnv *env, jobject callback_object) {
    if (!callback_object)
        return;


    liveCallback.objCls = env->GetObjectClass(callback_object);
    liveCallback.callback = env->GetMethodID(liveCallback.objCls,
                                             NATIVE_CALLBACK_METHOD_NAME,
                                             NATIVE_FACEINFO_CLASS_NAME);
    liveCallback.interfaceCallback = env->NewGlobalRef(callback_object);

//    jclass localObjCls = env->FindClass(NATIVE_LIVING_OBJECT_CLASS_NAME);
//    liveCallback.objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));
}

static void RegisterVerifyCallback(JNIEnv *env, jobject callback_object) {
    if (!callback_object)
        return;

    verifyCallback.objCls = env->GetObjectClass(callback_object);
    verifyCallback.callback = env->GetMethodID(verifyCallback.objCls,
                                               NATIVE_CALLBACK_METHOD_NAME,
                                               NATIVE_FACEINFO_CLASS_NAME);
    verifyCallback.interfaceCallback = env->NewGlobalRef(callback_object);
}

static void ConvertFaceInfo(JNIEnv *env, jobject jObj, const FaceInfo &face) {
    env->SetFloatField(jObj, faceObject.xId, static_cast<float>(face.location_.x));
    env->SetFloatField(jObj, faceObject.yId, static_cast<float>(face.location_.y));
    env->SetFloatField(jObj, faceObject.wId, static_cast<float>(face.location_.width));
    env->SetFloatField(jObj, faceObject.hId, static_cast<float>(face.location_.height));
    env->SetFloatField(jObj, faceObject.faceProbId, face.score_);
}


extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");
    g_camera = new MyNdkCamera();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");
    {
        ncnn::MutexLockGuard g(lock);

        if (FaceEngine::GetInstancePtr()) {
            FaceEngine::GetInstancePtr()->Save();
            FaceEngine::ReleaseInstance();
        }
    }

    delete g_camera;
    g_camera = nullptr;
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_asher_faceengine_visionEngineNcnn_loadModel(
        JNIEnv *env, jobject thiz, jobject assetManager, jstring db_path, jint det_model_id,
        jint rec_model_id, jint rec_enabled, jint live_enabled, jint show_kps, jint cpugpu) {

    if (det_model_id < 0 || det_model_id > 4) {
        return JNI_FALSE;
    }

    if (cpugpu < 0 || cpugpu > 1) {
        return JNI_FALSE;
    }

    if (rec_model_id < 0 || rec_model_id > 0) {
        return JNI_FALSE;
    }

    std::string faceFeaturePath = env->GetStringUTFChars(db_path, nullptr);
    if (faceFeaturePath.empty()) {
        return JNI_FALSE;
    }

    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    bool use_gpu = (int) cpugpu == 1;
    liveEnabled = (int) live_enabled == 1;
    recEnabled = (int) rec_enabled == 1;
    showKps = (int) show_kps == 1;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        if (use_gpu && ncnn::get_gpu_count() == 0) {
            // no gpu
            __android_log_print(ANDROID_LOG_WARN, "ncnn", "No gpu device found!");
            if (FaceEngine::GetInstancePtr()) {
                FaceEngine::ReleaseInstance();
            }
            return JNI_FALSE;

        } else {
            if (FaceEngine::GetInstancePtr()) {
                faceParams.mgr = mgr;
                faceParams.threadNum = 4;
                faceParams.gpuEnabled = use_gpu;
                faceParams.faceFeaturePath = faceFeaturePath;
                faceParams.faceDetectorType = (FaceDetectorType) det_model_id;
                faceParams.faceRecognizerEnabled = recEnabled;
                faceParams.faceRecognizerType = (FaceRecognizerType) rec_model_id;
                faceParams.livingThreshold = LivingThreshold;
                faceParams.faceAntiSpoofingEnabled = liveEnabled;
                // params.faceDetectorType = FaceDetectorType::SCRFD_FACE;
                int flag = FaceEngine::GetInstancePtr()->loadModel(faceParams);
                if (flag != 0) {
                    __android_log_print(ANDROID_LOG_ERROR, "ncnn", "loadModel failed!");
                    return JNI_FALSE;
                } else {
                    FaceEngine::GetInstancePtr()->Load();
                    return JNI_TRUE;
                }
            }
            return JNI_FALSE;
        }
    }
}

JNIEXPORT jboolean JNICALL
Java_com_asher_faceengine_visionEngineNcnn_loadLandmarkModel(JNIEnv *env, jobject thiz,
                                                             jobject asset_manager,
                                                             jint landmark_model_id, jint cpugpu) {
    if (landmark_model_id < 0 || landmark_model_id > 1) {
        return JNI_FALSE;
    }

    AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);
    bool use_gpu = (int) cpugpu == 1;
    if (FaceEngine::GetInstancePtr()) {
        ncnn::MutexLockGuard g(lock);
        faceParams.mgr = mgr;
        faceParams.threadNum = 4;
        faceParams.gpuEnabled = use_gpu;
        faceParams.faceDetectorEnabled = true;
        faceParams.faceLandMarkerEnabled = true;
        faceParams.faceLandMarkerType = (FaceLandMarkerType) landmark_model_id;
        int flag = FaceEngine::GetInstancePtr()->loadModel(faceParams);
        if (flag != 0) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "loadModel failed!");
            return JNI_FALSE;
        } else {
            return JNI_TRUE;
        }
    }
    return JNI_FALSE;
}


JNIEXPORT jboolean JNICALL
Java_com_asher_faceengine_visionEngineNcnn_addCallback(JNIEnv *env, jobject thiz,
                                                       jobject verify_call_back,
                                                       jobject living_call_back) {
    RegisterFaceObject(env);

    if (!verify_call_back && !living_call_back) {
        return JNI_FALSE;
    }

    if (verify_call_back) {
        RegisterVerifyCallback(env, verify_call_back);
    }

    if (living_call_back) {
        RegisterLiveCallback(env, living_call_back);
    }

    return JNI_TRUE;
}

JNIEXPORT jobjectArray JNICALL
Java_com_asher_faceengine_visionEngineNcnn_extractKeypoints(JNIEnv *env, jobject thiz,
                                                            jobject bitmap, jfloat threshold,
                                                            jfloat nms_threshold) {
    auto box_cls = env->FindClass("com/asher/faceengine/FaceKeyPoint");
    auto cid = env->GetMethodID(box_cls, "<init>", "(FF)V");

    cv::Mat img_src;
    bool ret = BitmapToMatrix(env, bitmap, img_src); // Bitmap to cv::Mat
    if (!ret) {
        return env->NewObjectArray(0, box_cls, nullptr);
    }

    ncnn::MutexLockGuard g(lock);
    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    faceParams.faceLandMarkerEnabled = true;
    faceParams.faceLandMarkerEnabled = true;
    faceParams.scoreThreshold = static_cast<float>(threshold);
    faceParams.nmsThreshold = static_cast<float>(nms_threshold);
    if (!face_engine || face_engine->loadModel(faceParams) != 0) {
        return env->NewObjectArray(0, box_cls, nullptr);
    }

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);
    std::vector<cv::Point2f> keypoints;
    for (const FaceInfo &face : faces) {
        std::vector<cv::Point2f> keypoints_tmp;
        face_engine->extractKeypoints(img_src, face.location_, keypoints_tmp);
        keypoints.insert(keypoints.end(), keypoints_tmp.begin(), keypoints_tmp.end());
    }

    jobjectArray arrayInfo = env->NewObjectArray(keypoints.size(), box_cls, nullptr);
    int i = 0;
    for (auto &keypoint : keypoints) {
        env->PushLocalFrame(1);
        jobject obj = env->NewObject(box_cls, cid, keypoint.x, keypoint.y);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(arrayInfo, i++, obj);
    }
    return arrayInfo;
}

JNIEXPORT jobjectArray JNICALL
Java_com_asher_faceengine_visionEngineNcnn_detectFaces(JNIEnv *env, jobject thiz, jobject bitmap,
                                                       jfloat threshold, jfloat nms_threshold) {
    auto box_cls = env->FindClass("com/asher/faceengine/KeyPoint");
    auto cid = env->GetMethodID(box_cls, "<init>", "([F[FFFFFF)V");

    cv::Mat img_src;
    bool ret = BitmapToMatrix(env, bitmap, img_src); // Bitmap to cv::Mat
    if (!ret) {
        return env->NewObjectArray(0, box_cls, nullptr);
    }

    ncnn::MutexLockGuard g(lock);
    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    faceParams.faceDetectorEnabled = true;
    faceParams.scoreThreshold = static_cast<float>(threshold);
    faceParams.nmsThreshold = static_cast<float>(nms_threshold);
    if (!face_engine || face_engine->loadModel(faceParams) != 0) {
        return env->NewObjectArray(0, box_cls, nullptr);
    }

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);
    jobjectArray arrayInfo = env->NewObjectArray(faces.size(), box_cls, nullptr);
    int i = 0;
    int KEY_NUM = 5;
    for (const FaceInfo &face : faces) {
        env->PushLocalFrame(1);
        float x[KEY_NUM];
        float y[KEY_NUM];
        for (int j = 0; j < KEY_NUM; j++) {
            x[j] = face.keypoints_[j].x;
            y[j] = face.keypoints_[j].y;
        }
        jfloatArray xs = env->NewFloatArray(KEY_NUM);
        env->SetFloatArrayRegion(xs, 0, KEY_NUM, x);
        jfloatArray ys = env->NewFloatArray(KEY_NUM);
        env->SetFloatArrayRegion(ys, 0, KEY_NUM, y);

        jobject obj = env->NewObject(box_cls, cid, xs, ys,
                                     static_cast<jfloat>(face.location_.x),
                                     static_cast<jfloat>(face.location_.y),
                                     static_cast<jfloat>(face.location_.br().x),
                                     static_cast<jfloat>(face.location_.br().y),
                                     static_cast<jfloat>(face.score_));
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(arrayInfo, i++, obj);
    }

    return arrayInfo;

}

JNIEXPORT jobject JNICALL
Java_com_asher_faceengine_visionEngineNcnn_detectLiving(JNIEnv *env, jobject thiz, jobject bitmap,
                                                        jobject key_point, jfloat threshold) {
    auto living_cls = env->FindClass("com/asher/faceengine/LivingInfo");
    auto cid = env->GetMethodID(living_cls, "<init>", "(FZ)V");

    jclass keyPoint = env->GetObjectClass(key_point);
    jfieldID x0Id = env->GetFieldID(keyPoint, "x0", "F");
    jfieldID y0Id = env->GetFieldID(keyPoint, "y0", "F");
    jfieldID x1Id = env->GetFieldID(keyPoint, "x1", "F");
    jfieldID y1Id = env->GetFieldID(keyPoint, "y1", "F");
    env->DeleteLocalRef(keyPoint);

    cv::Mat img_src;
    bool ret = BitmapToMatrix(env, bitmap, img_src); // Bitmap to cv::Mat
    if (!ret) {
        return env->NewObject(living_cls, cid, 0.0f, JNI_FALSE);
    }

    ncnn::MutexLockGuard g(lock);
    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    faceParams.faceAntiSpoofingEnabled = true;
    faceParams.livingThreshold = static_cast<float>(threshold);
    if (!face_engine || face_engine->loadModel(faceParams) != 0) {
        return env->NewObject(living_cls, cid, 0.0f, JNI_FALSE);
    }

    cv::Rect rect;
    rect.x = static_cast<int>(env->GetFloatField(key_point, x0Id));
    rect.y = static_cast<int>(env->GetFloatField(key_point, y0Id));
    rect.width = static_cast<int>(env->GetFloatField(key_point, x1Id)) - rect.x;
    rect.height = static_cast<int>(env->GetFloatField(key_point, y1Id)) - rect.y;
    float livingScore;
    bool is_living = face_engine->detectLivingFace(img_src, rect, livingScore);
    return env->NewObject(living_cls, cid, livingScore, is_living);
}


JNIEXPORT jboolean JNICALL
Java_com_asher_faceengine_visionEngineNcnn_verifyFacesWithCallback(JNIEnv *env, jobject thiz,
                                                                   jobject bitmap,
                                                                   jboolean inPlace) {
    cv::Mat img_src;
    bool ret = BitmapToMatrix(env, bitmap, img_src); // Bitmap to cv::Mat
    if (!ret) {
        return JNI_FALSE;
    }

    bool drawing = inPlace == JNI_TRUE;

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    if (face_engine) {
        ncnn::MutexLockGuard g(lock);
        // face detection
        std::vector<FaceInfo> faces;
        face_engine->detectFace(img_src, faces);
        __android_log_print(ANDROID_LOG_VERBOSE, "ncnn",
                            "detect %s faces", std::to_string(faces.size()).c_str());
        if (faces.empty()) { return JNI_FALSE; }

        FaceInfo &face = faces[0];
        {
            char text[64];
            int next_y = 0;
            if (drawing) {
                cv::rectangle(img_src, face.location_, boxColor);
                if (showKps) {
                    std::vector<cv::Point2f> keyPoints;
                    ConvertKeyPoints(face.keypoints_, keypointsNum, keyPoints);
                    utility::DrawKeyPoints(img_src, keyPoints, 2, boxColor);
                }
                sprintf(text, "%s %.1f%%", "face score", face.score_ * 100);
                next_y = utility::DrawText(img_src, face.location_.tl(), text);
            }

            // living detection
            bool is_living = true;
            float livingScore = 1.0f;
            if (liveEnabled) {
                cv::Scalar color;
                is_living = face_engine->detectLivingFace(
                        img_src, face.location_, livingScore);
                if (drawing) {
                    if (is_living) {
                        color = boxColor;
                        sprintf(text, "%s %.1f%%", "real face", livingScore * 100);
                    } else {
                        color = cv::Scalar(0, 0, 255);
                        sprintf(text, "%s %.1f%%", "fake face", livingScore * 100);
                    }
                    cv::rectangle(img_src, face.location_, color, 2);
                    cv::Point2i position(face.location_.tl().x, next_y);
                    next_y = utility::DrawText(img_src, position, text);
                }
            }

            // face recognition
            QueryResult query_result;
            query_result.sim_ = 0.0f;
            query_result.name_ = "stranger";
            bool isSame = false;
            if (is_living && recEnabled) {

                // align face
                cv::Mat faceAligned;
                std::vector<cv::Point2f> keyPoints;
                ConvertKeyPoints(face.keypoints_, keypointsNum, keyPoints);
                face_engine->alignFace(img_src, keyPoints, faceAligned);

                // extract feature
                std::vector<float> feat;
                face_engine->extractFeature(faceAligned, feat);

                // compare feature
                int flag = face_engine->QueryTop(feat, query_result);
                if (flag == ErrorCode::EMPTY_DATA_ERROR) {
                    __android_log_print(ANDROID_LOG_WARN, "ncnn",
                                        "face database is empty, please register first!");
                }

                if (query_result.sim_ > IDScoreThreshold) {
                    isSame = true;
                    sprintf(text, "%s %.1f%%", query_result.name_.c_str(),
                            query_result.sim_ * 100);
                } else {
                    isSame = false;
                    sprintf(text, "%s %.1f%%", "stranger", query_result.sim_ * 100);
                }

                if (drawing) {
                    utility::DrawText(img_src,
                                      cv::Point2i(face.location_.tl().x, next_y), text);
                }
            }

            if (drawing) {
                MatrixToBitmap(env, img_src, bitmap);

            }

            // convert
            jobject face_info = env->NewObject(faceObject.objCls, faceObject.constructortorId,
                                               thiz);
            {
                // construct FaceObject
                ConvertFaceInfo(env, face_info, face);
                env->SetObjectField(face_info, faceObject.nameId,
                                    env->NewStringUTF(query_result.name_.c_str()));
                env->SetFloatField(face_info, faceObject.similarityId, query_result.sim_);
                env->SetFloatField(face_info, faceObject.liveProbId, livingScore);
                env->SetBooleanField(face_info, faceObject.isLiveId, is_living);
                env->SetBooleanField(face_info, faceObject.isSameId, isSame);

                if (!is_living && liveCallback.isValid()) {
                    env->CallVoidMethod(liveCallback.interfaceCallback, liveCallback.callback,
                                        face_info);
                }

                if (verifyCallback.isValid()) {
                    env->CallVoidMethod(verifyCallback.interfaceCallback, verifyCallback.callback,
                                        face_info);
                }
            }

        }
    }

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_asher_faceengine_visionEngineNcnn_verifyFace(JNIEnv *env, jobject thiz, jobject bitmap,
                                                      jobject key_point) {
    cv::Mat img_src;
    bool ret = BitmapToMatrix(env, bitmap, img_src); // Bitmap to cv::Mat
    if (!ret) {
        return JNI_FALSE;
    }

    auto keyPoint_cls = reinterpret_cast<jclass>(env->NewGlobalRef(env->GetObjectClass(key_point)));
    jfieldID keyXIds = env->GetFieldID(keyPoint_cls, "x", "[F");
    jfieldID keyYIds = env->GetFieldID(keyPoint_cls, "y", "[F");

    ncnn::MutexLockGuard g(lock);
    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    faceParams.faceRecognizerEnabled = true;
    if (!face_engine || face_engine->loadModel(faceParams) != 0) {
        return JNI_FALSE;
    }

    // face recognition
    jfloatArray xArray = (jfloatArray) env->GetObjectField(key_point, keyXIds);
    jfloatArray yArray = (jfloatArray) env->GetObjectField(key_point, keyYIds);
    jboolean isCopy = JNI_FALSE;
    jfloat *x = env->GetFloatArrayElements(xArray, &isCopy);
    jfloat *y = env->GetFloatArrayElements(yArray, &isCopy);

    cv::Mat faceAligned;
    std::vector<cv::Point2f> keyPoints(keypointsNum);
    for (int j = 0; j < keypointsNum; j++) {
        keyPoints[j].x = static_cast<float>(x[j]);
        keyPoints[j].y = static_cast<float>(y[j]);
    }
    if (isCopy) {
        delete[] x;
        delete[] y;
        x = nullptr;
        y = nullptr;
    }
    face_engine->alignFace(img_src, keyPoints, faceAligned);

    // extract feature
    std::vector<float> feat;
    face_engine->extractFeature(faceAligned, feat);

    // compare feature
    QueryResult query_result;
    query_result.sim_ = 0.0f;
    query_result.name_ = "stranger";
    int flag = face_engine->QueryTop(feat, query_result);
    if (flag == ErrorCode::EMPTY_DATA_ERROR) {
        __android_log_print(ANDROID_LOG_WARN, "ncnn",
                            "face database is empty, please register first!");
    }

    bool isSame = false;
    if (query_result.sim_ > IDScoreThreshold) {
        isSame = true;
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                            "query id is %s!", query_result.name_.c_str());
    } else {
        isSame = false;
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                            "unknown person!");
    }

    return static_cast<jboolean>(isSame);
}

// public native boolean registerFace(Bitmap bitmap, String name);
JNIEXPORT jboolean JNICALL
Java_com_asher_faceengine_visionEngineNcnn_registerFace(JNIEnv *env, jobject thiz,
                                                        jobject bitmap, jstring name) {

    cv::Mat imageSrc;
    bool ret = BitmapToMatrix(env, bitmap, imageSrc); // Bitmap to cv::Mat
    if (!ret) {
        return JNI_FALSE;
    }

    std::string registeredName = env->GetStringUTFChars(name, nullptr);
    if (registeredName.empty()) {
        return JNI_FALSE;
    }

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    if (face_engine) {
        ncnn::MutexLockGuard g(lock);
        // face detection
        std::vector<FaceInfo> faces;
        face_engine->detectFace(imageSrc, faces);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                            "register detect %s faces", std::to_string(faces.size()).c_str());
        if (faces.empty()) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Cannot detect face!");
            return JNI_FALSE;
        }

        // only register first face

        // align face
        cv::Mat faceAligned;
        std::vector<cv::Point2f> keyPoints;
        ConvertKeyPoints(faces.at(0).keypoints_, keypointsNum, keyPoints);
        face_engine->alignFace(imageSrc, keyPoints, faceAligned);

        // extract feature
        std::vector<float> feat;
        int flag = face_engine->extractFeature(faceAligned, feat);
        if (flag != 0 ||
            feat.size() != kFaceFeatureDim) {
            if (flag == ErrorCode::UNINITIALIZED_ERROR) {
                __android_log_print(ANDROID_LOG_WARN, "ncnn",
                                    "Please enable recognizer first!");
            } else if (flag == ErrorCode::EMPTY_INPUT_ERROR) {
                __android_log_print(ANDROID_LOG_WARN, "ncnn",
                                    "Empty input aligned face image!");
            } else {
                __android_log_print(ANDROID_LOG_WARN, "ncnn",
                                    "Unknown error!");
            }
            return JNI_FALSE;
        }

        face_engine->Insert(feat, registeredName);
        if (face_engine->Save() != 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                                "save face database failed!");
            return JNI_FALSE;
        }

        return JNI_TRUE;
    }

    return JNI_FALSE;

}

JNIEXPORT jboolean JNICALL
Java_com_asher_faceengine_visionEngineNcnn_deleteFace(JNIEnv *env, jobject thiz, jstring name) {
    std::string registeredName = env->GetStringUTFChars(name, nullptr);
    if (registeredName.empty()) {
        return JNI_FALSE;
    }

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    if (face_engine) {
        if (face_engine->Delete(registeredName) != 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                                "delete registered face: %s failed", registeredName.c_str());
            return JNI_FALSE;
        } else {

            if (face_engine->Save() != 0) {
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                                    "save face database failed!");
                return JNI_FALSE;
            }
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                                "delete registered face: %s successfully", registeredName.c_str());
            return JNI_TRUE;
        }
    }

    return JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_asher_faceengine_visionEngineNcnn_clearFaces(JNIEnv *env, jobject thiz) {
    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    if (face_engine) {
        if (face_engine->Clear() != 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                                "clear all registered faces failed");
            return JNI_FALSE;
        } else {
            if (face_engine->Save() != 0) {
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                                    "save face database failed!");
                return JNI_FALSE;
            }
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                                "clear all registered faces successfully");
            return JNI_TRUE;
        }
    }

    return JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_asher_faceengine_visionEngineNcnn_findFaces(JNIEnv *env, jobject thiz, jobject names) {
    if (!names) {
        return JNI_FALSE;
    }

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    if (face_engine) {
        std::vector<std::string> namesC;
        if (face_engine->Find(namesC) != 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                                "find registered faces failed");
            return JNI_FALSE;
        } else {
            jclass objCls = env->GetObjectClass(names);
            jmethodID java_util_ArrayList_add = env->GetMethodID(objCls, "add",
                                                                 "(Ljava/lang/Object;)Z");
            if (!java_util_ArrayList_add) {
                return JNI_FALSE;
            }

            for (auto &name : namesC) {
                env->CallBooleanMethod(names, java_util_ArrayList_add,
                                       env->NewStringUTF(name.c_str()));
            }

            __android_log_print(ANDROID_LOG_DEBUG, "ncnn",
                                "find registered faces successfully");

            return JNI_TRUE;
        }
    }

    return JNI_FALSE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL
Java_com_asher_faceengine_visionEngineNcnn_openCamera(JNIEnv *env, jobject thiz, jint facing) {
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int) facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL
Java_com_asher_faceengine_visionEngineNcnn_closeCamera(JNIEnv *env, jobject thiz) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL
Java_com_asher_faceengine_visionEngineNcnn_setOutputWindow(JNIEnv *env, jobject thiz,
                                                           jobject surface) {
    ANativeWindow *win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

}