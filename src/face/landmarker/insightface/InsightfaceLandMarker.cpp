#include "InsightfaceLandMarker.h"
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <ncnn/net.h>

namespace mirror {
    InsightfaceLandMarker::InsightfaceLandMarker(FaceLandMarkerType type) : LandMarker(type) {
        inputSize_.width = 192;
        inputSize_.height = 192;
    }

    int InsightfaceLandMarker::loadModel(const char *root_path) {
        std::string fl_param = std::string(root_path) + modelPath_ + "/insightface/2d106.param";
        std::string fl_bin = std::string(root_path) + modelPath_ + "/insightface/2d106.bin";
        return Super::loadModel(fl_param.c_str(), fl_bin.c_str());
    }

#if defined __ANDROID__
    int InsightfaceLandMarker::loadModel(AAssetManager *mgr) {
        std::string fl_param = "models" + modelPath_ + "/insightface/2d106.param";
        std::string fl_bin = "models" + modelPath_ + "/insightface/2d106.bin";
        return Super::loadModel(mgr, fl_param.c_str(), fl_bin.c_str());
    }
#endif

    int InsightfaceLandMarker::extractKeypoints(const cv::Mat &img_src,
                                                const cv::Rect &face, std::vector<cv::Point2f> &keypoints) const {
        // 1 enlarge the face rect
        cv::Rect face_enlarged = face;
        EnlargeRect(enlarge_scale, face_enlarged);

        // 2 square the rect
        RectifyRect(face_enlarged);
        face_enlarged = face_enlarged & cv::Rect(0, 0, img_src.cols, img_src.rows);

        // 3 crop the face
        cv::Mat img_face = img_src(face_enlarged).clone();

        // 4 do inference
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_face.data,
                                                     ncnn::Mat::PIXEL_BGR2RGB, img_face.cols,
                                                     img_face.rows, inputSize_.width, inputSize_.height);

        ncnn::Extractor ex = net_->create_extractor();
#if NCNN_VULKAN
        ex.set_vulkan_compute(this->gpu_mode_);
#endif
        ex.input("data", in);
        ncnn::Mat out;
        ex.extract("fc1", out);

        for (int i = 0; i < 106; ++i) {
            float x = (out[2 * i] + 1.0f) * img_face.cols / 2 + face_enlarged.x;
            float y = (out[2 * i + 1] + 1.0f) * img_face.rows / 2 + face_enlarged.y;
            keypoints.emplace_back(x, y);
        }

        return 0;
    }

}
