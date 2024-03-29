#include "ZQLandMarker.h"
#include <string>

#include <ncnn/net.h>

namespace mirror {
    ZQLandMarker::ZQLandMarker(FaceLandMarkerType type) : LandMarker(type) {
        inputSize_.width = 112;
        inputSize_.height = 112;
    }

    int ZQLandMarker::loadModel(const char *root_path) {
        std::string fl_param = std::string(root_path) + modelPath_ + "/zq/fl.param";
        std::string fl_bin = std::string(root_path) + modelPath_ + "/zq/fl.bin";
        return Super::loadModel(fl_param.c_str(), fl_bin.c_str());
    }

#if defined __ANDROID__
    int ZQLandMarker::loadModel(AAssetManager *mgr) {
        std::string fl_param = "models" + modelPath_ + "/zq/fl.param";
        std::string fl_bin = "models" + modelPath_ + "/zq/fl.bin";
        return Super::loadModel(mgr, fl_param.c_str(), fl_bin.c_str());
    }
#endif

    int ZQLandMarker::extractKeypoints(const cv::Mat &img_src,
                                       const cv::Rect &face,
                                       std::vector<cv::Point2f> &keypoints) const {
        cv::Mat img_face = img_src(face).clone();
        ncnn::Extractor ex = net_->create_extractor();
#if NCNN_VULKAN
        if (this->gpu_mode_) {
            ex.set_vulkan_compute(this->gpu_mode_);
        }
#endif
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_face.data,
                                                     ncnn::Mat::PIXEL_BGR, img_face.cols,
                                                     img_face.rows, inputSize_.width, inputSize_.height);
        in.substract_mean_normalize(meanVals, normVals);
        ex.input("data", in);
        ncnn::Mat out;
        ex.extract("bn6_3", out);

        for (int i = 0; i < 106; ++i) {
            float x = abs(out[2 * i] * img_face.cols) + face.x;
            float y = abs(out[2 * i + 1] * img_face.rows) + face.y;
            keypoints.emplace_back(x, y);
        }

        return 0;
    }

}
