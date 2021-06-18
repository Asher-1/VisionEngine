#include "ZQLandMarker.h"
#include <string>

#include <ncnn/net.h>

namespace mirror {
    ZQLandMarker::ZQLandMarker(FaceLandMarkerType type) : LandMarker(type) {
        inputSize_.width = 112;
        inputSize_.height = 112;
    }

    int ZQLandMarker::loadModel(const char *root_path) {
        std::string sub_dir = "/landmarkers/zq";
        std::string fl_param = std::string(root_path) + sub_dir + "/fl.param";
        std::string fl_bin = std::string(root_path) + sub_dir + "/fl.bin";
        if (net_->load_param(fl_param.c_str()) == -1 ||
            net_->load_model(fl_bin.c_str()) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }
        return 0;
    }

    int ZQLandMarker::extractKeypoints(const cv::Mat &img_src,
                                       const cv::Rect &face,
                                       std::vector<cv::Point2f> &keypoints) const {
        cv::Mat img_face = img_src(face).clone();
        ncnn::Extractor ex = net_->create_extractor();
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
