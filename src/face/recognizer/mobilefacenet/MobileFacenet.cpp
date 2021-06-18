#include "MobileFacenet.h"
#include <string>

#include <ncnn/net.h>

namespace mirror {
    MobileFacenet::MobileFacenet(FaceRecognizerType type) : Recognizer(type) {
        faceFaceFeatureDim_ = 128;
        inputSize_.width = 112;
        inputSize_.height = 112;
    }

    int MobileFacenet::loadModel(const char *root_path) {
        std::string sub_dir = "/recognizers/mobilefacenet";
        std::string fr_param = std::string(root_path) + sub_dir + "/fr.param";
        std::string fr_bin = std::string(root_path) + sub_dir + "/fr.bin";
        if (this->net_->load_param(fr_param.c_str()) == -1 ||
            this->net_->load_model(fr_bin.c_str()) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }
        return 0;
    }

    int MobileFacenet::extractFeature(const cv::Mat &img_face, std::vector<float> &feature) const {
        cv::Mat face_cpy = img_face.clone();
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(face_cpy.data,
                                                     ncnn::Mat::PIXEL_BGR2RGB, face_cpy.cols,
                                                     face_cpy.rows, inputSize_.width, inputSize_.height);
        feature.resize(faceFaceFeatureDim_);
        ncnn::Extractor ex = this->net_->create_extractor();
        ex.input("data", in);
        ncnn::Mat out;
        ex.extract("fc1", out);
        for (int i = 0; i < faceFaceFeatureDim_; ++i) {
            feature.at(i) = out[i];
        }

        return 0;
    }

}


