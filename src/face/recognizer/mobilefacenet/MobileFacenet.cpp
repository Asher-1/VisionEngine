#include "MobileFacenet.h"
#include <string>
#include <iostream>
#include <ncnn/net.h>

namespace mirror {
    MobileFacenet::MobileFacenet(FaceRecognizerType type) : Recognizer(type) {
        faceFaceFeatureDim_ = 128;
        inputSize_.width = 112;
        inputSize_.height = 112;
    }

    int MobileFacenet::loadModel(const char *root_path) {
        std::string fr_param = std::string(root_path) + modelPath_ + "/mobilefacenet/fr.param";
        std::string fr_bin = std::string(root_path) + modelPath_ + "/mobilefacenet/fr.bin";
        return Super::loadModel(fr_param.c_str(), fr_bin.c_str());
    }

#if defined __ANDROID__
    int MobileFacenet::loadModel(AAssetManager *mgr) {
        std::string sub_dir = "models";
        std::string fr_param = sub_dir + modelPath_ + "/mobilefacenet/fr.param";
        std::string fr_bin = sub_dir + modelPath_ + "/mobilefacenet/fr.bin";
        return Super::loadModel(mgr, fr_param.c_str(), fr_bin.c_str());
    }
#endif

    int MobileFacenet::extractFeature(const cv::Mat &img_face, std::vector<float> &feature) const {
        cv::Mat face_cpy = img_face.clone();
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(face_cpy.data,
                                                     ncnn::Mat::PIXEL_BGR2RGB, face_cpy.cols,
                                                     face_cpy.rows, inputSize_.width, inputSize_.height);
        feature.resize(faceFaceFeatureDim_);
        ncnn::Extractor ex = this->net_->create_extractor();
#if NCNN_VULKAN
        if (this->gpu_mode_) {
            ex.set_vulkan_compute(this->gpu_mode_);
        }
#endif
        ex.input("data", in);
        ncnn::Mat out;
        ex.extract("fc1", out);

#if defined(_OPENMP)
#pragma omp parallel for num_threads(CUSTOM_THREAD_NUMBER)
#endif
        for (int i = 0; i < faceFaceFeatureDim_; ++i) {
            feature.at(i) = out[i];
        }

        return 0;
    }

}


