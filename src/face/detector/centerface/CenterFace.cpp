#include "CenterFace.h"
#include <iostream>
#include "opencv2/imgproc.hpp"

#include <ncnn/net.h>

namespace mirror {
    CenterFace::CenterFace(FaceDetectorType type) : Detector(type) {
        iouThreshold_ = 0.5f;
        scoreThreshold_ = 0.5f;
    }

    int CenterFace::loadModel(const char *root_path) {
        std::string fd_param = std::string(root_path) + modelPath_ + "/centerface/centerface.param";
        std::string fd_bin = std::string(root_path) + modelPath_ + "/centerface/centerface.bin";
        return Super::loadModel(fd_param.c_str(), fd_bin.c_str());
    }


#if defined __ANDROID__
    int CenterFace::loadModel(AAssetManager *mgr) {
        std::string fd_param = "models" + modelPath_ + "/centerface/centerface.param";
        std::string fd_bin = "models" + modelPath_ + "/centerface/centerface.bin";
        return Super::loadModel(mgr, fd_param.c_str(), fd_bin.c_str());
    }
#endif


    int CenterFace::detectFace(const cv::Mat &img_src, std::vector<FaceInfo> &faces) const {
        int img_width = img_src.cols;
        int img_height = img_src.rows;

        int img_width_new = img_width / 32 * 32;
        int img_height_new = img_height / 32 * 32;
        float scale_x = static_cast<float>(img_width) / img_width_new;
        float scale_y = static_cast<float>(img_height) / img_height_new;

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                     img_width, img_height, img_width_new, img_height_new);
        ncnn::Extractor ex = net_->create_extractor();
#if NCNN_VULKAN
        if (this->gpu_mode_) {
            ex.set_vulkan_compute(this->gpu_mode_);
        }
#endif
        ex.input("input.1", in);
        ncnn::Mat mat_heatmap, mat_scale, mat_offset, mat_landmark;
        ex.extract("537", mat_heatmap);
        ex.extract("538", mat_scale);
        ex.extract("539", mat_offset);
        ex.extract("540", mat_landmark);

        int height = mat_heatmap.h;
        int width = mat_heatmap.w;
        faces.clear();
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int index = h * width + w;
                float score = mat_heatmap[index];
                if (score < scoreThreshold_) {
                    continue;
                }
                float s0 = 4 * exp(mat_scale.channel(0)[index]);
                float s1 = 4 * exp(mat_scale.channel(1)[index]);
                float o0 = mat_offset.channel(0)[index];
                float o1 = mat_offset.channel(1)[index];

                float ymin = MAX(0, 4 * (h + o0 + 0.5) - 0.5 * s0);
                float xmin = MAX(0, 4 * (w + o1 + 0.5) - 0.5 * s1);
                float ymax = MIN(ymin + s0, img_height_new);
                float xmax = MIN(xmin + s1, img_width_new);

                FaceInfo face_info;
                face_info.score_ = score;
                face_info.location_.x = static_cast<int>(scale_x * xmin);
                face_info.location_.y = static_cast<int>(scale_y * ymin);
                face_info.location_.width = static_cast<int>(scale_x * (xmax - xmin));
                face_info.location_.height = static_cast<int>(scale_y * (ymax - ymin));

                for (int num = 0; num < 5; ++num) {
                    face_info.keypoints_[num].x = scale_x * (s1 * mat_landmark.channel(2 * num + 1)[index] + xmin);
                    face_info.keypoints_[num].y = scale_y * (s0 * mat_landmark.channel(2 * num + 0)[index] + ymin);
                }
                faces.push_back(face_info);
            }
        }
        return 0;
    }
}
