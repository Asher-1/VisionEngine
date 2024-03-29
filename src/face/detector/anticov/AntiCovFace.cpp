#include "AntiCovFace.h"
#include <iostream>

#include <ncnn/net.h>

namespace mirror {
    AntiCovFace::AntiCovFace(FaceDetectorType type) : Detector(type) {
        iouThreshold_ = 0.4f;
        scoreThreshold_ = 0.8f;
        has_kps_ = false;
    }

    int AntiCovFace::loadModel(const char *root_path) {
        std::string fd_param = std::string(root_path) + modelPath_ + "/anticov/mask.param";
        std::string fd_bin = std::string(root_path) + modelPath_ + "/anticov/mask.bin";
        int flag = Super::loadModel(fd_param.c_str(), fd_bin.c_str());
        if (flag != 0) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        // generate anchors
        for (int i = 0; i < 3; ++i) {
            ANCHORS anchors;
            if (0 == i) {
                GenerateAnchors(16, {1.0f}, {32, 16}, anchors);
            } else if (1 == i) {
                GenerateAnchors(16, {1.0f}, {8, 4}, anchors);
            } else {
                GenerateAnchors(16, {1.0f}, {2, 1}, anchors);
            }
            anchors_generated_.push_back(anchors);
        }
        return 0;
    }

#if defined __ANDROID__
    int AntiCovFace::loadModel(AAssetManager *mgr) {
        std::string fd_param = "models" + modelPath_ + "/anticov/mask.param";
        std::string fd_bin = "models" + modelPath_ + "/anticov/mask.bin";
        int flag = Super::loadModel(mgr, fd_param.c_str(), fd_bin.c_str());
        if (flag != 0)
        {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        // generate anchors
        for (int i = 0; i < 3; ++i) {
            ANCHORS anchors;
            if (0 == i) {
                GenerateAnchors(16, {1.0f}, {32, 16}, anchors);
            } else if (1 == i) {
                GenerateAnchors(16, {1.0f}, {8, 4}, anchors);
            } else {
                GenerateAnchors(16, {1.0f}, {2, 1}, anchors);
            }
            anchors_generated_.push_back(anchors);
        }
        return 0;
    }
#endif


    int AntiCovFace::detectFace(const cv::Mat &img_src, std::vector<FaceInfo> &faces) const {

        cv::Mat img_cpy = img_src.clone();
        int img_width = img_cpy.cols;
        int img_height = img_cpy.rows;
        float factor_x = static_cast<float>(img_width) / inputSize_.width;
        float factor_y = static_cast<float>(img_height) / inputSize_.height;

        // pad to multiple of 32
        int w = img_width;
        int h = img_height;
        if (w > h) {
            w = inputSize_.width;
            h = h / factor_x;
            factor_y = factor_x;
        } else {
            h = inputSize_.height;
            w = w / factor_y;
            factor_x = factor_y;
        }

//        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_cpy.data,
//                                                     ncnn::Mat::PIXEL_BGR2RGB, img_width, img_height,
//                                                     inputSize_.width, inputSize_.height);

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_cpy.data,
                                                     ncnn::Mat::PIXEL_BGR2RGB,
                                                     img_width, img_height,
                                                     w, h);

        ncnn::Extractor ex = net_->create_extractor();
#if NCNN_VULKAN
        if (this->gpu_mode_) {
            ex.set_vulkan_compute(this->gpu_mode_);
        }
#endif
        ex.input("data", in);

        faces.clear();
        for (int i = 0; i < 3; ++i) {
            std::string class_layer_name = "face_rpn_cls_prob_reshape_stride" + std::to_string(RPNs_[i]);
            std::string bbox_layer_name = "face_rpn_bbox_pred_stride" + std::to_string(RPNs_[i]);
            std::string landmark_layer_name = "face_rpn_landmark_pred_stride" + std::to_string(RPNs_[i]);
            std::string type_layer_name = "face_rpn_type_prob_reshape_stride" + std::to_string(RPNs_[i]);

            ncnn::Mat class_mat, bbox_mat, landmark_mat, type_mat;
            ex.extract(class_layer_name.c_str(), class_mat);
            ex.extract(bbox_layer_name.c_str(), bbox_mat);
            ex.extract(landmark_layer_name.c_str(), landmark_mat);
            ex.extract(type_layer_name.c_str(), type_mat);

            ANCHORS anchors = anchors_generated_.at(i);
            int width = class_mat.w;
            int height = class_mat.h;
            int anchor_num = static_cast<int>(anchors.size());
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int index = h * width + w;
                    for (int a = 0; a < anchor_num; ++a) {
                        float score = class_mat.channel(anchor_num + a)[index];
                        if (score < scoreThreshold_) {
                            continue;
                        }
                        float prob = type_mat.channel(2 * anchor_num + a)[index];
                        cv::Rect box = cv::Rect(w * RPNs_[i] + anchors[a].x,
                                                h * RPNs_[i] + anchors[a].y,
                                                anchors[a].width,
                                                anchors[a].height);

                        float delta_x = bbox_mat.channel(a * 4 + 0)[index];
                        float delta_y = bbox_mat.channel(a * 4 + 1)[index];
                        float delta_w = bbox_mat.channel(a * 4 + 2)[index];
                        float delta_h = bbox_mat.channel(a * 4 + 3)[index];
                        cv::Point2f center = cv::Point2f(box.x + box.width * 0.5f,
                                                         box.y + box.height * 0.5f);
                        center.x = center.x + delta_x * box.width;
                        center.y = center.y + delta_y * box.height;
                        float curr_width = std::exp(delta_w) * (box.width + 1);
                        float curr_height = std::exp(delta_h) * (box.height + 1);
                        cv::Rect curr_box = cv::Rect(center.x - curr_width * 0.5f,
                                                     center.y - curr_height * 0.5f, curr_width, curr_height);
                        curr_box.x = MAX(curr_box.x * factor_x, 0);
                        curr_box.y = MAX(curr_box.y * factor_y, 0);
                        curr_box.width = MIN(img_width - curr_box.x, curr_box.width * factor_x);
                        curr_box.height = MIN(img_height - curr_box.y, curr_box.height * factor_y);

                        FaceInfo face_info;
                        memset(&face_info, 0, sizeof(face_info));
                        face_info.score_ = score;
                        face_info.mask_ = (prob > maskThreshold_);
                        face_info.location_ = curr_box;
                        faces.push_back(face_info);
                    }
                }
            }
        }

        return 0;
    }

}
