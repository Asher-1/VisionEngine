#include "MtcnnFace.h"
#include <iostream>
#include "opencv2/imgproc.hpp"

#include <ncnn/net.h>

namespace mirror {
    MtcnnFace::MtcnnFace(FaceDetectorType type) : Detector(type),
                                                  pnet_(net_),
                                                  rnet_(new ncnn::Net()),
                                                  onet_(new ncnn::Net()),
                                                  pnet_size_(12),
                                                  min_face_size_(40),
                                                  scale_factor_(0.709f) {
    }

    MtcnnFace::~MtcnnFace() {
        if (pnet_) {
            // pnet_ should be managed by Detector class
            pnet_ = nullptr;
        }
        if (rnet_) {
            rnet_->clear();
            delete rnet_;
            rnet_ = nullptr;
        }
        if (onet_) {
            onet_->clear();
            delete onet_;
            onet_ = nullptr;
        }
    }

    int MtcnnFace::loadModel(const char *root_path) {
        if (!pnet_ || !rnet_ || !onet_) return ErrorCode::NULL_ERROR;
        std::string pnet_param = std::string(root_path) + modelPath_ + "/mtcnn/pnet.param";
        std::string pnet_bin = std::string(root_path) + modelPath_ + "/mtcnn/pnet.bin";

        if (pnet_->load_param(pnet_param.c_str()) == -1 ||
            pnet_->load_model(pnet_bin.c_str()) == -1) {
            std::cout << "Load pnet model failed." << std::endl;
            std::cout << "pnet param: " << pnet_param << std::endl;
            std::cout << "pnet bin: " << pnet_bin << std::endl;
            return ErrorCode::MODEL_LOAD_ERROR;
        }
        std::string rnet_param = std::string(root_path) + modelPath_ + "/mtcnn/rnet.param";
        std::string rnet_bin = std::string(root_path) + modelPath_ + "/mtcnn/rnet.bin";
        if (rnet_->load_param(rnet_param.c_str()) == -1 ||
            rnet_->load_model(rnet_bin.c_str()) == -1) {
            std::cout << "Load rnet model failed." << std::endl;
            std::cout << "rnet param: " << rnet_param << std::endl;
            std::cout << "rnet bin: " << rnet_bin << std::endl;
            return ErrorCode::MODEL_LOAD_ERROR;
        }
        std::string onet_param = std::string(root_path) + modelPath_ + "/mtcnn/onet.param";
        std::string onet_bin = std::string(root_path) + modelPath_ + "/mtcnn/onet.bin";
        if (onet_->load_param(onet_param.c_str()) == -1 ||
            onet_->load_model(onet_bin.c_str()) == -1) {
            std::cout << "Load onet model failed." << std::endl;
            std::cout << "onet param: " << onet_param << std::endl;
            std::cout << "onet bin: " << onet_bin << std::endl;
            return ErrorCode::MODEL_LOAD_ERROR;
        }
        return 0;
    }

#if defined __ANDROID__
    int MtcnnFace::loadModel(AAssetManager *mgr) {
        if (!pnet_ || !rnet_ || !onet_) return ErrorCode::NULL_ERROR;

        std::string pnet_param = "models" + modelPath_ + "/mtcnn/pnet.param";
        std::string pnet_bin = "models" + modelPath_ + "/mtcnn/pnet.bin";
        if (net_->load_param(mgr, pnet_param.c_str()) == -1 ||
            net_->load_model(mgr, pnet_bin.c_str()) == -1) {
            std::cout << "Load pnet model failed." << std::endl;
            std::cout << "pnet param: " << pnet_param << std::endl;
            std::cout << "pnet bin: " << pnet_bin << std::endl;
            return ErrorCode::MODEL_LOAD_ERROR;
        }
        std::string rnet_param = "models" + modelPath_ + "/mtcnn/rnet.param";
        std::string rnet_bin = "models" + modelPath_ + "/mtcnn/rnet.bin";
        if (rnet_->load_param(mgr, rnet_param.c_str()) == -1 ||
            rnet_->load_model(mgr, rnet_bin.c_str()) == -1) {
            std::cout << "Load rnet model failed." << std::endl;
            std::cout << "rnet param: " << rnet_param << std::endl;
            std::cout << "rnet bin: " << rnet_bin << std::endl;
            return ErrorCode::MODEL_LOAD_ERROR;
        }
        std::string onet_param = "models" + modelPath_ + "/mtcnn/onet.param";
        std::string onet_bin = "models" + modelPath_ + "/mtcnn/onet.bin";
        if (onet_->load_param(mgr, onet_param.c_str()) == -1 ||
            onet_->load_model(mgr, onet_bin.c_str()) == -1) {
            std::cout << "Load onet model failed." << std::endl;
            std::cout << "onet param: " << onet_param << std::endl;
            std::cout << "onet bin: " << onet_bin << std::endl;
            return ErrorCode::MODEL_LOAD_ERROR;
        }
        return 0;
    }
#endif


    int MtcnnFace::detectFace(const cv::Mat &img_src,
                              std::vector<FaceInfo> &faces) const {
        cv::Size max_size = cv::Size(img_src.cols, img_src.rows);
        cv::Mat img_cpy = img_src.clone();
        ncnn::Mat img_in = ncnn::Mat::from_pixels(img_src.data,
                                                  ncnn::Mat::PIXEL_BGR2RGB,
                                                  img_src.cols, img_src.rows);
        img_in.substract_mean_normalize(meanVals, normVals);

        std::vector<FaceInfo> first_bboxes, second_bboxes;
        std::vector<FaceInfo> first_bboxes_result;
        PDetect(img_in, first_bboxes);
        NMS(first_bboxes, first_bboxes_result, nms_threshold_[0]);
        Refine(first_bboxes_result, max_size);

        RDetect(img_in, first_bboxes_result, second_bboxes);
        std::vector<FaceInfo> second_bboxes_result;
        NMS(second_bboxes, second_bboxes_result, nms_threshold_[1]);
        Refine(second_bboxes_result, max_size);

        std::vector<FaceInfo> third_bboxes;
        ODetect(img_in, second_bboxes_result, third_bboxes);
        NMS(third_bboxes, faces, nms_threshold_[2], "MIN");
        Refine(faces, max_size);
        return 0;
    }

    int MtcnnFace::PDetect(const ncnn::Mat &img_in,
                           std::vector<FaceInfo> &first_bboxes) const {
        first_bboxes.clear();
        int width = img_in.w;
        int height = img_in.h;
        float min_side = MIN(width, height);
        float curr_scale = float(pnet_size_) / min_face_size_;
        min_side *= curr_scale;
        std::vector<float> scales;
        while (min_side > pnet_size_) {
            scales.push_back(curr_scale);
            min_side *= scale_factor_;
            curr_scale *= scale_factor_;
        }

        // mutiscale resize the image
        for (float scale : scales) {
            int new_w = static_cast<int>(width * scale);
            int new_h = static_cast<int>(height * scale);
            ncnn::Mat img_resized;
            ncnn::resize_bilinear(img_in, img_resized, new_w, new_h);
            ncnn::Extractor ex = pnet_->create_extractor();
            //ex.set_num_threads(2);
            ex.set_light_mode(true);
#if NCNN_VULKAN
            if (this->gpu_mode_) {
                ex.set_vulkan_compute(this->gpu_mode_);
            }
#endif
            ex.input("data", img_resized);
            ncnn::Mat score_mat, location_mat;
            ex.extract("prob1", score_mat);
            ex.extract("conv4-2", location_mat);
            const int stride = 2;
            const int cell_size = 12;
            for (int h = 0; h < score_mat.h; ++h) {
                for (int w = 0; w < score_mat.w; ++w) {
                    int index = h * score_mat.w + w;
                    // pnet output: 1x1x2  no-face && face
                    // face score: channel(1)
                    float score = score_mat.channel(1)[index];
                    if (score < threshold_[0]) continue;

                    // 1. generated bounding box
                    int x1 = static_cast<int>(round((stride * w + 1) / scale));
                    int y1 = static_cast<int>(round((stride * h + 1) / scale));
                    int x2 = static_cast<int>(round((stride * w + 1 + cell_size) / scale));
                    int y2 = static_cast<int>(round((stride * h + 1 + cell_size) / scale));

                    // 2. regression bounding box
                    float x1_reg = location_mat.channel(0)[index];
                    float y1_reg = location_mat.channel(1)[index];
                    float x2_reg = location_mat.channel(2)[index];
                    float y2_reg = location_mat.channel(3)[index];

                    int bbox_width = x2 - x1 + 1;
                    int bbox_height = y2 - y1 + 1;

                    FaceInfo face_info;
                    face_info.score_ = score;
                    face_info.location_.x = static_cast<int>(x1 + x1_reg * bbox_width);
                    face_info.location_.y = static_cast<int>(y1 + y1_reg * bbox_height);
                    face_info.location_.width = static_cast<int>(x2 + x2_reg * bbox_width - face_info.location_.x);
                    face_info.location_.height = static_cast<int>(y2 + y2_reg * bbox_height - face_info.location_.y);
                    face_info.location_ = face_info.location_ & cv::Rect(0, 0, width, height);
                    first_bboxes.push_back(face_info);
                }
            }
        }
        return 0;
    }

    int MtcnnFace::RDetect(const ncnn::Mat &img_in,
                           const std::vector<FaceInfo> &first_bboxes,
                           std::vector<FaceInfo> &second_bboxes) const {
        second_bboxes.clear();
        for (const auto &first_bboxe : first_bboxes) {
            cv::Rect face = first_bboxe.location_ & cv::Rect(0, 0, img_in.w, img_in.h);
            ncnn::Mat img_face, img_resized;
            ncnn::copy_cut_border(img_in, img_face, face.y, img_in.h - face.br().y,
                                  face.x, img_in.w - face.br().x);
            ncnn::resize_bilinear(img_face, img_resized, 24, 24);
            ncnn::Extractor ex = rnet_->create_extractor();
            ex.set_light_mode(true);
            ex.set_num_threads(2);
#if NCNN_VULKAN
            if (this->gpu_mode_) {
                ex.set_vulkan_compute(this->gpu_mode_);
            }
#endif
            ex.input("data", img_resized);
            ncnn::Mat score_mat, location_mat;
            ex.extract("prob1", score_mat);
            ex.extract("conv5-2", location_mat);
            float score = score_mat[1];
            if (score < threshold_[1]) continue;
            float x_reg = location_mat[0];
            float y_reg = location_mat[1];
            float w_reg = location_mat[2];
            float h_reg = location_mat[3];

            FaceInfo face_info;
            face_info.score_ = score;
            face_info.location_.x = static_cast<int>(face.x + x_reg * face.width);
            face_info.location_.y = static_cast<int>(face.y + y_reg * face.height);
            face_info.location_.width = static_cast<int>(face.x + face.width +
                                                         w_reg * face.width - face_info.location_.x);
            face_info.location_.height = static_cast<int>(face.y + face.height +
                                                          h_reg * face.height - face_info.location_.y);
            second_bboxes.push_back(face_info);
        }
        return 0;
    }

    int MtcnnFace::ODetect(const ncnn::Mat &img_in,
                           const std::vector<FaceInfo> &second_bboxes,
                           std::vector<FaceInfo> &third_bboxes) const {
        third_bboxes.clear();
        for (const auto &second_bboxe : second_bboxes) {
            cv::Rect face = second_bboxe.location_ & cv::Rect(0, 0, img_in.w, img_in.h);
            ncnn::Mat img_face, img_resized;
            ncnn::copy_cut_border(img_in, img_face, face.y,
                                  img_in.h - face.br().y, face.x,
                                  img_in.w - face.br().x);
            ncnn::resize_bilinear(img_face, img_resized, 48, 48);

            ncnn::Extractor ex = onet_->create_extractor();
            ex.set_light_mode(true);
            ex.set_num_threads(2);
#if NCNN_VULKAN
            if (this->gpu_mode_) {
                ex.set_vulkan_compute(this->gpu_mode_);
            }
#endif
            ex.input("data", img_resized);
            ncnn::Mat score_mat, location_mat, keypoints_mat;
            ex.extract("prob1", score_mat);
            ex.extract("conv6-2", location_mat);
            ex.extract("conv6-3", keypoints_mat);
            float score = score_mat[1];
            if (score < threshold_[1]) continue;
            float x_reg = location_mat[0];
            float y_reg = location_mat[1];
            float w_reg = location_mat[2];
            float h_reg = location_mat[3];

            FaceInfo face_info;
            face_info.score_ = score;
            face_info.location_.x = static_cast<int>(face.x + x_reg * face.width);
            face_info.location_.y = static_cast<int>(face.y + y_reg * face.height);
            face_info.location_.width = static_cast<int>(face.x + face.width +
                                                         w_reg * face.width - face_info.location_.x);
            face_info.location_.height = static_cast<int>(face.y + face.height +
                                                          h_reg * face.height - face_info.location_.y);

            for (int num = 0; num < 5; num++) {
                face_info.keypoints_[num].x = face.x + face.width * keypoints_mat[num];
                face_info.keypoints_[num].y = face.y + face.height * keypoints_mat[num + 5];
            }

            third_bboxes.push_back(face_info);
        }
        return 0;
    }

    int MtcnnFace::Refine(std::vector<FaceInfo> &bboxes, const cv::Size &max_size) const {
        int num_boxes = static_cast<int>(bboxes.size());
        for (int i = 0; i < num_boxes; ++i) {
            FaceInfo face_info = bboxes.at(i);
            int width = face_info.location_.width;
            int height = face_info.location_.height;
            int max_side = MAX(width, height);

            face_info.location_.x = static_cast<int>(face_info.location_.x + 0.5 * width - 0.5 * max_side);
            face_info.location_.y = static_cast<int>(face_info.location_.y + 0.5 * height - 0.5 * max_side);
            face_info.location_.width = max_side;
            face_info.location_.height = max_side;
            face_info.location_ = face_info.location_ & cv::Rect(0, 0, max_size.width, max_size.height);
            bboxes.at(i) = face_info;
        }

        return 0;
    }

}
