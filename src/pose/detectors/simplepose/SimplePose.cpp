#include "SimplePose.h"

#include <vector>
#include <string>
#include <opencv2/core.hpp>

#include <ncnn/net.h>


namespace mirror {

    SimplePose::SimplePose(PoseEstimationType type) : PoseDetector(type),
                                                      PersonNet(new ncnn::Net()) {
        inputSize_ = cv::Size(192, 256);
        // 0 nose, 1 left_eye, 2 right_eye, 3 left_Ear, 4 right_Ear, 5 left_Shoulder, 6 rigth_Shoulder,
        // 7 left_Elbow, 8 right_Elbow, 9 left_Wrist, 10 right_Wrist, 11 left_Hip, 12 right_Hip,
        // 13 left_Knee, 14 right_Knee, 15 left_Ankle, 16 right_Ankle
        joint_pairs_ = {{0,  1},
                        {1,  3},
                        {0,  2},
                        {2,  4},
                        {5,  6},
                        {5,  7},
                        {7,  9},
                        {6,  8},
                        {8,  10},
                        {5,  11},
                        {6,  12},
                        {11, 12},
                        {11, 13},
                        {12, 14},
                        {13, 15},
                        {14, 16}};
    }

    SimplePose::~SimplePose() {
        if (PersonNet) {
            PersonNet->clear();
            delete PersonNet;
            PersonNet = nullptr;
        }
    }

    int SimplePose::loadModel(const char *root_path) {
        // load person detector
        PersonNet->opt.use_vulkan_compute = this->gpu_mode_;  // gpu
        std::string person_param = std::string(root_path) + modelPath_ + "/simplepose/person_detector.param";
        std::string person_bin = std::string(root_path) + modelPath_ + "/simplepose/person_detector.bin";
        if (PersonNet->load_param(person_param.c_str()) == -1 ||
            PersonNet->load_model(person_bin.c_str()) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        // load pose detector
        std::string pose_param = std::string(root_path) + modelPath_ + "/simplepose/Ultralight-Nano-SimplePose.param";
        std::string pose_bin = std::string(root_path) + modelPath_ + "/simplepose/Ultralight-Nano-SimplePose.bin";
        return Super::loadModel(pose_param.c_str(), pose_bin.c_str());
    }

#if defined __ANDROID__
    int SimplePose::loadModel(AAssetManager *mgr) {
        // load person detector
        if (ncnn::get_gpu_count() != 0) {
            PersonNet->opt.use_vulkan_compute = this->gpu_mode_;  // gpu
        }
        PersonNet->opt.use_fp16_arithmetic = true;  // fp16运算加速
        std::string sub_dir = "models";
        std::string person_param = sub_dir + modelPath_ + "/simplepose/person_detector.param";
        std::string person_bin = sub_dir + modelPath_ + "/simplepose/person_detector.bin";
        if (PersonNet->load_param(mgr, person_param.c_str()) == -1 ||
            PersonNet->load_model(mgr, person_bin.c_str()) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        // load pose detector
        std::string pose_param = sub_dir + modelPath_ + "/simplepose/Ultralight-Nano-SimplePose.param";
        std::string pose_bin = sub_dir + modelPath_ + "/simplepose/Ultralight-Nano-SimplePose.bin";
        return Super::loadModel(mgr, pose_param.c_str(), pose_bin.c_str());
    }
#endif

    int SimplePose::detectPose(const cv::Mat &img_src, std::vector<PoseResult> &poses) const {
        cv::Mat img_cpy = img_src.clone();
        int img_width = img_cpy.cols;
        int img_height = img_cpy.rows;
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_cpy.data,
                                                     ncnn::Mat::PIXEL_BGR2RGB,
                                                     img_width,
                                                     img_height,
                                                     detector_size_width,
                                                     detector_size_height);
        in.substract_mean_normalize(mean, norm);

        auto ex = PersonNet->create_extractor();
        if (this->gpu_mode_) {  // 消除提示
            ex.set_vulkan_compute(this->gpu_mode_);
        }
        ex.input("data", in);
        ncnn::Mat out;
        ex.extract("output", out);

        poses.clear();
        for (int i = 0; i < out.h; i++) {
            float x1, y1, x2, y2, score, label;
            float pw, ph, cx, cy;
            const float *values = out.row(i);

            score = values[1];
            label = values[0];

            x1 = values[2] * img_width  ;
            y1 = values[3] * img_height ;
            x2 = values[4] * img_width  ;
            y2 = values[5] * img_height ;
            if (std::isnan(x1) || std::isnan(y1) || std::isnan(x2) || std::isnan(y2)) {
                continue;
            }

            pw = x2 - x1;
            ph = y2 - y1;
            cx = x1 + 0.5 * pw;
            cy = y1 + 0.5 * ph;

            x1 = cx - 0.7 * pw;
            y1 = cy - 0.6 * ph;
            x2 = cx + 0.7 * pw;
            y2 = cy + 0.6 * ph;

            if (x1 < 0) x1 = 0;
            if (y1 < 0) y1 = 0;
            if (x2 < 0) x2 = 0;
            if (y2 < 0) y2 = 0;

            if (x1 > img_width) x1 = img_width;
            if (y1 > img_height) y1 = img_height;
            if (x2 > img_width) x2 = img_width;
            if (y2 > img_height) y2 = img_height;

            PoseResult poseResult;

            // person ROI
            cv::Mat roi = img_cpy(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
            this->runPose(roi, x1, y1, poseResult.keyPoints);

            poseResult.boxInfos.location_.x = x1;
            poseResult.boxInfos.location_.y = y1;
            poseResult.boxInfos.location_.width = x2 - x1;
            poseResult.boxInfos.location_.height = y2 - y1;
            poseResult.boxInfos.name_ = std::to_string(label);
            poseResult.boxInfos.score_ = score;

            poses.push_back(poseResult);
        }
        return 0;
    }

    int SimplePose::runPose(cv::Mat &roi, float x1, float y1, std::vector<KeyPoint> &keypoints) const {
        int w = roi.cols;
        int h = roi.rows;
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(roi.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                     w, h, inputSize_.width, inputSize_.height);
        in.substract_mean_normalize(meanVals, normVals);

        auto ex = net_->create_extractor();
        if (this->gpu_mode_) {  // 消除提示
            ex.set_vulkan_compute(this->gpu_mode_);
        }
        ex.input("data", in);
        ncnn::Mat out;
        ex.extract("hybridsequential0_conv7_fwd", out);
        keypoints.clear();

        for (int p = 0; p < out.c; p++) {
            const ncnn::Mat m = out.channel(p);

            float max_prob = 0.f;
            int max_x = 0;
            int max_y = 0;
            for (int y = 0; y < out.h; y++) {
                const float *ptr = m.row(y);
                for (int x = 0; x < out.w; x++) {
                    float prob = ptr[x];
                    if (prob > max_prob) {
                        max_prob = prob;
                        max_x = x;
                        max_y = y;
                    }
                }
            }

            KeyPoint keypoint;
            keypoint.p = cv::Point2f(max_x * w / (float) out.w + x1, max_y * h / (float) out.h + y1);
            keypoint.prob = max_prob;
            keypoints.push_back(keypoint);
        }
        return 0;
    }

}
