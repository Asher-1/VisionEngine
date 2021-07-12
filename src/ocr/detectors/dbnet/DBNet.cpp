#define _CRT_SECURE_NO_WARNINGS

#include "DBNet.h"
#include "ZUtil.h"

#include <algorithm>
#include <string>
#include <ncnn/net.h>

namespace mirror {
    DBNet::DBNet(TextDetectorType type) : TextDetector(type) {
        topk_ = 3;
        nmsThreshold_ = 0.45f;
        scoreThreshold_ = 0.5f;
        inputSize_ = cv::Size(640, 640);
    }

    int DBNet::loadModel(const char *root_path) {
        std::string root_dir = std::string(root_path) + modelPath_ + "/dbnet";
        std::string param_file = root_dir + "/dbnet_op.param";
        std::string model_file = root_dir + "/dbnet_op.bin";
        if (Super::loadModel(param_file.c_str(), model_file.c_str()) != 0) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }

#if defined __ANDROID__
    int DBNet::loadModel(AAssetManager *mgr) {
        std::string root_dir = "models" + modelPath_ + "/dbnet";
        std::string param_file = root_dir + "/dbnet_op.param";
        std::string model_file = root_dir + "/dbnet_op.bin";
        if (Super::loadModel(mgr, param_file.c_str(), model_file.c_str()) != 0) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }
#endif

    int DBNet::detectText(const cv::Mat &img_src, std::vector<TextBox> &textBoxes) const {
        cv::Mat img_cpy;
        cv::cvtColor(img_src, img_cpy, cv::COLOR_BGR2RGB);
        int img_width = img_cpy.cols;
        int img_height = img_cpy.rows;

        // pad to multiple of 32
        int w = img_width;
        int h = img_height;
        float scale = 1.f;
        if (w > h) {
            scale = (float) inputSize_.width / w;
            w = inputSize_.width;
            h = h * scale;
        } else {
            scale = (float) inputSize_.height / h;
            h = inputSize_.height;
            w = w * scale;
        }
        if (h % 32 != 0) {
            h = (h / 32 + 1) * 32;
        }
        if (w % 32 != 0) {
            w = (w / 32 + 1) * 32;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_cpy.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                     img_width, img_height, w, h);
        in.substract_mean_normalize(meanVals, normVals);

        ncnn::Extractor ex = net_->create_extractor();
#if NCNN_VULKAN
        if (this->gpu_mode_) {
            ex.set_vulkan_compute(this->gpu_mode_);
        }
#endif
        ex.input("input0", in);
        ncnn::Mat out;
        ex.extract("out1", out);

        //printf("c=%d,wid=%d,hi=%d\n",dbnet_out.c,dbnet_out.w,dbnet_out.h);
        cv::Mat fmapmat(h, w, CV_32FC1);
        memcpy(fmapmat.data, (float *) out.data, w * h * sizeof(float));

        cv::Mat norfmapmat;

        norfmapmat = fmapmat > binThresh;

        textBoxes.clear();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(norfmapmat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        for (auto & contour : contours) {
            float score = box_score_fast(fmapmat, contour);
            if (score < scoreThreshold_)
                continue;

            TextBox textBox;
            float minedgesize, alledgesize;
            get_mini_boxes(contour, textBox.box, minedgesize, alledgesize);
            if (minedgesize < min_size)
                continue;

            textBox.score = score;

            std::vector<cv::Point> newbox;
            unclip(textBox.box, alledgesize, newbox, unclip_ratio);

            get_mini_boxes(newbox, textBox.box, minedgesize, alledgesize);

            if (minedgesize < min_size + 2)
                continue;

            for (int j = 0; j < textBox.box.size(); ++j) {
                textBox.box[j].x = (textBox.box[j].x / (float) w * img_width);
                textBox.box[j].x = (std::min)((std::max)(textBox.box[j].x, 0), img_width);

                textBox.box[j].y = (textBox.box[j].y / (float) h * img_height);
                textBox.box[j].y = (std::min)((std::max)(textBox.box[j].y, 0), img_height);
            }

            textBoxes.push_back(textBox);
        }

        return 0;
    }

}


