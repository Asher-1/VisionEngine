#include "MobileNetV3Seg.h"

#include <vector>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <ncnn/net.h>

namespace mirror {
    MobileNetV3Seg::MobileNetV3Seg(SegmentType type) : SegmentDetector(type) {
        inputSize_ = cv::Size(512, 512);
        class_names_ = {"road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                        "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
                        "motorcycle", "bicycle"};
    }

    int MobileNetV3Seg::loadModel(const char *root_path) {
        std::string pose_param = std::string(root_path) + modelPath_ + "/mobilenetv3seg/mbnv3_small.param";
        std::string pose_bin = std::string(root_path) + modelPath_ + "/mobilenetv3seg/mbnv3_small.bin";
        return Super::loadModel(pose_param.c_str(), pose_bin.c_str());
    }

#if defined __ANDROID__
    int MobileNetV3Seg::loadModel(AAssetManager *mgr) {
        std::string pose_param = "models" + modelPath_ + "/mobilenetv3seg/mbnv3_small.param";
        std::string pose_bin = "models" + modelPath_ + "/mobilenetv3seg/mbnv3_small.bin";
        return Super::loadModel(mgr, pose_param.c_str(), pose_bin.c_str());
    }
#endif

    int MobileNetV3Seg::detectSeg(const cv::Mat &img_src, std::vector<SegmentInfo> &segments) const {
        cv::Mat img_cpy = img_src.clone();
        int img_width = img_cpy.cols;
        int img_height = img_cpy.rows;
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_cpy.data,
                                                     ncnn::Mat::PIXEL_BGR2RGB,
                                                     img_width,
                                                     img_height,
                                                     inputSize_.width,
                                                     inputSize_.height);
        in.substract_mean_normalize(meanVals, normVals);

        ncnn::Mat maskout;

        auto ex = net_->create_extractor();
#if NCNN_VULKAN
        if (this->gpu_mode_) {
            ex.set_vulkan_compute(this->gpu_mode_);
        }
#endif

        ex.input("input", in);
        ex.extract("output", maskout);

        int mask_c = maskout.c;
        int mask_w = maskout.w;
        int mask_h = maskout.h;

        cv::Mat prediction = cv::Mat::zeros(mask_h, mask_w, CV_8UC1);
        ncnn::Mat chn[mask_c];
        for (int i = 0; i < mask_c; i++) {
            chn[i] = maskout.channel(i);
        }
        for (int i = 0; i < mask_h; i++) {
            const float *pChn[mask_c];
            for (int c = 0; c < mask_c; c++) {
                pChn[c] = chn[c].row(i);
            }

            auto *pCowMask = prediction.ptr<uchar>(i);

            for (int j = 0; j < mask_w; j++) {
                int maxindex = 0;
                float maxvalue = -1000;
                for (int n = 0; n < mask_c; n++) {
                    if (pChn[n][j] > maxvalue) {
                        maxindex = n;
                        maxvalue = pChn[n][j];
                    }
                }
                pCowMask[j] = maxindex;
            }
        }

        segments.clear();
        cv::Mat pred_resize;
        cv::resize(prediction, pred_resize, cv::Size(img_width, img_height),
                   0, 0, cv::INTER_NEAREST);
        ncnn::Mat maskMat;
        maskMat = ncnn::Mat::from_pixels_resize(pred_resize.data, ncnn::Mat::PIXEL_GRAY,
                                                pred_resize.cols, pred_resize.rows,
                                                img_width, img_height);

        SegmentInfo segment;
        segment.mask = cv::Mat(img_height, img_width, CV_8UC1);
        maskMat.to_pixels(segment.mask.data, ncnn::Mat::PIXEL_GRAY);
        segments.push_back(segment);

        return 0;
    }

}
