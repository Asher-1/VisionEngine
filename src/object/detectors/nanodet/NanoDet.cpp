#include "NanoDet.h"

#include <vector>
#include <string>
#include <opencv2/core.hpp>

#include <ncnn/net.h>

namespace mirror {

    inline float fast_exp(float x) {
        union {
            uint32_t i;
            float f;
        } v{};
        v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
        return v.f;
    }

    float sigmoid(float x) {
        return 1.0f / (1.0f + fast_exp(-x));
    }

    template<typename _Tp>
    int activation_function_softmax(const _Tp *src, _Tp *dst, int length) {
        const _Tp alpha = *std::max_element(src, src + length);
        _Tp denominator{0};

        for (int i = 0; i < length; ++i) {
            dst[i] = fast_exp(src[i] - alpha);
            denominator += dst[i];
        }

        for (int i = 0; i < length; ++i) {
            dst[i] /= denominator;
        }

        return 0;
    }

    NanoDet::NanoDet(ObjectDetectorType type) : ObjectDetector(type) {
        scoreThreshold_ = 0.4f;
        nmsThreshold_ = 0.6f;
        inputSize_ = cv::Size(320, 320);
        class_names_ = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    }

    int NanoDet::loadModel(const char *root_path) {
        net_->opt.use_bf16_storage = true;
        net_->opt.use_fp16_packed = true;
        net_->opt.use_fp16_storage = true;
        std::string subdir = std::string(root_path) + modelPath_ + "/nanodet/";
        std::string obj_param = subdir + "nanodet_m.param";
        std::string obj_bin = subdir + "nanodet_m.bin";
        return Super::loadModel(obj_param.c_str(), obj_bin.c_str());
    }

#if defined __ANDROID__
    int NanoDet::loadModel(AAssetManager *mgr) {
        net_->opt.use_bf16_storage = true;
        net_->opt.use_fp16_packed = true;
        net_->opt.use_fp16_storage = true;
        std::string subdir = "models" + modelPath_ + "/nanodet/";
        std::string obj_param = subdir + "nanodet_m.param";
        std::string obj_bin = subdir + "nanodet_m.bin";
        return Super::loadModel(mgr, obj_param.c_str(), obj_bin.c_str());
    }
#endif

    int NanoDet::detectObject(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const {
        int img_width = img_src.cols;
        int img_height = img_src.rows;
        float width_ratio = (float) img_width / (float) inputSize_.width;
        float height_ratio = (float) img_height / (float) inputSize_.height;
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR, img_width,
                                                     img_height, inputSize_.width, inputSize_.height);
        in.substract_mean_normalize(meanVals, normVals);

        ncnn::Extractor ex = net_->create_extractor();
#if NCNN_VULKAN
        if (this->gpu_mode_) {
            ex.set_vulkan_compute(this->gpu_mode_);
        }
#endif

        ex.input("input.1", in);

        std::vector<std::vector<ObjectInfo>> results;
        results.resize(class_names_.size());
        for (const auto &head_info : this->heads_info) {
            ncnn::Mat dis_pred;
            ncnn::Mat cls_pred;
            ex.extract(head_info.dis_layer.c_str(), dis_pred);
            ex.extract(head_info.cls_layer.c_str(), cls_pred);

            decode_infer(cls_pred, dis_pred, head_info.stride, scoreThreshold_, results, width_ratio, height_ratio);
        }

        objects.clear();
        for (int i = 0; i < (int) results.size(); i++) {
            nms(results[i], nmsThreshold_);
            for (auto box : results[i]) {
                objects.push_back(box);
            }
        }
        return ErrorCode::SUCCESS;
    }


    void NanoDet::decode_infer(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred, int stride, float threshold,
                               std::vector<std::vector<ObjectInfo>> &results, float width_ratio,
                               float height_ratio) const {
        int feature_h = inputSize_.height / stride;
        int feature_w = inputSize_.width / stride;

        //cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
        for (int idx = 0; idx < feature_h * feature_w; idx++) {
            const float *scores = cls_pred.row(idx);
            int row = idx / feature_w;
            int col = idx % feature_w;
            float score = 0;
            int cur_label = 0;
            for (int label = 0; label < class_names_.size(); label++) {
                if (scores[label] > score) {
                    score = scores[label];
                    cur_label = label;
                }
            }
            if (score > threshold) {
                //std::cout << "label:" << cur_label << " score:" << score << std::endl;
                const float *bbox_pred = dis_pred.row(idx);
                results[cur_label].push_back(
                        this->disPred2Bbox(bbox_pred, cur_label, score, col, row, stride, width_ratio, height_ratio));
                //debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
                //cv::imshow("debug", debug_heatmap);
            }

        }
    }

    ObjectInfo NanoDet::disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y,
                                     int stride, float width_ratio, float height_ratio) const {
        float ct_x = (x + 0.5) * stride;
        float ct_y = (y + 0.5) * stride;
        std::vector<float> dis_pred;
        dis_pred.resize(4);
        for (int i = 0; i < 4; i++) {
            float dis = 0;
            float *dis_after_sm = new float[regMax + 1];
            activation_function_softmax(dfl_det + i * (regMax + 1), dis_after_sm, regMax + 1);
            for (int j = 0; j < regMax + 1; j++) {
                dis += j * dis_after_sm[j];
            }
            dis *= stride;
            //std::cout << "dis:" << dis << std::endl;
            dis_pred[i] = dis;
            delete[] dis_after_sm;
        }

        int xmin = static_cast<int>((std::max)(ct_x - dis_pred[0], .0f) * width_ratio);
        int ymin = static_cast<int>((std::max)(ct_y - dis_pred[1], .0f) * height_ratio);
        int xmax = static_cast<int>((std::min)(ct_x + dis_pred[2], (float) inputSize_.width) * width_ratio);
        int ymax = static_cast<int>((std::min)(ct_y + dis_pred[3], (float) inputSize_.height) * height_ratio);

        //std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
        return ObjectInfo{cv::Rect(cv::Point2i(xmin, ymin), cv::Point2i(xmax, ymax)),
                          score, class_names_[label]};
    }

    void NanoDet::nms(std::vector<ObjectInfo> &input_boxes, float NMS_THRESH) {
        std::sort(input_boxes.begin(), input_boxes.end(),
                  [](ObjectInfo a, ObjectInfo b) { return a.score_ > b.score_; });
        std::vector<float> vArea(input_boxes.size());
        for (int i = 0; i < int(input_boxes.size()); ++i) {
            vArea[i] = (input_boxes.at(i).location_.width + 1)
                       * (input_boxes.at(i).location_.height + 1);
        }
        for (int i = 0; i < int(input_boxes.size()); ++i) {
            for (int j = i + 1; j < int(input_boxes.size());) {
                float xx1 = (std::max)(input_boxes[i].location_.x, input_boxes[j].location_.x);
                float yy1 = (std::max)(input_boxes[i].location_.y, input_boxes[j].location_.y);
                float xx2 = (std::min)(input_boxes[i].location_.br().x, input_boxes[j].location_.br().x);
                float yy2 = (std::min)(input_boxes[i].location_.br().y, input_boxes[j].location_.br().y);
                float w = (std::max)(float(0), xx2 - xx1 + 1);
                float h = (std::max)(float(0), yy2 - yy1 + 1);
                float inter = w * h;
                float ovr = inter / (vArea[i] + vArea[j] - inter);
                if (ovr >= NMS_THRESH) {
                    input_boxes.erase(input_boxes.begin() + j);
                    vArea.erase(vArea.begin() + j);
                } else {
                    j++;
                }
            }
        }
    }


}