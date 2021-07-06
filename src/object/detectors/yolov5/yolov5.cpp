#include "yolov5.h"

#include <vector>
#include <string>
#include "opencv2/core.hpp"
#include "ncnn/net.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

class YoloV5Focus : public ncnn::Layer {
public:
    YoloV5Focus() {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++) {
            const float *ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float *outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)


namespace mirror {

    static inline float sigmoid(float x) {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }

    static void generate_proposals(const ncnn::Mat &anchors, int stride,
                                   const ncnn::Mat &in_pad, const ncnn::Mat &feat_blob,
                                   float prob_threshold, std::vector<ObjectInfo> &objects) {
        const int num_grid = feat_blob.h;

        int num_grid_x;
        int num_grid_y;
        if (in_pad.w > in_pad.h) {
            num_grid_x = in_pad.w / stride;
            num_grid_y = num_grid / num_grid_x;
        } else {
            num_grid_y = in_pad.h / stride;
            num_grid_x = num_grid / num_grid_y;
        }

        const int num_class = feat_blob.w - 5;

        const int num_anchors = anchors.w / 2;

        for (int q = 0; q < num_anchors; q++) {
            const float anchor_w = anchors[q * 2];
            const float anchor_h = anchors[q * 2 + 1];

            const ncnn::Mat feat = feat_blob.channel(q);

            for (int i = 0; i < num_grid_y; i++) {
                for (int j = 0; j < num_grid_x; j++) {
                    const float *featptr = feat.row(i * num_grid_x + j);

                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++) {
                        float score = featptr[5 + k];
                        if (score > class_score) {
                            class_index = k;
                            class_score = score;
                        }
                    }

                    float box_score = featptr[4];

                    float confidence = sigmoid(box_score) * sigmoid(class_score);

                    if (confidence >= prob_threshold) {
                        // yolov5/models/yolo.py Detect forward
                        // y = x[i].sigmoid()
                        // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                        // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);

                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;

                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;

                        ObjectInfo obj;
                        obj.location_.x = x0;
                        obj.location_.y = y0;
                        obj.location_.width = x1 - x0;
                        obj.location_.height = y1 - y0;
                        obj.score_ = confidence;
                        obj.name_ = std::to_string(class_index);
                        objects.push_back(obj);
                    }
                }
            }
        }
    }


    YoloV5::YoloV5(ObjectDetectorType type) : ObjectDetector(type) {
        scoreThreshold_ = 0.25f;
        nmsThreshold_ = 0.45f;
        net_->opt.lightmode = true;
        net_->opt.blob_allocator = &g_blob_pool_allocator;
        net_->opt.workspace_allocator = &g_workspace_pool_allocator;
        net_->opt.use_packing_layout = true;
        net_->register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
        inputSize_ = cv::Size(640, 640);
        class_names_ = {
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                "scissors", "teddy bear", "hair drier", "toothbrush"};

    }

    int YoloV5::loadModel(const char *root_path) {
        std::string obj_param = std::string(root_path) + modelPath_ + "/yolov5/yolov5s.param";
        std::string obj_bin = std::string(root_path) + modelPath_ + "/yolov5/yolov5s.bin";
        return Super::loadModel(obj_param.c_str(), obj_bin.c_str());
    }

#if defined __ANDROID__
    int YoloV5::loadModel(AAssetManager *mgr) {
        std::string sub_dir = "models";
        std::string obj_param = sub_dir + modelPath_ + "/yolov5/yolov5s.param";
        std::string obj_bin =  sub_dir + modelPath_ + "/yolov5/yolov5s.bin";
        return Super::loadModel(mgr, obj_param.c_str(), obj_bin.c_str());
    }
#endif

    int YoloV5::detectObject(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const {
        int width = img_src.cols;
        int height = img_src.rows;
        // letterbox pad to multiple of 32
        int w = width;
        int h = height;
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

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data,
                                                     ncnn::Mat::PIXEL_BGR2RGB, width, height, w, h);
        // pad to target_size rectangle
        // yolov5/utility/datasets.py letterbox
        int wpad = (w + 31) / 32 * 32 - w;
        int hpad = (h + 31) / 32 * 32 - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2,
                               wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

        // yolov5
        objects.clear();
        {
            in_pad.substract_mean_normalize(0, normVals);

            ncnn::Extractor ex = net_->create_extractor();

#if NCNN_VULKAN
            if (this->gpu_mode_) {
                ex.set_vulkan_compute(this->gpu_mode_);
            }
#endif

            ex.input("images", in_pad);

            // anchor setting from yolov5/models/yolov5s.yaml
            // stride 8
            {
                ncnn::Mat out;
                ex.extract("output", out);

                ncnn::Mat anchors(6);
                anchors[0] = 10.f;
                anchors[1] = 13.f;
                anchors[2] = 16.f;
                anchors[3] = 30.f;
                anchors[4] = 33.f;
                anchors[5] = 23.f;

                std::vector<ObjectInfo> objects8;
                generate_proposals(anchors, 8, in_pad, out, scoreThreshold_, objects8);

                objects.insert(objects.end(), objects8.begin(), objects8.end());
            }

            // stride 16
            {
                ncnn::Mat out;
                ex.extract("781", out);

                ncnn::Mat anchors(6);
                anchors[0] = 30.f;
                anchors[1] = 61.f;
                anchors[2] = 62.f;
                anchors[3] = 45.f;
                anchors[4] = 59.f;
                anchors[5] = 119.f;

                std::vector<ObjectInfo> objects16;
                generate_proposals(anchors, 16, in_pad, out, scoreThreshold_, objects16);

                objects.insert(objects.end(), objects16.begin(), objects16.end());
            }

            // stride 32
            {
                ncnn::Mat out;
                ex.extract("801", out);

                ncnn::Mat anchors(6);
                anchors[0] = 116.f;
                anchors[1] = 90.f;
                anchors[2] = 156.f;
                anchors[3] = 198.f;
                anchors[4] = 373.f;
                anchors[5] = 326.f;

                std::vector<ObjectInfo> objects32;
                generate_proposals(anchors, 32, in_pad, out, scoreThreshold_, objects32);

                objects.insert(objects.end(), objects32.begin(), objects32.end());
            }

            for (int i = 0; i < objects.size(); i++) {
                // adjust offset to original unpadded
                float x0 = (objects[i].location_.x - (wpad / 2)) / scale;
                float y0 = (objects[i].location_.y - (hpad / 2)) / scale;
                float x1 = (objects[i].location_.x + objects[i].location_.width - (wpad / 2)) / scale;
                float y1 = (objects[i].location_.y + objects[i].location_.height - (hpad / 2)) / scale;

                // clip
                x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
                y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
                x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
                y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

                objects[i].location_.x = static_cast<int>(x0);
                objects[i].location_.y = static_cast<int>(y0);
                objects[i].location_.width = static_cast<int>(x1 - x0);
                objects[i].location_.height = static_cast<int>(y1 - y0);
                objects[i].name_ = class_names_[std::stoi(objects[i].name_)];
            }
        }
        return 0;
    }

}
