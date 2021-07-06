#include "Yolact.h"

#include <vector>
#include <string>
#include <opencv2/core.hpp>

#include <ncnn/net.h>


namespace mirror {

    static inline float intersection_area(const SegmentInfo &a, const SegmentInfo &b) {
        cv::Rect inter = a.boxInfo.location_ & b.boxInfo.location_;
        return inter.area();
    }

    static void qsort_descent_inplace(std::vector<SegmentInfo> &objects, int left, int right) {
        int i = left;
        int j = right;
        float p = objects[(left + right) / 2].boxInfo.score_;

        while (i <= j) {
            while (objects[i].boxInfo.score_ > p)
                i++;

            while (objects[j].boxInfo.score_ < p)
                j--;

            if (i <= j) {
                // swap
                std::swap(objects[i], objects[j]);

                i++;
                j--;
            }
        }

#pragma omp parallel sections
        {
#pragma omp section
            {
                if (left < j) qsort_descent_inplace(objects, left, j);
            }
#pragma omp section
            {
                if (i < right) qsort_descent_inplace(objects, i, right);
            }
        }
    }

    static void qsort_descent_inplace(std::vector<SegmentInfo> &objects) {
        if (objects.empty())
            return;

        qsort_descent_inplace(objects, 0, objects.size() - 1);
    }

    static void nms_sorted_bboxes(const std::vector<SegmentInfo> &objects,
                                  std::vector<int> &picked,
                                  float nms_threshold) {
        picked.clear();

        const int n = objects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++) {
            areas[i] = objects[i].boxInfo.location_.area();
        }

        for (int i = 0; i < n; i++) {
            const SegmentInfo &a = objects[i];

            int keep = 1;
            for (int j = 0; j < (int) picked.size(); j++) {
                const SegmentInfo &b = objects[picked[j]];

                // intersection over union
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                //             float IoU = inter_area / union_area
                if (inter_area / union_area > nms_threshold)
                    keep = 0;
            }

            if (keep)
                picked.push_back(i);
        }
    }


    Yolact::Yolact(SegmentType type) : SegmentDetector(type) {
        scoreThreshold_ = 0.05f;
        nmsThreshold_ = 0.5f;
        inputSize_ = cv::Size(550, 550);
        class_names_ = {"background", "person", "bicycle", "car", "motorcycle", "airplane",
                        "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                        "skis", "snowboard", "sports ball", "kite", "baseball bat",
                        "baseball glove", "skateboard", "surfboard", "tennis racket",
                        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                        "hot dog", "pizza", "donut", "cake", "chair", "couch",
                        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                        "toaster", "sink", "refrigerator", "book", "clock", "vase",
                        "scissors", "teddy bear", "hair drier", "toothbrush"};
    }

    int Yolact::loadModel(const char *root_path) {
        // load pose detector
        std::string seg_param = std::string(root_path) + modelPath_ + "/yolact/yolact.param";
        std::string seg_bin = std::string(root_path) + modelPath_ + "/yolact/yolact.bin";
        return Super::loadModel(seg_param.c_str(), seg_bin.c_str());
    }

#if defined __ANDROID__
    int Yolact::loadModel(AAssetManager *mgr) {
        // load pose detector
        std::string sub_dir = "models";
        std::string seg_param = sub_dir + modelPath_ + "/yolact/yolact.param";
        std::string seg_bin = sub_dir + modelPath_ + "/yolact/yolact.bin";
        return Super::loadModel(mgr, seg_param.c_str(), seg_bin.c_str());
    }
#endif

    int Yolact::detectSeg(const cv::Mat &img_src, std::vector<SegmentInfo> &segments) const {
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

        ncnn::Mat maskmaps;
        ncnn::Mat location;
        ncnn::Mat mask;
        ncnn::Mat confidence;
        ncnn::Extractor ex = net_->create_extractor();
#if NCNN_VULKAN
        if (this->gpu_mode_) {
            ex.set_vulkan_compute(this->gpu_mode_);
        }
#endif

        ex.input("input.1", in);
        ex.extract("619", maskmaps);   // 138x138 x 32
        ex.extract("816", location);   // 4 x 19248
        ex.extract("818", mask);       // maskdim 32 x 19248
        ex.extract("820", confidence); // 81 x 19248

        int num_class = confidence.w;
        int num_priors = confidence.h;

        // make priorbox
        ncnn::Mat priorbox(4, num_priors);
        {
            const int conv_ws[5] = {69, 35, 18, 9, 5};
            const int conv_hs[5] = {69, 35, 18, 9, 5};

            const float aspect_ratios[3] = {1.f, 0.5f, 2.f};
            const float scales[5] = {24.f, 48.f, 96.f, 192.f, 384.f};

            float *pb = priorbox;

            for (int p = 0; p < 5; p++) {
                int conv_w = conv_ws[p];
                int conv_h = conv_hs[p];

                float scale = scales[p];

                for (int i = 0; i < conv_h; i++) {
                    for (int j = 0; j < conv_w; j++) {
                        // +0.5 because priors are in center-size notation
                        float cx = (j + 0.5f) / conv_w;
                        float cy = (i + 0.5f) / conv_h;

                        for (int k = 0; k < 3; k++) {
                            float ar = aspect_ratios[k];

                            ar = sqrt(ar);

                            float w = scale * ar / inputSize_.width;
                            float h = scale / ar / inputSize_.height;

                            // This is for backward compatability with a bug where I made everything square by accident
                            // cfg.backbone.use_square_anchors:
                            h = w;

                            pb[0] = cx;
                            pb[1] = cy;
                            pb[2] = w;
                            pb[3] = h;

                            pb += 4;
                        }
                    }
                }
            }
        }

        std::vector<std::vector<SegmentInfo> > class_candidates;
        class_candidates.resize(num_class);

        for (int i = 0; i < num_priors; i++) {
            const float *conf = confidence.row(i);
            const float *loc = location.row(i);
            const float *pb = priorbox.row(i);
            const float *maskdata = mask.row(i);

            // find class id with highest score
            // start from 1 to skip background
            int label = 0;
            float score = 0.f;
            for (int j = 1; j < num_class; j++) {
                float class_score = conf[j];
                if (class_score > score) {
                    label = j;
                    score = class_score;
                }
            }

            // ignore background or low score
            if (label == 0 || score <= scoreThreshold_)
                continue;

            // CENTER_SIZE
            float var[4] = {0.1f, 0.1f, 0.2f, 0.2f};

            float pb_cx = pb[0];
            float pb_cy = pb[1];
            float pb_w = pb[2];
            float pb_h = pb[3];

            float bbox_cx = var[0] * loc[0] * pb_w + pb_cx;
            float bbox_cy = var[1] * loc[1] * pb_h + pb_cy;
            float bbox_w = (float) (exp(var[2] * loc[2]) * pb_w);
            float bbox_h = (float) (exp(var[3] * loc[3]) * pb_h);

            float obj_x1 = bbox_cx - bbox_w * 0.5f;
            float obj_y1 = bbox_cy - bbox_h * 0.5f;
            float obj_x2 = bbox_cx + bbox_w * 0.5f;
            float obj_y2 = bbox_cy + bbox_h * 0.5f;

            // clip
            obj_x1 = std::max(std::min(obj_x1 * img_width, (float) (img_width - 1)), 0.f);
            obj_y1 = std::max(std::min(obj_y1 * img_height, (float) (img_height - 1)), 0.f);
            obj_x2 = std::max(std::min(obj_x2 * img_width, (float) (img_width - 1)), 0.f);
            obj_y2 = std::max(std::min(obj_y2 * img_height, (float) (img_height - 1)), 0.f);

            // append object
            SegmentInfo obj;
            obj.boxInfo.location_ = cv::Rect(obj_x1, obj_y1, obj_x2 - obj_x1 + 1, obj_y2 - obj_y1 + 1);
            obj.boxInfo.name_ = class_names_[int(label)];
            obj.boxInfo.score_ = score;
            obj.maskData = std::vector<float>(maskdata, maskdata + mask.w);

            class_candidates[label].push_back(obj);
        }

        segments.clear();
        for (int i = 0; i < (int) class_candidates.size(); i++) {
            std::vector<SegmentInfo> &candidates = class_candidates[i];

            qsort_descent_inplace(candidates);

            std::vector<int> picked;
            nms_sorted_bboxes(candidates, picked, nmsThreshold_);

            for (int j = 0; j < (int) picked.size(); j++) {
                int z = picked[j];
                segments.push_back(candidates[z]);
            }
        }

        qsort_descent_inplace(segments);

        // keep_top_k
        if (keep_top_k < (int) segments.size()) {
            segments.resize(keep_top_k);
        }

        // generate mask
        for (int i = 0; i < segments.size(); i++) {
            SegmentInfo &obj = segments[i];

            cv::Mat mask(maskmaps.h, maskmaps.w, CV_32FC1);
            {
                mask = cv::Scalar(0.f);

                for (int p = 0; p < maskmaps.c; p++) {
                    const float *maskmap = maskmaps.channel(p);
                    float coeff = obj.maskData[p];
                    float *mp = (float *) mask.data;

                    // mask += m * coeff
                    for (int j = 0; j < maskmaps.w * maskmaps.h; j++) {
                        mp[j] += maskmap[j] * coeff;
                    }
                }
            }

            cv::Mat mask2;
            cv::resize(mask, mask2, cv::Size(img_width, img_height));

            // crop obj box and binarize
            obj.mask = cv::Mat(img_height, img_width, CV_8UC1);
            {
                obj.mask = cv::Scalar(0);

                for (int y = 0; y < img_height; y++) {
                    if (y < obj.boxInfo.location_.y ||
                        y > obj.boxInfo.location_.y + obj.boxInfo.location_.height) {
                        continue;
                    }

                    const float *mp2 = mask2.ptr<const float>(y);
                    uchar *bmp = obj.mask.ptr<uchar>(y);

                    for (int x = 0; x < img_width; x++) {
                        if (x < obj.boxInfo.location_.x ||
                            x > obj.boxInfo.location_.x + obj.boxInfo.location_.width)
                            continue;

                        bmp[x] = mp2[x] > 0.5f ? 255 : 0;
                    }
                }
            }
        }

        return 0;
    }
}
