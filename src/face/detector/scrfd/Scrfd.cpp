// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "Scrfd.h"

#include <opencv2/core.hpp>
#include <ncnn/net.h>

namespace mirror {
    // insightface/detection/scrfd/mmdet/core/anchor/anchor_generator.py gen_single_level_base_anchors()
    static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat &ratios, const ncnn::Mat &scales) {
        int num_ratio = ratios.w;
        int num_scale = scales.w;

        ncnn::Mat anchors;
        anchors.create(4, num_ratio * num_scale);

        const float cx = 0;
        const float cy = 0;

        for (int i = 0; i < num_ratio; i++) {
            float ar = ratios[i];

            int r_w = round(base_size / sqrt(ar));
            int r_h = round(r_w * ar); //round(base_size * sqrt(ar));

            for (int j = 0; j < num_scale; j++) {
                float scale = scales[j];

                float rs_w = r_w * scale;
                float rs_h = r_h * scale;

                float *anchor = anchors.row(i * num_scale + j);

                anchor[0] = cx - rs_w * 0.5f;
                anchor[1] = cy - rs_h * 0.5f;
                anchor[2] = cx + rs_w * 0.5f;
                anchor[3] = cy + rs_h * 0.5f;
            }
        }

        return anchors;
    }

    static void generate_proposals(const ncnn::Mat &anchors, int feat_stride, const ncnn::Mat &score_blob,
                                   const ncnn::Mat &bbox_blob, const ncnn::Mat &kps_blob, float prob_threshold,
                                   std::vector<FaceInfo> &faces) {
        int w = score_blob.w;
        int h = score_blob.h;

        // generate face proposal from bbox deltas and shifted anchors
        const int num_anchors = anchors.h;

        for (int q = 0; q < num_anchors; q++) {
            const float *anchor = anchors.row(q);

            const ncnn::Mat score = score_blob.channel(q);
            const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);

            // shifted anchor
            float anchor_y = anchor[1];

            float anchor_w = anchor[2] - anchor[0];
            float anchor_h = anchor[3] - anchor[1];

            for (int i = 0; i < h; i++) {
                float anchor_x = anchor[0];

                for (int j = 0; j < w; j++) {
                    int index = i * w + j;

                    float prob = score[index];

                    if (prob >= prob_threshold) {
                        // insightface/detection/scrfd/mmdet/models/dense_heads/scrfd_head.py _get_bboxes_single()
                        float dx = bbox.channel(0)[index] * feat_stride;
                        float dy = bbox.channel(1)[index] * feat_stride;
                        float dw = bbox.channel(2)[index] * feat_stride;
                        float dh = bbox.channel(3)[index] * feat_stride;

                        // insightface/detection/scrfd/mmdet/core/bbox/transforms.py distance2bbox()
                        float cx = anchor_x + anchor_w * 0.5f;
                        float cy = anchor_y + anchor_h * 0.5f;

                        int x0 = static_cast<int>(cx - dx);
                        int y0 = static_cast<int>(cy - dy);
                        int x1 = static_cast<int>(cx + dw);
                        int y1 = static_cast<int>(cy + dh);

                        FaceInfo obj;
                        obj.location_.x = x0;
                        obj.location_.y = y0;
                        obj.location_.width = x1 - x0 + 1;
                        obj.location_.height = y1 - y0 + 1;
                        obj.score_ = prob;

                        if (!kps_blob.empty()) {
                            const ncnn::Mat kps = kps_blob.channel_range(q * 10, 10);

                            obj.keypoints_[0].x = cx + kps.channel(0)[index] * feat_stride;
                            obj.keypoints_[0].y = cy + kps.channel(1)[index] * feat_stride;
                            obj.keypoints_[1].x = cx + kps.channel(2)[index] * feat_stride;
                            obj.keypoints_[1].y = cy + kps.channel(3)[index] * feat_stride;
                            obj.keypoints_[2].x = cx + kps.channel(4)[index] * feat_stride;
                            obj.keypoints_[2].y = cy + kps.channel(5)[index] * feat_stride;
                            obj.keypoints_[3].x = cx + kps.channel(6)[index] * feat_stride;
                            obj.keypoints_[3].y = cy + kps.channel(7)[index] * feat_stride;
                            obj.keypoints_[4].x = cx + kps.channel(8)[index] * feat_stride;
                            obj.keypoints_[4].y = cy + kps.channel(9)[index] * feat_stride;
                        }

                        faces.push_back(obj);
                    }

                    anchor_x += feat_stride;
                }

                anchor_y += feat_stride;
            }
        }
    }


    Scrfd::Scrfd(FaceDetectorType type) : Detector(type) {
        iouThreshold_ = 0.45f;
        scoreThreshold_ = 0.5f;
    }

    int Scrfd::loadModel(const char *root_path) {
        std::string sub_dir = "/detectors/scrfd";
        std::string fd_param = std::string(root_path) + sub_dir + "/scrfd_500m_kps-opt2.param";
        std::string fd_bin = std::string(root_path) + sub_dir + "/scrfd_500m_kps-opt2.bin";
        return Super::loadModel(fd_param.c_str(), fd_bin.c_str());
    }

#if defined __ANDROID__
    int Scrfd::loadModel(AAssetManager *mgr) {
        std::string sub_dir = "models/detectors/scrfd";
        std::string fd_param = sub_dir + "/scrfd_500m_kps-opt2.param";
        std::string fd_bin = sub_dir + "/scrfd_500m_kps-opt2.bin";
        return Super::loadModel(mgr, fd_param.c_str(), fd_bin.c_str());
    }
#endif

    int Scrfd::detectFace(const cv::Mat &img_src, std::vector<FaceInfo> &faces) const {
        cv::Mat img_cpy = img_src.clone();
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

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_cpy.data,
                                                     ncnn::Mat::PIXEL_BGR2RGB,
                                                     img_width, img_height,
                                                     w, h);

        // pad to target_size rectangle
        int wpad = (w + 31) / 32 * 32 - w;
        int hpad = (h + 31) / 32 * 32 - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2,
                               wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT,
                               0.f);

        in_pad.substract_mean_normalize(mean_vals_, norm_vals_);

        ncnn::Extractor ex = net_->create_extractor();
        ex.input("input.1", in_pad);

        faces.clear();
        // stride 8
        {
            ncnn::Mat score_blob, bbox_blob, kps_blob;
            ex.extract("score_8", score_blob);
            ex.extract("bbox_8", bbox_blob);
            if (has_kps_)
                ex.extract("kps_8", kps_blob);

            const int base_size = 16;
            const int feat_stride = 8;
            ncnn::Mat ratios(1);
            ratios[0] = 1.f;
            ncnn::Mat scales(2);
            scales[0] = 1.f;
            scales[1] = 2.f;
            ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

            std::vector<FaceInfo> faceobjects32;
            generate_proposals(anchors, feat_stride, score_blob, bbox_blob,
                               kps_blob, scoreThreshold_, faceobjects32);

            faces.insert(faces.end(), faceobjects32.begin(), faceobjects32.end());
        }

        // stride 16
        {
            ncnn::Mat score_blob, bbox_blob, kps_blob;
            ex.extract("score_16", score_blob);
            ex.extract("bbox_16", bbox_blob);
            if (has_kps_)
                ex.extract("kps_16", kps_blob);

            const int base_size = 64;
            const int feat_stride = 16;
            ncnn::Mat ratios(1);
            ratios[0] = 1.f;
            ncnn::Mat scales(2);
            scales[0] = 1.f;
            scales[1] = 2.f;
            ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

            std::vector<FaceInfo> faceobjects16;
            generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob,
                               scoreThreshold_, faceobjects16);

            faces.insert(faces.end(), faceobjects16.begin(), faceobjects16.end());
        }

        // stride 32
        {
            ncnn::Mat score_blob, bbox_blob, kps_blob;
            ex.extract("score_32", score_blob);
            ex.extract("bbox_32", bbox_blob);
            if (has_kps_)
                ex.extract("kps_32", kps_blob);

            const int base_size = 256;
            const int feat_stride = 32;
            ncnn::Mat ratios(1);
            ratios[0] = 1.f;
            ncnn::Mat scales(2);
            scales[0] = 1.f;
            scales[1] = 2.f;
            ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

            std::vector<FaceInfo> faceobjects8;
            generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob,
                               scoreThreshold_, faceobjects8);

            faces.insert(faces.end(), faceobjects8.begin(), faceobjects8.end());
        }

        for (int i = 0; i < faces.size(); i++) {
            // adjust offset to original unpadded
            float x0 = (faces[i].location_.x - (wpad / 2)) / scale;
            float y0 = (faces[i].location_.y - (hpad / 2)) / scale;
            float x1 = (faces[i].location_.x + faces[i].location_.width - (wpad / 2)) / scale;
            float y1 = (faces[i].location_.y + faces[i].location_.height - (hpad / 2)) / scale;

            x0 = std::max(std::min(x0, (float) img_width - 1), 0.f);
            y0 = std::max(std::min(y0, (float) img_height - 1), 0.f);
            x1 = std::max(std::min(x1, (float) img_width - 1), 0.f);
            y1 = std::max(std::min(y1, (float) img_height - 1), 0.f);

            faces[i].location_.x = static_cast<int>(x0);
            faces[i].location_.y = static_cast<int>(y0);
            faces[i].location_.width = static_cast<int>(x1 - x0);
            faces[i].location_.height = static_cast<int>(y1 - y0);

            if (has_kps_) {
                for (int k = 0; k < 5; ++k) {
                    float x = (faces[i].keypoints_[k].x - (wpad / 2)) / scale;
                    float y = (faces[i].keypoints_[k].y - (hpad / 2)) / scale;
                    faces[i].keypoints_[k].x = std::max(std::min(x, (float) img_width - 1), 0.f);
                    faces[i].keypoints_[k].y = std::max(std::min(y, (float) img_height - 1), 0.f);
                }
            }
        }
        return 0;
    }

}
