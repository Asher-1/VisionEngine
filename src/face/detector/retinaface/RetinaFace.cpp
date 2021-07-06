#include "RetinaFace.h"
#include <iostream>

#include <ncnn/net.h>

namespace mirror {

    static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat &ratios, const ncnn::Mat &scales) {
        int num_ratio = ratios.w;
        int num_scale = scales.w;

        ncnn::Mat anchors;
        anchors.create(4, num_ratio * num_scale);

        const float cx = base_size * 0.5f;
        const float cy = base_size * 0.5f;

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
                                   const ncnn::Mat &bbox_blob, const ncnn::Mat &landmark_blob, float scoreThreshold_,
                                   std::vector<FaceInfo> &faceobjects) {
        int w = score_blob.w;
        int h = score_blob.h;

        // generate face proposal from bbox deltas and shifted anchors
        const int num_anchors = anchors.h;

        for (int q = 0; q < num_anchors; q++) {
            const float *anchor = anchors.row(q);

            const ncnn::Mat score = score_blob.channel(q + num_anchors);
            const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);
            const ncnn::Mat landmark = landmark_blob.channel_range(q * 10, 10);

            // shifted anchor
            float anchor_y = anchor[1];

            float anchor_w = anchor[2] - anchor[0];
            float anchor_h = anchor[3] - anchor[1];

            for (int i = 0; i < h; i++) {
                float anchor_x = anchor[0];

                for (int j = 0; j < w; j++) {
                    int index = i * w + j;

                    float prob = score[index];

                    if (prob >= scoreThreshold_) {
                        // apply center size
                        float dx = bbox.channel(0)[index];
                        float dy = bbox.channel(1)[index];
                        float dw = bbox.channel(2)[index];
                        float dh = bbox.channel(3)[index];

                        float cx = anchor_x + anchor_w * 0.5f;
                        float cy = anchor_y + anchor_h * 0.5f;

                        float pb_cx = cx + anchor_w * dx;
                        float pb_cy = cy + anchor_h * dy;

                        float pb_w = anchor_w * exp(dw);
                        float pb_h = anchor_h * exp(dh);

                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;

                        FaceInfo obj;
                        obj.location_.x = x0;
                        obj.location_.y = y0;
                        obj.location_.width = (x1 - x0 + 1);
                        obj.location_.height = (y1 - y0 + 1);
                        obj.keypoints_[0].x = cx + (anchor_w + 1) * landmark.channel(0)[index];
                        obj.keypoints_[0].y = cy + (anchor_h + 1) * landmark.channel(1)[index];
                        obj.keypoints_[1].x = cx + (anchor_w + 1) * landmark.channel(2)[index];
                        obj.keypoints_[1].y = cy + (anchor_h + 1) * landmark.channel(3)[index];
                        obj.keypoints_[2].x = cx + (anchor_w + 1) * landmark.channel(4)[index];
                        obj.keypoints_[2].y = cy + (anchor_h + 1) * landmark.channel(5)[index];
                        obj.keypoints_[3].x = cx + (anchor_w + 1) * landmark.channel(6)[index];
                        obj.keypoints_[3].y = cy + (anchor_h + 1) * landmark.channel(7)[index];
                        obj.keypoints_[4].x = cx + (anchor_w + 1) * landmark.channel(8)[index];
                        obj.keypoints_[4].y = cy + (anchor_h + 1) * landmark.channel(9)[index];
                        obj.score_ = prob;

                        faceobjects.push_back(obj);
                    }
                    anchor_x += feat_stride;
                }
                anchor_y += feat_stride;
            }
        }
    }

    RetinaFace::RetinaFace(FaceDetectorType type) : Detector(type) {
        iouThreshold_ = 0.4f;
        scoreThreshold_ = 0.7f;
    }

    int RetinaFace::loadModel(const char *root_path) {
        std::string sub_dir = "/detectors";
        std::string fd_param = std::string(root_path) + modelPath_ + "/retinaface/mnet.25-opt.param";
        std::string fd_bin = std::string(root_path) + modelPath_ + "/retinaface/mnet.25-opt.bin";
        return Super::loadModel(fd_param.c_str(), fd_bin.c_str());
    }

#if defined __ANDROID__
    int RetinaFace::loadModel(AAssetManager *mgr) {
        std::string fd_param = "models" + modelPath_ + "/retinaface/mnet.25-opt.param";
        std::string fd_bin = "models" + modelPath_ + "/retinaface/mnet.25-opt.bin";
        return Super::loadModel(mgr, fd_param.c_str(), fd_bin.c_str());
    }
#endif

    int RetinaFace::detectFace(const cv::Mat &img_src, std::vector<FaceInfo> &faces) const {
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

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_cpy.data,
                                                     ncnn::Mat::PIXEL_BGR2RGB,
                                                     img_width,
                                                     img_height,
                                                     w,
                                                     h);

        ncnn::Extractor ex = net_->create_extractor();
#if NCNN_VULKAN
        if (this->gpu_mode_) {
            ex.set_vulkan_compute(this->gpu_mode_);
        }
#endif

        ex.input("data", in);

        faces.clear();
        // stride 32
        {
            ncnn::Mat score_blob, bbox_blob, landmark_blob;
            ex.extract("face_rpn_cls_prob_reshape_stride32", score_blob);
            ex.extract("face_rpn_bbox_pred_stride32", bbox_blob);
            ex.extract("face_rpn_landmark_pred_stride32", landmark_blob);

            const int base_size = 16;
            const int feat_stride = 32;
            ncnn::Mat ratios(1);
            ratios[0] = 1.f;
            ncnn::Mat scales(2);
            scales[0] = 32.f;
            scales[1] = 16.f;
            ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

            std::vector<FaceInfo> faceobjects32;
            generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, scoreThreshold_,
                               faceobjects32);

            faces.insert(faces.end(), faceobjects32.begin(), faceobjects32.end());
        }

        // stride 16
        {
            ncnn::Mat score_blob, bbox_blob, landmark_blob;
            ex.extract("face_rpn_cls_prob_reshape_stride16", score_blob);
            ex.extract("face_rpn_bbox_pred_stride16", bbox_blob);
            ex.extract("face_rpn_landmark_pred_stride16", landmark_blob);

            const int base_size = 16;
            const int feat_stride = 16;
            ncnn::Mat ratios(1);
            ratios[0] = 1.f;
            ncnn::Mat scales(2);
            scales[0] = 8.f;
            scales[1] = 4.f;
            ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

            std::vector<FaceInfo> faceobjects16;
            generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, scoreThreshold_,
                               faceobjects16);

            faces.insert(faces.end(), faceobjects16.begin(), faceobjects16.end());
        }

        // stride 8
        {
            ncnn::Mat score_blob, bbox_blob, landmark_blob;
            ex.extract("face_rpn_cls_prob_reshape_stride8", score_blob);
            ex.extract("face_rpn_bbox_pred_stride8", bbox_blob);
            ex.extract("face_rpn_landmark_pred_stride8", landmark_blob);

            const int base_size = 16;
            const int feat_stride = 8;
            ncnn::Mat ratios(1);
            ratios[0] = 1.f;
            ncnn::Mat scales(2);
            scales[0] = 2.f;
            scales[1] = 1.f;
            ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

            std::vector<FaceInfo> faceobjects8;
            generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, scoreThreshold_,
                               faceobjects8);

            faces.insert(faces.end(), faceobjects8.begin(), faceobjects8.end());
        }

        for (int i = 0; i < faces.size(); i++) {

            // location_
            float x0 = faces[i].location_.x * factor_x;
            float y0 = faces[i].location_.y * factor_y;
            float x1 = (faces[i].location_.x + faces[i].location_.width) * factor_x;
            float y1 = (faces[i].location_.y + faces[i].location_.height) * factor_y;

            x0 = std::max(std::min(x0, (float) img_width - 1), 0.f);
            y0 = std::max(std::min(y0, (float) img_height - 1), 0.f);
            x1 = std::max(std::min(x1, (float) img_width - 1), 0.f);
            y1 = std::max(std::min(y1, (float) img_height - 1), 0.f);

            faces[i].location_.x = static_cast<int>(x0);
            faces[i].location_.y = static_cast<int>(y0);
            faces[i].location_.width = static_cast<int>(x1 - x0);
            faces[i].location_.height = static_cast<int>(y1 - y0);

            // keypoints_
            for (int k = 0; k < 5; ++k) {
                float x = (faces[i].keypoints_[k].x) * factor_x;
                float y = (faces[i].keypoints_[k].y) * factor_y;
                faces[i].keypoints_[k].x = std::max(std::min(x, (float) img_width - 1), 0.f);
                faces[i].keypoints_[k].y = std::max(std::min(y, (float) img_height - 1), 0.f);
            }
        }
        return 0;
    }

}
