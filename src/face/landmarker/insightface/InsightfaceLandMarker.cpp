#include "InsightfaceLandMarker.h"
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <ncnn/net.h>

namespace mirror {
    InsightfaceLandMarker::InsightfaceLandMarker(FaceLandMarkerType type) : LandMarker(type) {
        inputSize_.width = 192;
        inputSize_.height = 192;
    }

    int InsightfaceLandMarker::loadModel(const char *root_path) {
        std::string sub_dir = "/landmarkers/insightface";
        std::string fl_param = std::string(root_path) + sub_dir + "/2d106.param";
        std::string fl_bin = std::string(root_path) + sub_dir + "/2d106.bin";
        if (net_->load_param(fl_param.c_str()) == -1 ||
            net_->load_model(fl_bin.c_str()) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }
        return 0;
    }

    int InsightfaceLandMarker::extractKeypoints(const cv::Mat &img_src,
                                                const cv::Rect &face, std::vector<cv::Point2f> &keypoints) const {
        // 1 enlarge the face rect
        cv::Rect face_enlarged = face;
        EnlargeRect(enlarge_scale, face_enlarged);

        // 2 square the rect
        RectifyRect(face_enlarged);
        face_enlarged = face_enlarged & cv::Rect(0, 0, img_src.cols, img_src.rows);

        // 3 crop the face
        cv::Mat img_face = img_src(face_enlarged).clone();

        // 4 do inference
        ncnn::Extractor ex = net_->create_extractor();
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_face.data,
                                                     ncnn::Mat::PIXEL_BGR2RGB, img_face.cols,
                                                     img_face.rows, inputSize_.width, inputSize_.height);
        ex.input("data", in);
        ncnn::Mat out;
        ex.extract("fc1", out);

        for (int i = 0; i < 106; ++i) {
            float x = (out[2 * i] + 1.0f) * img_face.cols / 2 + face_enlarged.x;
            float y = (out[2 * i + 1] + 1.0f) * img_face.rows / 2 + face_enlarged.y;
            keypoints.emplace_back(x, y);
        }

        return 0;
    }

}
