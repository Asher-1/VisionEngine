#define _CRT_SECURE_NO_WARNINGS

#include "CRNNNet.h"
#include <algorithm>
#include <string>
#include <ncnn/net.h>

#include <iostream>

namespace mirror {


    static std::vector<std::string> CrnnDeocde(const ncnn::Mat &score,
                                               const std::vector<std::string> &alphabetChinese) {
        float *srcdata = (float *) score.data;
        std::vector<std::string> str_res;
        int last_index = 0;
        for (int i = 0; i < score.h; i++) {
            int max_index = 0;

            float max_value = -1000;
            for (int j = 0; j < score.w; j++) {
                if (srcdata[i * score.w + j] > max_value) {
                    max_value = srcdata[i * score.w + j];
                    max_index = j;
                }
            }
            if (max_index > 0 && (not(i > 0 && max_index == last_index))) {
//            std::cout <<  max_index - 1 << std::endl;
//            std::string temp_str =  utf8_substr2(alphabetChinese,max_index - 1,1)  ;
                str_res.push_back(alphabetChinese[max_index - 1]);
            }
            last_index = max_index;
        }
        return str_res;
    }

    static void matRotateClockWise180(cv::Mat &src) { //顺时针180
        //0: 沿X轴翻转； >0: 沿Y轴翻转； <0: 沿X轴和Y轴翻转
        // 翻转模式，flipCode == 0垂直翻转（沿X轴翻转），flipCode>0水平翻转（沿Y轴翻转），
        // flipCode<0水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）
        flip(src, src, 0);
        flip(src, src, 1);
        //transpose(src, src);// 矩阵转置
    }

    static void matRotateClockWise90(cv::Mat &src) {
        // 矩阵转置
        transpose(src, src);
        //0: 沿X轴翻转； >0: 沿Y轴翻转； <0: 沿X轴和Y轴翻转
        // 翻转模式，flipCode == 0垂直翻转（沿X轴翻转），flipCode>0水平翻转（沿Y轴翻转），
        // flipCode<0水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）
        flip(src, src, 1);
    }


    static cv::Mat GetRotateCropImage(const cv::Mat &srcimage,
                                      const std::vector<cv::Point> &box) {
        cv::Mat image;
        srcimage.copyTo(image);
        std::vector<cv::Point> points = box;

        int x_collect[4] = {box[0].x, box[1].x, box[2].x, box[3].x};
        int y_collect[4] = {box[0].y, box[1].y, box[2].y, box[3].y};
        int left = int(*std::min_element(x_collect, x_collect + 4));
        int right = int(*std::max_element(x_collect, x_collect + 4));
        int top = int(*std::min_element(y_collect, y_collect + 4));
        int bottom = int(*std::max_element(y_collect, y_collect + 4));

        cv::Mat img_crop;
        left = left > 0 ? left : 0;
        top = top > 0 ? top : 0;
        right = right > image.size().width - 1 ? image.size().width - 1 : right;
        bottom = bottom > image.size().height - 1 ? image.size().height - 1 : bottom;
        image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

        for (int i = 0; i < points.size(); i++) {
            points[i].x -= left;
            points[i].y -= top;
        }

        int img_crop_width = int(sqrt(pow(points[0].x - points[1].x, 2) +
                                      pow(points[0].y - points[1].y, 2)));
        int img_crop_height = int(sqrt(pow(points[0].x - points[3].x, 2) +
                                       pow(points[0].y - points[3].y, 2)));

        cv::Point2f pts_std[4];
        pts_std[0] = cv::Point2f(0., 0.);
        pts_std[1] = cv::Point2f(img_crop_width, 0.);
        pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
        pts_std[3] = cv::Point2f(0.f, img_crop_height);

        cv::Point2f pointsf[4];
        pointsf[0] = cv::Point2f(points[0].x, points[0].y);
        pointsf[1] = cv::Point2f(points[1].x, points[1].y);
        pointsf[2] = cv::Point2f(points[2].x, points[2].y);
        pointsf[3] = cv::Point2f(points[3].x, points[3].y);

        cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

        cv::Mat dst_img;
        cv::warpPerspective(img_crop, dst_img, M,
                            cv::Size(img_crop_width, img_crop_height),
                            cv::BORDER_REPLICATE);

        if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
            cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
            cv::transpose(dst_img, srcCopy);
            cv::flip(srcCopy, srcCopy, 0);
            return srcCopy;
        } else {
            return dst_img;
        }
    }


    CRNNNet::CRNNNet(TextRecognizerType type) :
            TextRecognizer(type),
            angleNet_(new ncnn::Net()) {
        inputSize_ = cv::Size(224, 32);
    }

    CRNNNet::~CRNNNet() {
        if (angleNet_) {
            angleNet_->clear();
            delete angleNet_;
            angleNet_ = nullptr;
        }
    }

    int CRNNNet::loadModel(const char *root_path) {
        if (!angleNet_) {
            return ErrorCode::UNINITIALIZED_ERROR;
        }

        angleNet_->clear();
#if NCNN_VULKAN
        angleNet_->opt.use_vulkan_compute = this->gpu_mode_;
#endif // NCNN_VULKAN
        angleNet_->opt.num_threads = threadNum_;
        angleNet_->opt.use_fp16_arithmetic = true;  // fp16运算加速
        std::string angle_path = std::string(root_path) + modelPath_ + "/angles";
        std::string param_angle = angle_path + "/angle_op.param";
        std::string model_angle = angle_path + "/angle_op.bin";
        if (angleNet_->load_param(param_angle.c_str()) == -1 ||
            angleNet_->load_model(model_angle.c_str()) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        std::string root_dir = std::string(root_path) + modelPath_ + "/recognizers/crnn";
        std::string param_file = root_dir + "/crnn_lite_op.param";
        std::string model_file = root_dir + "/crnn_lite_op.bin";
        std::string label_file = root_dir + "/keys.txt";
        if (Super::loadModel(param_file.c_str(), model_file.c_str()) != 0 ||
            Super::loadLabels(label_file.c_str()) != 0) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }

#if defined __ANDROID__
    int CRNNNet::loadModel(AAssetManager *mgr) {
                if (!angleNet_) {
            return ErrorCode::UNINITIALIZED_ERROR;
        }

        angleNet_->clear();
#if NCNN_VULKAN
        angleNet_->opt.use_vulkan_compute = this->gpu_mode_;
#endif // NCNN_VULKAN
        angleNet_->opt.num_threads = threadNum_;
        angleNet_->opt.use_fp16_arithmetic = true;  // fp16运算加速
        std::string angle_path = "models" + modelPath_ + "/angles";
        std::string param_angle = angle_path + "/angle_op.param";
        std::string model_angle = angle_path + "/angle_op.bin";
        if (angleNet_->load_param(param_angle.c_str()) == -1 ||
            angleNet_->load_model(model_angle.c_str()) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        std::string root_dir = "models" + modelPath_ + "/crnn";
        std::string param_file = root_dir + "/crnn_lite_op.param";
        std::string model_file = root_dir + "/crnn_lite_op.bin";
        std::string label_file = root_dir + "/keys.txt";
        if (Super::loadModel(mgr, param_file.c_str(), model_file.c_str()) != 0 ||
            Super::loadLabels(mgr, label_file.c_str()) != 0) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }
#endif

    int CRNNNet::recognizeText(const cv::Mat &img_src,
                               const std::vector<TextBox> &textBoxes,
                               std::vector<OCRResult> &ocrResults) const {

        cv::Mat im_bgr = img_src.clone();

        //开始行文本角度检测和文字识别
        ocrResults.clear();
        for (int i = textBoxes.size() - 1; i >= 0; i--) {
            cv::Mat part_im;
            part_im = GetRotateCropImage(im_bgr, textBoxes[i].box);
            int part_im_w = part_im.cols;
            int part_im_h = part_im.rows;

            // 开始文本识别
            int crnn_w_target;
            float scale = crnn_h * 1.0 / part_im_h;
            crnn_w_target = int(part_im.cols * scale);

            cv::Mat img2 = part_im.clone();

            ncnn::Mat crnn_in = ncnn::Mat::from_pixels_resize(img2.data,
                                                              ncnn::Mat::PIXEL_BGR2RGB,
                                                              img2.cols, img2.rows,
                                                              crnn_w_target, crnn_h);

            //角度检测
            int crnn_w = crnn_in.w;
            ncnn::Mat angle_in;
            if (crnn_w >= angle_target_w) {
                ncnn::copy_cut_border(crnn_in, angle_in, 0, 0, 0,
                                      crnn_w - angle_target_w);
            } else {
                ncnn::copy_make_border(crnn_in, angle_in, 0, 0, 0,
                                       angle_target_w - crnn_w, 0, 255.f);
            }

            angle_in.substract_mean_normalize(angleMeanVals, angleNormVals);

            ncnn::Extractor angle_ex = angleNet_->create_extractor();
            angle_ex.set_light_mode(true);
            angle_ex.set_num_threads(threadNum_);
#if NCNN_VULKAN
            if (this->gpu_mode_) {
                angle_ex.set_vulkan_compute(this->gpu_mode_);
            }
#endif
            angle_ex.input("input", angle_in);
            ncnn::Mat angle_preds;
            angle_ex.extract("out", angle_preds);
            auto *srcdata = (float *) angle_preds.data;

            float angle_score = srcdata[0];
            if (verbose_) {
                std::cout << "jni ocr angle score: " << angle_score << std::endl;
            }

            //判断方向
            if (angle_score < 0.5) {
                matRotateClockWise180(part_im);
            }

            //crnn识别
            crnn_in.substract_mean_normalize(meanVals, normVals);
            ncnn::Mat crnn_preds;
            ncnn::Extractor crnn_ex = net_->create_extractor();
            crnn_ex.set_light_mode(true);
            crnn_ex.set_num_threads(threadNum_);
#if NCNN_VULKAN
            if (this->gpu_mode_) {
                crnn_ex.set_vulkan_compute(this->gpu_mode_);
            }
#endif
            crnn_ex.input("input", crnn_in);

            ncnn::Mat blob162;
            crnn_ex.extract("443", blob162);

            ncnn::Mat blob263(5532, blob162.h);
            //batch fc
            for (int j = 0; j < blob162.h; j++) {
                ncnn::Extractor crnn_ex_2 = net_->create_extractor();
                crnn_ex_2.set_light_mode(true);
                crnn_ex_2.set_num_threads(threadNum_);
#if NCNN_VULKAN
                if (this->gpu_mode_) {
                    crnn_ex_2.set_vulkan_compute(this->gpu_mode_);
                }
#endif
                ncnn::Mat blob243_i = blob162.row_range(j, 1);
                crnn_ex_2.input("457", blob243_i);

                ncnn::Mat blob263_i;
                crnn_ex_2.extract("458", blob263_i);

                memcpy(blob263.row(j), blob263_i, 5532 * sizeof(float));
            }

            crnn_preds = blob263;

            OCRResult ocrInfo;
            ocrInfo.predictions = CrnnDeocde(crnn_preds, class_names_);
            ocrInfo.boxes = textBoxes[i].box;
            ocrInfo.boxScore = textBoxes[i].score;
            ocrResults.push_back(ocrInfo);
        }

        return 0;
    }
}


