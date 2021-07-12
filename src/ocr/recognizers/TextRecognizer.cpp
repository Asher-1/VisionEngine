#include "TextRecognizer.h"
#include "crnn/CRNNNet.h"

#include <ncnn/net.h>
#include <ncnn/cpu.h>
#include <opencv2/imgproc.hpp>

#include <iostream>

namespace mirror {
    TextRecognizer::TextRecognizer(TextRecognizerType type) :
            type_(type),
            net_(new ncnn::Net()),
            verbose_(false),
            gpu_mode_(false),
            initialized_(false),
            threadNum_(4),
            inputSize_(cv::Size(224, 224)),
            modelPath_("/ocr") {
        class_names_.clear();
    }

    TextRecognizer::~TextRecognizer() {
        if (net_) {
            net_->clear();
            delete net_;
            net_ = nullptr;
        }
    }

    int TextRecognizer::loadModel(const char *params, const char *models) {
        if (net_->load_param(params) == -1 ||
            net_->load_model(models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }

    int TextRecognizer::loadLabels(const char *label_path) {
        FILE *fp = fopen(label_path, "r");
        if (!fp) {
            return ErrorCode::NULL_ERROR;
        }

        class_names_.clear();
        while (!feof(fp)) {
            char str[1024];
            if (nullptr == fgets(str, 1024, fp)) continue;
            std::string str_s(str);

            if (str_s.length() > 0) {
                for (int i = 0; i < str_s.length(); i++) {
                    if (str_s[i] == '\n') {
                        std::string strr = str_s.substr(0, i);
                        class_names_.push_back(strr);
                        i = str_s.length();
                    }
                }
            }
        }
        class_names_.emplace_back(" ");
        class_names_.emplace_back("·");
        return 0;
    }


#if defined __ANDROID__
    int TextRecognizer::loadModel(AAssetManager* mgr, const char* params, const char* models)
    {
        if (net_->load_param(mgr, params) == -1 ||
            net_->load_model(mgr, models) == -1) {
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        return 0;
    }

    int TextRecognizer::loadLabels(AAssetManager *mgr, const char *label_path) {
        AAsset *asset = AAssetManager_open(mgr, label_path, AASSET_MODE_BUFFER);
        if (!asset) {
            std::cout << "open label file :" << label_path << " failed!" << std::endl;
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        int len = AAsset_getLength(asset);

        std::string words_buffer;
        words_buffer.resize(len);
        int ret = AAsset_read(asset, (void *) words_buffer.data(), len);
        AAsset_close(asset);
        if (ret != len) {
            std::cout << "read label file :" << label_path << " failed!" << std::endl;
            return ErrorCode::MODEL_LOAD_ERROR;
        }

        class_names_.clear();
        SplitString(words_buffer, "\n", 0, class_names_);
        class_names_.emplace_back(" ");
        class_names_.emplace_back("·");
        return 0;
    }
#endif

    int TextRecognizer::load(const OcrEngineParams &params) {
        if (!net_) return ErrorCode::NULL_ERROR;
        verbose_ = params.verbose;

        if (verbose_) {
            std::cout << "start load classifiers model: "
                      << GetTextRecognizerTypeName(this->type_) << std::endl;
        }

        this->net_->clear();

        ncnn::Option opt;

#if defined __ANDROID__
        opt.lightmode = true;
        ncnn::set_cpu_powersave(CUSTOM_THREAD_NUMBER);
#endif
        threadNum_ = ncnn::get_big_cpu_count();
        if (params.threadNum > 0 && params.threadNum < threadNum_) {
            threadNum_ = params.threadNum;
        }
        ncnn::set_omp_num_threads(threadNum_);
        opt.num_threads = threadNum_;

#if NCNN_VULKAN
        this->gpu_mode_ = params.gpuEnabled && ncnn::get_gpu_count() > 0;
        opt.use_vulkan_compute = this->gpu_mode_;
#endif // NCNN_VULKAN

        this->net_->opt = opt;

#if defined __ANDROID__
        int flag = this->loadModel(params.mgr);
#else
        int flag = this->loadModel(params.modelPath.c_str());
#endif

        if (flag != 0) {
            initialized_ = false;
            std::cout << "load classifiers model: " <<
                      GetTextRecognizerTypeName(this->type_) << " failed!" << std::endl;
        } else {
            initialized_ = true;
            if (verbose_) {
                std::cout << "end load classifiers model." << std::endl;
            }
        }
        return flag;
    }

    int TextRecognizer::update(const OcrEngineParams &params) {
        verbose_ = params.verbose;
        int flag = 0;
        if (this->gpu_mode_ != params.gpuEnabled) {
            flag = load(params);
        }

        if (params.threadNum > 0) {
            threadNum_ = params.threadNum;
        }

        return flag;
    }

    int TextRecognizer::recognize(const cv::Mat &img_src,
                                  const std::vector<TextBox> &textBoxes,
                                  std::vector<OCRResult> &ocrResults) const {
        ocrResults.clear();
        if (!initialized_) {
            std::cout << "text recognizer model: "
                      << GetTextRecognizerTypeName(this->type_)
                      << " uninitialized!" << std::endl;
            return ErrorCode::UNINITIALIZED_ERROR;
        }

        if (img_src.empty()) {
            std::cout << "input empty." << std::endl;
            return ErrorCode::EMPTY_INPUT_ERROR;
        }

        if (verbose_) {
            std::cout << "start object classify." << std::endl;
        }

        int flag = this->recognizeText(img_src, textBoxes, ocrResults);
        if (flag != 0) {
            std::cout << "object classify failed." << std::endl;
        } else {
            if (verbose_) {
                std::cout << "end object classify." << std::endl;
            }
        }
        return flag;
    }

    TextRecognizer *CRNNNetFactory::create() const {
        return new CRNNNet();
    }
}