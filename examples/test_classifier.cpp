#define CLASSIFIER_EXPORTS

#include "VisionTools.h"
#include "ClassifierEngine.h"

#include <iostream>
#include <opencv2/opencv.hpp>

static const bool use_gpu = false;

int main(int argc, char *argv[]) {
    const char *img_path = "../../data/images/dog.jpg";
    cv::Mat img_src = cv::imread(img_path);

    const char *root_path = "../../data/models";
    mirror::ClassifierEngine *classifier_engine = mirror::ClassifierEngine::GetInstancePtr();

    mirror::ClassifierEigenParams params;
    params.gpuEnabled = use_gpu;
    params.topK = 3;
    classifier_engine->loadModel(root_path, params);
    double start = static_cast<double>(cv::getTickCount());
    std::vector<mirror::ImageInfo> images;
    classifier_engine->classify(img_src, images);

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << " ms." << std::endl;

    utility::DrawClassifications(img_src, images);
    cv::imwrite("../../data/images/classify_result.jpg", img_src);

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::imshow("result", img_src);
    cv::waitKey(0);
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    classifier_engine->destroyEngine();

    return 0;
}

