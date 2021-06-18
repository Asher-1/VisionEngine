#define OBJECT_EXPORTS

#include "VisionTools.h"
#include "ObjectEngine.h"
#include <iostream>
#include <opencv2/highgui.hpp>

using namespace mirror;

static const bool use_gpu = false;

int main(int argc, char *argv[]) {
    const char *img_path = "../../data/images/cat.jpg";
    cv::Mat img_src = cv::imread(img_path);

    const char *model_root_path = "../../data/models";
    mirror::ObjectEngine *object_engine = ObjectEngine::GetInstancePtr();

    ObjectEigenParams params;
    params.gpuEnabled = use_gpu;
//	params.objectDetectorType = ObjectDetectorType::YOLOV5;
    object_engine->loadModel(model_root_path, params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<mirror::ObjectInfo> objects;
    object_engine->detectObject(img_src, objects);

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << " ms." << std::endl;

    utility::DrawObjects(img_src, objects);

    cv::imwrite("../../data/images/object_result.jpg", img_src);

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::imshow("result", img_src);
    cv::waitKey(0);
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    object_engine->destroyEngine();
    return 0;
}