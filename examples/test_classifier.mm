
#ifdef __APPLE__
#define CLASSIFIER_EXPORTS

#include "VisionTools.h"
#include "ClassifierEngine.h"

#include <iostream>
#include <opencv2/opencv.hpp>

static const bool use_gpu = true;
const char *root_path = "../../data/models";

int TestImages(int argc, char *argv[]) {
    std::cout << "Image Classification Test......" << std::endl;
    const char *img_path = "../../data/images/dog.jpg";
    cv::Mat img_src = cv::imread(img_path);

    mirror::ClassifierEngine *classifier_engine = mirror::ClassifierEngine::GetInstancePtr();

    mirror::ClassifierEngineParams params;
    params.modelPath = root_path;
    params.gpuEnabled = use_gpu;
    params.topK = 3;
    params.classifierType = mirror::ClassifierType::SQUEEZE_NET;
    classifier_engine->loadModel(params);
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

int TestVideos(int argc, char *argv[]) {
    std::cout << "Video Classification Test......" << std::endl;
    int thickness = 1;
    float fontScale = 0.5;
    int orientation = 1; // top right

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::VideoCapture cam(0);
    if (!cam.isOpened()) {
        std::cout << "open camera failed." << std::endl;
        return -1;
    }

    mirror::ClassifierEngine *classifier_engine = mirror::ClassifierEngine::GetInstancePtr();
    mirror::ClassifierEngineParams params;
    params.modelPath = root_path;
    params.gpuEnabled = use_gpu;
    params.topK = 3;
    params.classifierType = mirror::ClassifierType::SQUEEZE_NET;
    classifier_engine->loadModel(params);

    cv::Mat frame;
    while (true) {
        cam >> frame;
        if (frame.empty()) {
            continue;
        }

        double start = static_cast<double>(cv::getTickCount());

        // detect objects
        std::vector<mirror::ImageInfo> images;
        classifier_engine->classify(frame, images);
        utility::DrawClassifications(frame, images);

        double end = static_cast<double>(cv::getTickCount());
        double time_cost = (end - start) / cv::getTickFrequency();
        char text[32];
        sprintf(text, "FPS=%.2f", 1 / time_cost);

        cv::Point2i position;
        utility::GetTextCornerPosition(frame, text, orientation, fontScale, thickness, position);
        utility::DrawText(frame, position, text, fontScale, thickness);

        cv::imshow("result", frame);

        // If press space bar(32), then pause video, until any key is pressed, then restart
        if (cv::waitKey(60) == ' ') {
            cv::waitKey(-1);
        }
        if (cv::waitKey(60) == 'q' || cv::waitKey(60) == 27) {
            // press q or esc to quit,  113 is q, 27 is esc
            cv::destroyAllWindows();
            break;
        }
    }

    classifier_engine->destroyEngine();
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    return 0;
}

int main(int argc, char *argv[]) {
    TestImages(argc, argv);
    TestVideos(argc, argv);
}

#endif // __APPLE__
