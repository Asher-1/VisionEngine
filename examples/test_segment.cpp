#define SEGMENT_EXPORTS

#include "VisionTools.h"
#include "SegmentEngine.h"
#include <iostream>
#include <opencv2/highgui.hpp>

using namespace mirror;

static bool use_gpu = false;
static std::string img_path = "../../data/images/playground.jpg";
static std::string model_path = "../../data/models";
static std::string result_path = "../../data/images/segment_result.jpg";
static SegmentType modelType = SegmentType::YOLACT_SEG;

int TestImages(int argc, char *argv[]) {
    if (argc >= 2) {
        result_path = "segment_result.jpg";
    }

    std::cout << "Image Pose Detection Test......" << std::endl;
    cv::Mat img_src = cv::imread(img_path);

    SegmentEngine *seg_engine = SegmentEngine::GetInstancePtr();

    SegmentEngineParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
    params.segmentType = modelType;
    seg_engine->loadModel(params);

    double start = static_cast<double>(cv::getTickCount());

    // detect mask
    std::vector<SegmentInfo> segments;
    seg_engine->detect(img_src, segments);

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << " ms." << std::endl;

    utility::DrawMask(img_src, segments, int(params.segmentType));

    cv::imwrite(result_path, img_src);

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::imshow("result", img_src);
    cv::waitKey(0);
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    seg_engine->destroyEngine();
    return 0;
}

int TestVideos(int argc, char *argv[]) {
    std::cout << "Video Pose Detection Test......" << std::endl;
    int thickness = 1;
    float fontScale = 0.5;
    int orientation = 1; // top right

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::VideoCapture cam(0);
    if (!cam.isOpened()) {
        std::cout << "open camera failed." << std::endl;
        return -1;
    }

    SegmentEngine *seg_engine = SegmentEngine::GetInstancePtr();
    SegmentEngineParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
    params.segmentType = modelType;
    seg_engine->loadModel(params);

    cv::Mat frame;
    while (true) {
        cam >> frame;
        if (frame.empty()) {
            continue;
        }

        double start = static_cast<double>(cv::getTickCount());

        // detect mask
        std::vector<SegmentInfo> segments;
        seg_engine->detect(frame, segments);
        utility::DrawMask(frame, segments, int(params.segmentType));

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

    seg_engine->destroyEngine();
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    return 0;
}

int main(int argc, char *argv[]) {
    if (argc == 2) {
        model_path = argv[1];
    } else if (argc == 3) {
        model_path = argv[1];
        img_path = argv[2];
    } else if (argc == 4) {
        model_path = argv[1];
        img_path = argv[2];
        modelType = SegmentType(std::stoi(argv[3]));
    } else if (argc >= 5) {
        model_path = argv[1];
        img_path = argv[2];
        modelType = SegmentType(std::stoi(argv[3]));
        use_gpu = std::string(argv[4]) == "1" ? true : false;
    }

    TestImages(argc, argv);
    TestVideos(argc, argv);
}