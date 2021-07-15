#define SEGMENT_EXPORTS

#include "VisionTools.h"
#include "SegmentEngine.h"
#include <iostream>
#include <opencv2/highgui.hpp>

using namespace mirror;

static const bool use_gpu = true;
const char *model_root_path = "../../data/models";

int TestImages(int argc, char *argv[]) {
    std::cout << "Image Pose Detection Test......" << std::endl;
    const char *img_path = "../../data/images/playground.jpg";
    cv::Mat img_src = cv::imread(img_path);

    SegmentEngine *seg_engine = SegmentEngine::GetInstancePtr();

    SegmentEngineParams params;
    params.modelPath = model_root_path;
    params.gpuEnabled = use_gpu;
    params.segmentType = SegmentType::YOLACT_SEG;
    //params.segmentType = SegmentType::MOBILENETV3_SEG;
    seg_engine->loadModel(params);

    double start = static_cast<double>(cv::getTickCount());

    // detect mask
    std::vector<SegmentInfo> segments;
    seg_engine->detect(img_src, segments);

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << " ms." << std::endl;

    utility::DrawMask(img_src, segments, int(params.segmentType));

    cv::imwrite("../../data/images/segment_result.jpg", img_src);

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
    params.modelPath = model_root_path;
    params.gpuEnabled = use_gpu;
    params.segmentType = SegmentType::YOLACT_SEG;
//    params.segmentType = SegmentType::MOBILENETV3_SEG;
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
    TestImages(argc, argv);
    TestVideos(argc, argv);
}