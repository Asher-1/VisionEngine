#define OCR_EXPORTS

#include "VisionTools.h"
#include "OcrEngine.h"

#include <iostream>
#include <opencv2/opencv.hpp>

static const bool use_gpu = true;
const char *root_path = "../../data/models";


void printDetectionResult(const std::vector<mirror::OCRResult>& ocrResults) {
        for (const auto &ocrRes: ocrResults) {
        std::string jointStr;
        for (const auto &str: ocrRes.predictions) {
            jointStr += str;
        }
        std::cout << jointStr << std::endl;
    }
}

int TestImages(int argc, char *argv[]) {
    std::cout << "Image OCR Test......" << std::endl;
    const char *img_path = "../../data/images/idcard.jpg";
    cv::Mat img_src = cv::imread(img_path);

    mirror::OcrEngine *ocr_engine = mirror::OcrEngine::GetInstancePtr();

    mirror::OcrEngineParams params;
    params.modelPath = root_path;
    params.gpuEnabled = use_gpu;
    ocr_engine->loadModel(params);
    double start = static_cast<double>(cv::getTickCount());
    std::vector<mirror::TextBox> textBoxes;
    ocr_engine->detectText(img_src, textBoxes);

    std::vector<mirror::OCRResult> ocrResults;
    ocr_engine->recognizeText(img_src, textBoxes, ocrResults);

    printDetectionResult(ocrResults);

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << " ms." << std::endl;

    utility::DrawOcrResults(img_src, ocrResults);
    cv::imwrite("../../data/images/ocr_result.jpg", img_src);

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::imshow("result", img_src);
    cv::waitKey(0);
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    ocr_engine->destroyEngine();

    return 0;
}

int TestVideos(int argc, char *argv[]) {
    std::cout << "Video OCR Test......" << std::endl;
    int thickness = 1;
    float fontScale = 0.5;
    int orientation = 1; // top right

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::VideoCapture cam(0);
    if (!cam.isOpened()) {
        std::cout << "open camera failed." << std::endl;
        return -1;
    }

    mirror::OcrEngine *ocr_engine = mirror::OcrEngine::GetInstancePtr();
    mirror::OcrEngineParams params;
    params.modelPath = root_path;
    params.gpuEnabled = use_gpu;
    ocr_engine->loadModel(params);

    cv::Mat frame;
    while (true) {
        cam >> frame;
        if (frame.empty()) {
            continue;
        }

        double start = static_cast<double>(cv::getTickCount());

        // detect objects
        std::vector<mirror::TextBox> textBoxes;
        ocr_engine->detectText(frame, textBoxes);

        std::vector<mirror::OCRResult> ocrResults;
        ocr_engine->recognizeText(frame, textBoxes, ocrResults);
        printDetectionResult(ocrResults);
        utility::DrawOcrResults(frame, ocrResults);

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

    ocr_engine->destroyEngine();
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    return 0;
}

int main(int argc, char *argv[]) {
    TestImages(argc, argv);
    TestVideos(argc, argv);
}
