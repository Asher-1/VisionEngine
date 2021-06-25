#define FACE_EXPORTS

#include "FaceEngine.h"
#include "VisionTools.h"

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace mirror;

static const bool use_gpu = false;
static const bool use_living = false;
static const char *model_path = "../../data/models";


int TestDetector(int argc, char **argv) {
    std::cout << "Face Detection Test......" << std::endl;

    const char *img_file = "../../data/images/4.jpg";
    cv::Mat img_src = cv::imread(img_file);

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
//    params.faceDetectorType = FaceDetectorType::SCRFD_FACE;
    face_engine->loadModel(params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

    utility::DrawFaces(img_src, faces, true);
    cv::imwrite("../../data/images/result.jpg", img_src);

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::imshow("result", img_src);
    cv::waitKey(0);
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    face_engine->destroyEngine();

    return 0;
}

int TestLandmark(int argc, char *argv[]) {
    std::cout << "Face LandMark Test......" << std::endl;
    const char *img_file = "../../data/images/4.jpg";
    cv::Mat img_src = cv::imread(img_file);

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
    params.faceLandMarkerEnabled = true;
    face_engine->loadModel(params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);
    for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
        cv::Rect face = faces.at(i).location_;
        std::vector<cv::Point2f> keypoints;
        face_engine->extractKeypoints(img_src, face, keypoints);
        utility::DrawKeyPoints(img_src, keypoints, 1, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
    }

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

    cv::imwrite("../../images/result.jpg", img_src);

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::imshow("result", img_src);
    cv::waitKey(0);
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    face_engine->destroyEngine();

    return 0;

}

int TestRecognize(int argc, char *argv[]) {
    std::cout << "Face Recognition Test......" << std::endl;
    const char *img_file = "../../data/images/4.jpg";
    cv::Mat img_src = cv::imread(img_file);

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
    face_engine->loadModel(params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);

//    cv::Mat face1 = img_src(faces[0].location_).clone();
//    cv::Mat face2 = img_src(faces[1].location_).clone();
    cv::Mat face1;
    cv::Mat face2;
    std::vector<cv::Point2f> keyPoints;
    ConvertKeyPoints(faces[0].keypoints_, 5, keyPoints);
    face_engine->alignFace(img_src, keyPoints, face1);
    ConvertKeyPoints(faces[1].keypoints_, 5, keyPoints);
    face_engine->alignFace(img_src, keyPoints, face2);

    std::vector<float> feature1, feature2;
    face_engine->extractFeature(face1, feature1);
    face_engine->extractFeature(face2, feature2);
    float sim = CalculateSimilarity(feature1, feature2);

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

    for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
        cv::Rect face = faces.at(i).location_;
        cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("../../data/images/face1.jpg", face1);
    cv::imwrite("../../data/images/face2.jpg", face2);
    cv::imwrite("result.jpg", img_src);
    std::cout << "similarity is: " << sim << std::endl;

    face_engine->destroyEngine();
    return 0;

}

int TestAlignFace(int argc, char *argv[]) {
    std::cout << "Face Alignment Test......" << std::endl;
    const char *img_file = "../../data/images/4.jpg";
    cv::Mat img_src = cv::imread(img_file);

    const bool use_landmark = false;

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
    params.faceLandMarkerEnabled = use_landmark;
    face_engine->loadModel(params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);
    for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
        cv::Rect face = faces.at(i).location_;
        std::vector<cv::Point2f> keypoints;
        if (params.faceLandMarkerEnabled) {
            face_engine->extractKeypoints(img_src, face, keypoints);
        } else {
            ConvertKeyPoints(faces.at(i).keypoints_, 5, keypoints);
        }
        cv::Mat face_aligned;
        face_engine->alignFace(img_src, keypoints, face_aligned);
        std::string name = std::to_string(i) + ".jpg";
        cv::imwrite(name.c_str(), face_aligned);
        utility::DrawKeyPoints(img_src, keypoints, 1, cv::Scalar(0, 0, 255), 1);
        cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
    }

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::imshow("result", img_src);
    cv::waitKey(0);
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    face_engine->destroyEngine();

    return 0;
}

int TestDatabase(int argc, char *argv[]) {
    std::cout << "Face Database Test......" << std::endl;
    const char *img_path = "../../data/images/asher.jpg";
    cv::Mat img_src = cv::imread(img_path);
    if (img_src.empty()) {
        std::cout << "load image failed." << std::endl;
        return 10001;
    }

    bool show_aligned = false;

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
//    params.faceDetectorType = FaceDetectorType::SCRFD_FACE;
    face_engine->loadModel(params);
    face_engine->Load();
    face_engine->Clear();
    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);

    int faces_num = static_cast<int>(faces.size());
    std::cout << "faces number: " << faces_num << std::endl;
    for (int i = 0; i < faces_num; ++i) {
        cv::Rect face = faces.at(i).location_;

        if (show_aligned) {
#if MIRROR_BUILD_WITH_FULL_OPENCV
            cv::imshow("aligned before", img_src(face));
#endif
        }

        // align face
        cv::Mat faceAligned;
        std::vector<cv::Point2f> keyPoints;
        ConvertKeyPoints(faces.at(i).keypoints_, 5, keyPoints);
        face_engine->alignFace(img_src, keyPoints, faceAligned);

        if (show_aligned) {
#if MIRROR_BUILD_WITH_FULL_OPENCV
            cv::imshow("aligned after", faceAligned);
#endif
        }

        // get face feature
        std::vector<float> feat;
        face_engine->extractFeature(faceAligned, feat);

        cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);

#if 1
        face_engine->Insert(feat, "face" + std::to_string(i));
#endif

#if 1
        QueryResult query_result;
        face_engine->QueryTop(feat, query_result);
        std::cout << i << "-th face is: " << query_result.name_ <<
                  " similarity is: " << query_result.sim_ << std::endl;
#endif

    }
    face_engine->Save();
    cv::imwrite("../../data/images/registered.jpg", img_src);

    face_engine->destroyEngine();

    return 0;
}

int TestMask(int argc, char *argv[]) {
    std::cout << "Face Mask Test......" << std::endl;
    const char *img_file = "../../data/images/mask3.jpg";
    cv::Mat img_src = cv::imread(img_file);

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
    params.faceDetectorType = FaceDetectorType::ANTICOV_FACE;
    face_engine->loadModel(params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

    int num_face = static_cast<int>(faces.size());
    for (int i = 0; i < num_face; ++i) {
        char text[256];
        if (faces[i].mask_) {
            sprintf(text, "%s %.1f%%", "mask", faces[i].score_ * 100);
            cv::rectangle(img_src, faces[i].location_, cv::Scalar(0, 255, 0), 2);
        } else {
            sprintf(text, "%s %.1f%%", "no mask", faces[i].score_ * 100);
            cv::rectangle(img_src, faces[i].location_, cv::Scalar(0, 0, 255), 2);
        }
        utility::DrawText(img_src, faces[i].location_.tl(), text);
    }
    cv::imwrite("../../data/images/mask_result.jpg", img_src);

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::imshow("result", img_src);
    cv::waitKey(0);
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    face_engine->destroyEngine();
    return 0;
}

int TestTrack(int argc, char *argv[]) {
    std::cout << "Face Track Test......" << std::endl;

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::VideoCapture cam(0);
    if (!cam.isOpened()) {
        std::cout << "open camera failed." << std::endl;
        return -1;
    }

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.modelPath = model_path;
    params.gpuEnabled = use_gpu;
    params.livingThreshold = 0.915;
//    params.faceDetectorType = FaceDetectorType::SCRFD_FACE;
    params.faceAntiSpoofingEnabled = use_living;
    params.faceRecognizerEnabled = true;
    face_engine->loadModel(params);
    face_engine->Load();

    cv::Mat frame;
    while (true) {
        cam >> frame;
        if (frame.empty()) {
            continue;
        }

        double start = static_cast<double>(cv::getTickCount());

        // detect faces
        std::vector<FaceInfo> curr_faces;
        face_engine->detectFace(frame, curr_faces);

        // track faces
        std::vector<TrackedFaceInfo> faces;
        face_engine->track(curr_faces, faces);

        // anti face detect
        for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
            TrackedFaceInfo tracked_face_info = faces.at(i);
            char text[64];
            cv::Rect roi = tracked_face_info.face_info_.location_;
            cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);

            sprintf(text, "%s", "tracking");
            int next_y = utility::DrawText(frame, roi.tl(), text);

            bool is_living = true;
            if (use_living) {
                float livingScore;
                is_living = face_engine->detectLivingFace(frame, roi, livingScore);
                cv::rectangle(frame, tracked_face_info.face_info_.location_,
                              cv::Scalar(0, 255, 0), 2);
                cv::Scalar color;
                if (is_living) {
                    color = cv::Scalar(0, 255, 0);
                    sprintf(text, "%s %.1f%%", "living", livingScore * 100);
                } else {
                    color = cv::Scalar(0, 0, 255);
                    sprintf(text, "%s %.1f%%", "anti living", livingScore * 100);
                }
                cv::rectangle(frame, roi, color, 2);
                next_y = utility::DrawText(frame, cv::Point2i(roi.tl().x, next_y), text);
            }

            // face recognition
            if (is_living) {

                // align face
                cv::Mat faceAligned;
                std::vector<cv::Point2f> keyPoints;
                ConvertKeyPoints(tracked_face_info.face_info_.keypoints_, 5, keyPoints);
                face_engine->alignFace(frame, keyPoints, faceAligned);

                // extract features
                std::vector<float> feat;
                face_engine->extractFeature(faceAligned, feat);

                // compare features
                QueryResult query_result;
                if (face_engine->QueryTop(feat, query_result) == 0) {
                    if (query_result.sim_ > 0.5) {
                        sprintf(text, "%s %.1f%%", query_result.name_.c_str(),
                                query_result.sim_ * 100);
                    } else {
                        sprintf(text, "%s %.1f%%", "stranger",
                                query_result.sim_ * 100);
                    }
                    utility::DrawText(frame, cv::Point2i(roi.tl().x, next_y), text);
                }
            }
        }

        double end = static_cast<double>(cv::getTickCount());
        double time_cost = (end - start) / cv::getTickFrequency();
        char text[32];
        sprintf(text, "FPS=%.2f", 1 / time_cost);

        int thickness = 1;
        float fontScale = 0.5;
        int orientation = 1; // top right
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

    face_engine->destroyEngine();
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    return 0;
}

int main(int argc, char *argv[]) {
    TestDetector(argc, argv);
    TestLandmark(argc, argv);
    TestRecognize(argc, argv);
    TestAlignFace(argc, argv);
    TestDatabase(argc, argv);
    TestMask(argc, argv);
    TestTrack(argc, argv);
    return 0;
}
