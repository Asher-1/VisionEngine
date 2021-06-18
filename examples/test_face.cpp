#define FACE_EXPORTS

#include "FaceEngine.h"
#include "VisionTools.h"

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace mirror;

static const bool use_gpu = false;

int TestLiving(int argc, char *argv[]) {
    std::cout << "Face Living Test......" << std::endl;
    const char *img_file = "../../data/images/mask3.jpg";
    cv::Mat img_src = cv::imread(img_file);
    const char *root_path = "../../data/models";

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.gpuEnabled = use_gpu;
    params.livingThreshold = 0.9;
    params.faceAntiSpoofingEnabled = true;
    params.faceRecognizerEnabled = false;
    face_engine->loadModel(root_path, params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

    int num_face = static_cast<int>(faces.size());
    for (int i = 0; i < num_face; ++i) {
        float livingScore;
        bool is_living = face_engine->detectLivingFace(img_src, faces[i].location_, livingScore);
        char text[256];
        cv::Scalar color;
        if (is_living) {
            color = cv::Scalar(0, 255, 0);
            cv::rectangle(img_src, faces[i].location_, color, 2);
            sprintf(text, "%s %.1f%%", "living", livingScore * 100);
        } else {
            color = cv::Scalar(0, 0, 255);
            cv::rectangle(img_src, faces[i].location_, color, 2);
            sprintf(text, "%s %.1f%%", "anti living", livingScore * 100);
        }
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                              0.5, 1, &baseLine);
        int x = faces[i].location_.x;
        int y = faces[i].location_.y - baseLine;
        cv::putText(img_src, text, cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

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

int TestLandmark(int argc, char *argv[]) {
    std::cout << "Face LandMark Test......" << std::endl;
    const char *img_file = "../../data/images/4.jpg";
    cv::Mat img_src = cv::imread(img_file);
    const char *root_path = "../../data/models";

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.gpuEnabled = use_gpu;
    params.faceLandMarkerEnabled = true;
    face_engine->loadModel(root_path, params);

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
    const char *root_path = "../../data/models";

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.gpuEnabled = use_gpu;
    face_engine->loadModel(root_path, params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);

    cv::Mat face1 = img_src(faces[0].location_).clone();
    cv::Mat face2 = img_src(faces[1].location_).clone();
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
    const char *root_path = "../../data/models";

    const bool use_landmark = false;

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.gpuEnabled = use_gpu;
    params.faceLandMarkerEnabled = use_landmark;
    face_engine->loadModel(root_path, params);

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

int TestDetector(int argc, char **argv) {
    std::cout << "Face Detection Test......" << std::endl;

    const char *img_file = "../../data/images/4.jpg";
    cv::Mat img_src = cv::imread(img_file);
    const char *root_path = "../../data/models";

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.gpuEnabled = use_gpu;
    face_engine->loadModel(root_path, params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

    utility::DrawFaces(img_src, faces, true);
    cv::imwrite("../../data/images/retinaface_result.jpg", img_src);

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
    const char *img_file = "../../data/images/4.jpg";

#if MIRROR_BUILD_WITH_FULL_OPENCV
    cv::Mat img_src = cv::imread(img_file);
    cv::VideoCapture cam(0);
    if (!cam.isOpened()) {
        std::cout << "open camera failed." << std::endl;
        return -1;
    }

    const char *root_path = "../../data/models";
    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.gpuEnabled = use_gpu;
    face_engine->loadModel(root_path, params);

    cv::Mat frame;
    while (true) {
        cam >> frame;
        if (frame.empty()) {
            continue;
        }
        std::vector<FaceInfo> curr_faces;
        face_engine->detectFace(frame, curr_faces);
        std::vector<TrackedFaceInfo> faces;
        face_engine->track(curr_faces, faces);

        for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
            TrackedFaceInfo tracked_face_info = faces.at(i);
            cv::rectangle(frame, tracked_face_info.face_info_.location_,
                          cv::Scalar(0, 255, 0), 2);
        }


        cv::imshow("result", frame);
        if (cv::waitKey(60) == 'q') {
            break;
        }
    }

    face_engine->destroyEngine();
#else
    std::cout << "Inorder to support visualization, please rebuild with full opencv support!" << std::endl;
#endif

    return 0;
}

int TestDatabase(int argc, char *argv[]) {
    std::cout << "Face Database Test......" << std::endl;
    const char *img_path = "../../data/images/4.jpg";
    cv::Mat img_src = cv::imread(img_path);
    if (img_src.empty()) {
        std::cout << "load image failed." << std::endl;
        return 10001;
    }

    const char *root_path = "../../data/models";
    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.gpuEnabled = use_gpu;
    face_engine->loadModel(root_path, params);
    face_engine->Load();
    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);

    int faces_num = static_cast<int>(faces.size());
    std::cout << "faces number: " << faces_num << std::endl;
    for (int i = 0; i < faces_num; ++i) {
        cv::Rect face = faces.at(i).location_;
        cv::rectangle(img_src, face, cv::Scalar(0, 255, 0), 2);
        std::vector<float> feat;
        face_engine->extractFeature(img_src(face).clone(), feat);

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
    cv::imwrite("../../data/images/result.jpg", img_src);

    face_engine->destroyEngine();

    return 0;
}

int TestMask(int argc, char *argv[]) {
    std::cout << "Face Mask Test......" << std::endl;
    const char *img_file = "../../data/images/mask3.jpg";
    cv::Mat img_src = cv::imread(img_file);
    const char *root_path = "../../data/models";

    FaceEngine *face_engine = FaceEngine::GetInstancePtr();
    FaceEigenParams params;
    params.gpuEnabled = use_gpu;
    params.faceDetectorType = FaceDetectorType::ANTICOV_FACE;
    face_engine->loadModel(root_path, params);

    double start = static_cast<double>(cv::getTickCount());

    std::vector<FaceInfo> faces;
    face_engine->detectFace(img_src, faces);

    double end = static_cast<double>(cv::getTickCount());
    double time_cost = (end - start) / cv::getTickFrequency() * 1000;
    std::cout << "time cost: " << time_cost << "ms" << std::endl;

    int num_face = static_cast<int>(faces.size());
    for (int i = 0; i < num_face; ++i) {
        if (faces[i].mask_) {
            cv::rectangle(img_src, faces[i].location_, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::rectangle(img_src, faces[i].location_, cv::Scalar(0, 0, 255), 2);
        }
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

int main(int argc, char *argv[]) {
    TestLiving(argc, argv);
    TestLandmark(argc, argv);
    TestRecognize(argc, argv);
    TestAlignFace(argc, argv);
    TestDetector(argc, argv);
    TestTrack(argc, argv);
    TestDatabase(argc, argv);
    TestMask(argc, argv);
    return 0;
}
