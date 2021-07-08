#pragma once

#include "../ObjectDetector.h"

namespace mirror {
    struct HeadInfo {
        std::string cls_layer;
        std::string dis_layer;
        int stride;
    };

    class NanoDet : public ObjectDetector {
    public:
        explicit NanoDet(ObjectDetectorType type = ObjectDetectorType::NANO_DET);

        ~NanoDet() override = default;

    protected:
#if defined __ANDROID__
        int loadModel(AAssetManager* mgr) override;
#endif

        int loadModel(const char *model_path) override;

        int detectObject(const cv::Mat &img_src, std::vector<ObjectInfo> &objects) const override;

    private:
        void decode_infer(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred, int stride, float threshold,
                          std::vector<std::vector<ObjectInfo>> &results, float width_ratio, float height_ratio) const;

        ObjectInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y,
                                int stride, float width_ratio, float height_ratio) const;

        static void nms(std::vector<ObjectInfo> &result, float nms_threshold);
        

    private:
        const std::vector<HeadInfo> heads_info{
                // cls_pred|dis_pred|stride
                {"792", "795", 8},
                {"814", "817", 16},
                {"836", "839", 32},
        };
        const int regMax = 7;
        const float meanVals[3] = {103.53f, 116.28f, 123.675f};
        const float normVals[3] = {0.017429f, 0.017507f, 0.01712475};
    };

}