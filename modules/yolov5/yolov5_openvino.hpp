#ifndef YOLOV5_OPENVINO_H
#define YOLOV5_OPENVINO_H

#include "yolov5/utils.hpp"

namespace yolov5
{

class DetectorOpenVINO
{
public:
    DetectorOpenVINO(const std::string& model_path_,
        const std::string& class_list_path, 
        const bool& is_GPU_ = true,
        const int& resolution_ = 480);
    ~DetectorOpenVINO();

    std::vector<yolov5::Detection> detect(const cv::Mat &src_, 
        const float & confidence_th_ = 0.4,
        const float& iou_th_ = 0.4);

    inline std::vector<std::string> classList(){
        return this->class_list;
    }
private:
    ov::Core ie;
    std::shared_ptr<ov::Model> network;
	ov::CompiledModel executable_network;

    std::vector<std::string> class_list;
    int resolution;

    void networkInfo(const ov::Model& network);
    void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
    float& bestConf, int& bestClassId);
    cv::Rect scaleBBox(const cv::Rect &bbox_, const float &scale_);

    cv::Mat format_img(const cv::Mat &src);
};

} // namespace yolov5

#endif // YOLOV5_OPENVINO_H