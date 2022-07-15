#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <math.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace yolov5
{

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect bbox;
};

} // namespace yolov5


#endif //UTILS_H