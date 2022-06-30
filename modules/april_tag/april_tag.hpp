#ifndef ARPIL_TAG_H
#define ARPIL_TAG_H

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "apriltag/common/getopt.h"
}

#include <opencv2/opencv.hpp>

class AprilTagDetector
{
public:
    AprilTagDetector();
    ~AprilTagDetector();

    zarray_t* detectTag(cv::Mat _gray_img); 

private:
    apriltag_detector_t *td;
    apriltag_family_t   *tf;
};

#endif // ARPIL_TAG_H