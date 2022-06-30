#include "april_tag/april_tag.hpp"

AprilTagDetector::AprilTagDetector(){
    // aprilTag config
    td = apriltag_detector_create();
    tf = tag36h11_create();

    apriltag_detector_add_family(td, tf);
    td->nthreads = 4;
}

zarray_t* AprilTagDetector::detectTag(cv::Mat _gray_img){
    image_u8_t img_header = {
        .width = _gray_img.cols,
        .height = _gray_img.rows,
        .stride = _gray_img.cols,
        .buf = _gray_img.data
    };

    zarray_t *detections = apriltag_detector_detect(td, &img_header);

    return detections;
}

AprilTagDetector::~AprilTagDetector(){

}