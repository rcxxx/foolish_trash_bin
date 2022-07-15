#include "april_tag/april_tag.hpp"

AprilTagDetector::AprilTagDetector(){
    // aprilTag config
    this->td = apriltag_detector_create();
    this->tf = tag36h11_create();

    apriltag_detector_add_family(td, tf);
    this->td->nthreads = 4;
}

zarray_t* AprilTagDetector::detectTag(cv::Mat _gray_img){
    image_u8_t img_header = {
        .width = _gray_img.cols,
        .height = _gray_img.rows,
        .stride = _gray_img.cols,
        .buf = _gray_img.data
    };

    return apriltag_detector_detect(this->td, &img_header);
}

AprilTagDetector::~AprilTagDetector(){

}