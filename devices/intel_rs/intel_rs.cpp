#include "intel_rs/intel_rs.hpp"

Rs2Camera::Rs2Camera():align_to(RS2_STREAM_COLOR){
    rs2::config* config = new rs2::config();

    // Request a specific configuration
    config->enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 60);
    config->enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 60);

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    // Start streaming with default recommended configuration
    this->pipeProfile = this->pipe.start(*config);
    this->intrinsics = this->pipeProfile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
}

void Rs2Camera::updateFrame(){
    //  realsenseD435 拍摄到的帧
    rs2::frameset frames = this->pipe.wait_for_frames();
    rs2::frameset aligned_set = this->align_to.process(frames);

    this->color_frames = aligned_set.get_color_frame();
    this->depth_frames = aligned_set.get_depth_frame().apply_filter(this->color_map);
}

cv::Mat Rs2Camera::rgbImg(){
    const int w = this->color_frames.as<rs2::video_frame>().get_width();
    const int h = this->color_frames.as<rs2::video_frame>().get_height();

    return cv::Mat(cv::Size(w, h), CV_8UC3, const_cast<void*>(this->color_frames.get_data()), cv::Mat::AUTO_STEP);
}

cv::Mat Rs2Camera::depthImg(){
    const int w = this->color_frames.as<rs2::video_frame>().get_width();
    const int h = this->color_frames.as<rs2::video_frame>().get_height();

    return cv::Mat(cv::Size(w, h), CV_8UC3, const_cast<void*>(this->depth_frames.get_data()), cv::Mat::AUTO_STEP);
}

Rs2Camera::~Rs2Camera(){
    
}