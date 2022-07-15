#ifndef INTEL_RS_H
#define INTEL_RS_H

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

class Rs2Camera
{
public:
    Rs2Camera(const int &r_w, const int & r_h, const int & fps);
    ~Rs2Camera();

    /**
     * @brief update camera frame
     * 
     */
    void updateFrame();

    /**
     * @brief convert color_frames to cv::Mat
     * 
     * @return cv::Mat
     */
    cv::Mat rgbImg();

    /**
     * @brief convert depth_frames to cv::Mat
     * 
     * @return cv::Mat 
     */
    cv::Mat depthImg();

    /**
     * @return &intrinsics
     */
    inline rs2_intrinsics&  Intrinsics(){
        return this->intrinsics;
    }

private:
    rs2::colorizer          color_map;
    rs2::align              align_to;
    rs2::pipeline           pipe;
    rs2::pipeline_profile   pipeProfile;
    rs2_intrinsics          intrinsics;

    rs2::frame color_frames;
    rs2::frame depth_frames;
};

#endif // INTEL_RS_H