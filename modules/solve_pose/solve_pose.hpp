#ifndef SOLVE_POSE_H
#define SOLVE_POSE_H

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

class SolvePose
{
public:
    SolvePose(rs2_intrinsics &intrinsics);
    ~SolvePose();

    void Solver(std::vector<cv::Point2f> img_2d, float w, float h);
    cv::Point3f coordinateImageToWorld(cv::Point2d img_uv);
    void drawCoordinate(cv::Mat &src);

private:
    cv::Mat camera_matrix, distortion_coeffs;
    cv::Mat rvec = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    cv::Mat tvec = cv::Mat::zeros(3, 1, cv::DataType<double>::type);

    std::vector<cv::Point3f> object_point;
    std::vector<cv::Point2f> image_point;
};

#endif // SOLVE_POSE_H
