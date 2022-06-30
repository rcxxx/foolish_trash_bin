#include "solve_pose/solve_pose.hpp"

SolvePose::SolvePose(rs2_intrinsics &intrinsics)
{
    //读取相机内参和畸变矩阵
    double fx = static_cast<double>(intrinsics.fx);
    double fy = static_cast<double>(intrinsics.fy);
    double cx = static_cast<double>(intrinsics.ppx);
    double cy = static_cast<double>(intrinsics.ppy);

    this->camera_matrix = (cv::Mat_<double>(3, 3) <<   fx, 0,  cx,
                                                 0,  fy, cy,
                                                 0,  0,  1);

    // intrinsics.coeffs[0~4] == 0
    this->distortion_coeffs = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);

    std::cout << camera_matrix << std::endl;
    std::cout << distortion_coeffs << std::endl;
}

void SolvePose::Solver(std::vector<cv::Point2f> img_2d, float w, float h){

    /*    World coordinates

                     ▲z
                     |
                     |
                     |
                     |
                     |        w         p1-(w, 0, 0)
         p0-(0, 0, 0).-----------------.-----▷ x
                    /                 /
                   /                 /
                  /                 /
                 /                 / h
                /                 /
               /                 /
  p3-(0, h, 0)._________________/.p2-(w, h, 0)
             /
            /
           ◣ y
    */
    object_point.clear();
    // cv::SOLVEPNP_IPPE
    object_point.push_back(cv::Point3f(0, 0, 0));
    object_point.push_back(cv::Point3f(w, 0, 0));
    object_point.push_back(cv::Point3f(w, h, 0));
    object_point.push_back(cv::Point3f(0, h, 0));
//    // cv::SOLVEPNP_IPPE_SQUARE
//    object_point.push_back(cv::Point3f(-w/2, -h/2, 0));
//    object_point.push_back(cv::Point3f(w/2, -h/2, 0));
//    object_point.push_back(cv::Point3f(w/2, h/2, 0));
//    object_point.push_back(cv::Point3f(-w/2, h/2, 0));

    /*     Image coordinates
     _____________________________________
    |      x                             |
    |   ----x---▷                        |
    |  |                                 |
    | y|        .         .              |
    |  ▼       p0-(x,y)   p3-(x,y)       |
    |                                    |
    |      .                  .          |
    |       p1-(x,y)          p2-(x,y)   |
    |                                    |
    |____________________________________|
    */
    this->image_point = img_2d;

    // solve rotation (rvec) and translation (tvec) vectors
    cv::solvePnP(object_point, image_point, this->camera_matrix, this->distortion_coeffs, this->rvec, this->tvec, false,cv::SOLVEPNP_IPPE);
}

cv::Point3f SolvePose::coordinateImageToWorld(cv::Point2d img_point){
    cv::Mat rotation_matrix = cv::Mat(3,3,cv::DataType<double>::type);
    cv::Rodrigues(this->rvec, rotation_matrix);

    /* img point (image coordinate)
      \begin{bmatrix}
          u \\
          v \\
          1
      \end{bmatrix}
    */
    cv::Mat uv_pt = cv::Mat::ones(3,1,cv::DataType<double>::type);
    uv_pt.at<double>(0,0) = static_cast<double>(img_point.x);
    uv_pt.at<double>(1,0) = static_cast<double>(img_point.y);

    cv::Mat M_1, M_2;
    double S, z_const = 0;
    M_1 = rotation_matrix.inv() * camera_matrix.inv() * uv_pt;
    M_2 = rotation_matrix.inv() * tvec;

    /* S = (z_const + M_2[2]) / M_1[2] */
    S = z_const + M_2.at<double>(2,0);
    S /= M_1.at<double>(2,0);

    /* world_pt = R^{-1}(M_{camera}^{-1} * S * uv_pt - t) */
    cv::Mat world_pt = rotation_matrix.inv() * (S * camera_matrix.inv() * uv_pt - tvec);
    cv::Point3f dst_pt = cv::Point3d(world_pt.at<double>(0, 0), world_pt.at<double>(1, 0), world_pt.at<double>(2, 0));

    return dst_pt;
}

void SolvePose::drawCoordinate(cv::Mat &src)
{
    std::vector<cv::Point2f> reference_img;
    std::vector<cv::Point3f> reference_obj;
    reference_obj.clear();
    reference_obj.push_back(cv::Point3f(0.0, 0.0, 0.0));    //  origin
    reference_obj.push_back(cv::Point3f(100, 0.0, 0.0));    //  x-axis
    reference_obj.push_back(cv::Point3f(0.0, 100, 0.0));    //  y-axis
    reference_obj.push_back(cv::Point3f(0.0, 0.0, 100));    //  z-axis

    cv::projectPoints(reference_obj, rvec, tvec, camera_matrix, distortion_coeffs, reference_img);

    // x
    cv::line(src, reference_img[0], reference_img[1], cv::Scalar(0, 0, 255), 2);
    cv::putText(src, "x", reference_img[1], cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);
    // y
    cv::line(src, reference_img[0], reference_img[2], cv::Scalar(0, 255, 0), 2);
    cv::putText(src, "y", reference_img[2], cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 1);
    // z
    cv::line(src, reference_img[0], reference_img[3], cv::Scalar(255, 0, 0), 2);
    cv::putText(src, "z", reference_img[3], cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0), 1);
}

SolvePose::~SolvePose(){

}
