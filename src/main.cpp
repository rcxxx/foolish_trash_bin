#include <iostream>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include "intel_rs/intel_rs.hpp"
#include "solve_pose/solve_pose.hpp"

int main() try
{
    // init camera
    Rs2Camera camera;

    // init solver
    SolvePose slover = SolvePose(camera.Intrinsics());

    for (int i = 0; i < 30; i++)
    {
        //  Wait for all configured streams to produce a frame
        camera.updateFrame();
    }

    namedWindow("color_img", cv::WINDOW_AUTOSIZE);
    namedWindow("depth_img", cv::WINDOW_AUTOSIZE);

    while (cv::waitKey(1) != 27) {
        // update image
        camera.updateFrame();
        cv::Mat color_img = camera.rgbImg();
        cv::Mat depth_img = camera.depthImg();

        // update pose

        // find people

        // find trash

        // find cat

        // show image
        cv::imshow("color_img", color_img);
        cv::imshow("depth_img", depth_img);

    }


    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}