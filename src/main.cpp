#include <iostream>
#include <unistd.h>
#include <map>

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include "intel_rs/intel_rs.hpp"
#include "solve_pose/solve_pose.hpp"
#include "april_tag/april_tag.hpp"
#include "yolov5/yolov5_onnx.hpp"
#include "yolov5/yolov5_openvino.hpp"

#include "position.hpp"

int main() try
{
    // init camera
    Rs2Camera camera(1280, 720, 30);

    // init solver
    SolvePose slover = SolvePose(camera.Intrinsics());
    const float GROUND_W = 1;
    const float GROUND_H = 1;

    // init tag detector
    AprilTagDetector tag_detector;
    const int TRASH_TAG = 24;

    // init yolo
    // std::string model_path = "../models/yolov5s-480x.onnx";
    // std::string classes_path = "../models/classes.txt";
    
    std::string model_path = "../models/best.onnx";
    std::string classes_path = "../models/trash_classes.txt";
    
    yolov5::Net yolo(model_path, classes_path, 480);
    // yolov5::DetectorOpenVINO yolo(model_path, classes_path, 480);
    const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

    // yolov5::DetectorOpenVINO yolo()

    for (int i = 0; i < 30; i++)
    {
        //  Wait for all configured streams to produce a frame
        camera.updateFrame();
    }

    namedWindow("depth_img", cv::WINDOW_AUTOSIZE);
    namedWindow("color_img", cv::WINDOW_AUTOSIZE);

    while (cv::waitKey(1) != 27) {
        // update image
        camera.updateFrame();
        cv::Mat color_img = camera.rgbImg();
        cv::Mat depth_img = camera.depthImg();
        cv::flip(color_img, color_img, -1);
        // update pose
        cv::Mat gray_img;
        cv::cvtColor(color_img, gray_img, cv::COLOR_BGR2GRAY);

        zarray_t *detections = tag_detector.detectTag(gray_img);

        std::map<int ,apriltag_detection_t> april_tags;
        for(int i = 0; i < zarray_size(detections); ++i){
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);
            april_tags[det->id] = *det;

           for(size_t i = 0; i < 4; ++i){
                std::stringstream _s;
                _s << i;
                cv::String _t = _s.str();
                putText(color_img, _t, cv::Point( static_cast<int>(det->p[i][0]),static_cast<int>(det->p[i][1])),
                        cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
                cv::line(color_img, cv::Point(static_cast<int>(det->p[i][0]), static_cast<int>(det->p[i][1])),
                            cv::Point(static_cast<int>(det->p[(i+1)%4][0]), static_cast<int>(det->p[(i+1)%4][1])),
                            cv::Scalar(255, 255, 0), 2);
           }
        }

        if(april_tags.size() > 4){
            // four apriltag midpoints as plane vertices
            std::vector<cv::Point2f> plane_2d;
            plane_2d.clear();
            for(int i = 0; i < 4; ++i){
                cv::Point c = cv::Point(static_cast<int>(april_tags[i].c[0]), static_cast<int>(april_tags[i].c[1]));
                plane_2d.push_back(c);
            }
            vertexesSort(plane_2d);
            slover.Solver(plane_2d, GROUND_W * 1000.0, GROUND_H * 1000.0);
            slover.drawCoordinate(color_img);
        }

        // YOLO detect
        std::vector<yolov5::Detection> result =  yolo.detect(color_img, 0.001);
        
        for (size_t i = 0; i < result.size(); ++i)
        {
            auto detection = result[i];
            auto bbox = detection.bbox;
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];
            cv::rectangle(color_img, bbox, color, 2);
            cv::rectangle(color_img, cv::Point(bbox.x, bbox.y - 20), cv::Point(bbox.x + bbox.width, bbox.y), color, cv::FILLED);
            cv::putText(color_img, yolo.classList()[classId].c_str(), cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
        
        // find trash

        if(april_tags.contains(TRASH_TAG)){
            cv::Point trash_center  = cv::Point(april_tags[TRASH_TAG].c[0], april_tags[TRASH_TAG].c[1]);
            cv::Point p_0           = cv::Point(static_cast<int>(april_tags[TRASH_TAG].p[0][0]), static_cast<int>(april_tags[TRASH_TAG].p[0][1]));
            cv::Point p_1           = cv::Point(static_cast<int>(april_tags[TRASH_TAG].p[1][0]), static_cast<int>(april_tags[TRASH_TAG].p[1][1]));
            cv::Point trash_face    = solveCollinearPoints(trash_center, cv::Point((p_0.x + p_1.x)*0.5, (p_0.y + p_1.y)*0.5));

            cv::line(color_img, trash_center, trash_face, cv::Scalar(0, 255, 255), 2);
        }

        // find people

        // find cat

        // show image
        
        // cv::imshow("depth_img", depth_img);
        cv::imshow("color_img", color_img);
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