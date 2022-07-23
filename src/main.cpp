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
    const float GROUND_W = 3.7;
    const float GROUND_H = 1.8;

    // init tag detector
    AprilTagDetector tag_detector;

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

    namedWindow("color_img", cv::WINDOW_AUTOSIZE);

    while (cv::waitKey(1) != 27) {
        // update image
        camera.updateFrame();
        cv::Mat color_img = camera.rgbImg();
        cv::Mat src_img;
        cv::flip(color_img, src_img, -1);
        cv::Mat dst_img;
        src_img.copyTo(dst_img);
        
        // update pose
        cv::Mat gray_img;
        cv::cvtColor(src_img, gray_img, cv::COLOR_BGR2GRAY);
        zarray_t *detections = tag_detector.detectTag(gray_img);
        std::map<int ,apriltag_detection_t> april_tags;
        for(int i = 0; i < zarray_size(detections); ++i){
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);
            april_tags[det->id] = *det;
           for(size_t i = 0; i < 4; ++i){
                cv::line(dst_img, cv::Point(static_cast<int>(det->p[i][0]), static_cast<int>(det->p[i][1])),
                            cv::Point(static_cast<int>(det->p[(i+1)%4][0]), static_cast<int>(det->p[(i+1)%4][1])),
                            cv::Scalar(255, 255, 0), 2);
           }
        }

        if(april_tags.size() > 3){
            // four apriltag midpoints as plane vertices
            std::vector<cv::Point2f> plane_2d;
            plane_2d.clear();
            for(int i = 0; i < 4; ++i){
                cv::Point c = cv::Point(static_cast<int>(april_tags[i].c[0]), static_cast<int>(april_tags[i].c[1]));
                plane_2d.push_back(c);
            }
            vertexesSort(plane_2d);
            slover.Solver(plane_2d, GROUND_W * 1000.0, GROUND_H * 1000.0);
            slover.drawCoordinate(dst_img);
        }

        // YOLO detect
        std::vector<yolov5::Detection> result =  yolo.detect(src_img);
        
        yolov5::Detection trash_bin;
        std::vector<yolov5::Detection> peoples;
        std::vector<yolov5::Detection> hands;
        std::vector<yolov5::Detection> cats;
        for (size_t i = 0; i < result.size(); ++i)
        {
                        switch (result[i].class_id)
            {
            case 0: {
                peoples.emplace_back(result[i]);
            }
                break;
            case 1: {
                trash_bin = result[i];
            }
                break;
            case 2: {
                hands.emplace_back(result[i]);
                
            }
            case 3: {
                cats.emplace_back(result[i]);
                
            }
                break;
            }
            auto detection = result[i];
            auto bbox = detection.bbox;
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];
            cv::rectangle(color_img, bbox, color, 2);
            cv::rectangle(color_img, cv::Point(bbox.x, bbox.y - 20), cv::Point(bbox.x + bbox.width, bbox.y), color, cv::FILLED);
            cv::putText(color_img, yolo.classList()[classId].c_str(), cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
        
        // find trash

        if (trash_bin.class_id != -1){
            cv::Point t_base = cv::Point(trash_bin.bbox.x + trash_bin.bbox.width * 0.5, trash_bin.bbox.y + trash_bin.bbox.height);
            cv::Point3f t_base_3d = slover.coordinateImageToWorld(t_base);
            // find people
            for (size_t i = 0; i < peoples.size(); ++i){
                cv::Rect p_rect = peoples[i].bbox;
                for (size_t j = 0; j < hands.size(); j++)
                {
                    cv::Rect h_rect = hands[j].bbox;
                    cv::Rect intersection = p_rect | h_rect;
                    if(intersection.area() > 0){
                        cv::Point p_base = cv::Point(peoples[i].bbox.x + peoples[i].bbox.width * 0.5, peoples[i].bbox.y + peoples[i].bbox.height);
                        cv::Point3f p_base_3d = slover.coordinateImageToWorld(p_base);
                        float dis = sqrt(pow(t_base_3d.x - p_base_3d.x, 2) + pow(t_base_3d.y - p_base_3d.y, 2));
                        if (dis >= 800){
                            cv::line(dst_img, t_base, p_base, cv::Scalar(150, 105, 255), 4);
                        } else {
                            cv::line(dst_img, t_base, p_base, cv::Scalar(40, 255, 40), 4);
                        }
                        
                        break;
                    }
                }
            }
            // find cat
            for (size_t i = 0; i < cats.size(); ++i){
                cv::Rect c_rect = cats[i].bbox;
                cv::Point c_base = cv::Point(cats[i].bbox.x + cats[i].bbox.width * 0.5, cats[i].bbox.y + cats[i].bbox.height);
                cv::Point3f c_base_3d = slover.coordinateImageToWorld(c_base);
                float dis = sqrt(pow(t_base_3d.x - c_base_3d.x, 2) + pow(t_base_3d.y - c_base_3d.y, 2));
            }
        }

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