#ifndef POSITION_H
#define POSITION_H

#include <opencv2/opencv.hpp>

void vertexesSort(std::vector<cv::Point2f> &vertexes){
    sort(vertexes.begin(), vertexes.end(), [](cv::Point2f p1, cv::Point2f p2){return p1.x < p2.x;});
    cv::Point2f tl = (vertexes[0].y < vertexes[1].y ? vertexes[0]:vertexes[1]);
    cv::Point2f bl = (vertexes[0].y > vertexes[1].y ? vertexes[0]:vertexes[1]);
    cv::Point2f tr = (vertexes[2].y < vertexes[3].y ? vertexes[2]:vertexes[3]);
    cv::Point2f br = (vertexes[2].y > vertexes[3].y ? vertexes[2]:vertexes[3]);

    vertexes.clear();
    vertexes.push_back(tl);
    vertexes.push_back(tr);
    vertexes.push_back(br);
    vertexes.push_back(bl);
}

cv::Point solveCollinearPoints(const cv::Point &p1, const cv::Point &p2){

    if (p1.x == p2.x){
        return p2.y > p1.y ? cv::Point(p2.x, p2.y + 10) : cv::Point(p2.x, p2.y - 10);
    }

    if (p1.y == p2.y){
        return p2.x > p1.x ? cv::Point(p2.x + 10, p2.y) : cv::Point(p2.x - 10, p2.y);
    }
    float k = ((p2.y- p1.y) / (p2.x - p1.x));
    float b = p2.y - p2.x * k;

    return p2.x > p1.x ? cv::Point(p2.x + 10, (p2.x + 10) * k + b) : cv::Point(p2.x - 10, (p2.x - 10) * k + b);
}

#endif