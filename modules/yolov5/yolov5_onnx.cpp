#include "yolov5/yolov5_onnx.hpp"

namespace yolov5{

Net::Net(const std::string& onnx_model_path, 
        const std::string& class_list_path, 
        const float resolution_,
        bool is_cuda)
{
    // load model
    this->net = cv::dnn::readNet(onnx_model_path);
    if (is_cuda)
    {
        std::cout << "Attempty to use CUDA\n";
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    // load class_list
    std::ifstream ifs(class_list_path);
    std::string line;
    while (getline(ifs, line))
    {
        this->class_list.push_back(line);
    }

    this->resolution = resolution_;

    this->output_dimensions = 5 + this->class_list.size();
    this->output_rows = 3 *(pow(this->resolution/8,2) + pow(this->resolution/16,2) + pow(this->resolution/32,2)) ;

}

std::vector<yolov5::Detection> Net::detect(cv::Mat &src,
                                    float _score_threshold,
                                    float _NMS_threshold,
                                    float _confidence_threshold){
    cv::Mat blob;
    cv::Mat input = this->format_img(src);

    cv::dnn::blobFromImage(input, blob, 1./255., cv::Size(this->resolution, this->resolution), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input.cols / this->resolution;
    float y_factor = input.rows / this->resolution;

    float *data = (float *)outputs[0].data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < this->output_rows; ++i) {

        float confidence = data[4];
        if (confidence >= _confidence_threshold) {
            float * classes_scores = data + 5;
            
            cv::Mat scores(1, this->class_list.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > _score_threshold) {
                confidences.push_back(confidence);
                
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }
        data += this->output_dimensions;
    }
    
    std::vector<Detection> output;
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, _score_threshold, _NMS_threshold, nms_result);
    for (size_t i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.bbox = boxes[idx];
        output.push_back(result);
    }

    return output;
}

cv::Mat Net::format_img(const cv::Mat &src){
    int format_size = MAX(src.cols, src.rows);

    cv::Mat dst = cv::Mat::zeros(cv::Size(format_size, format_size), CV_8UC3);
    src.copyTo(dst(cv::Rect(0, 0, src.cols, src.rows)));
    
    return dst;
}

} // namespace yolov5