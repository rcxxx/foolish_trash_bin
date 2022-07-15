#include "yolov5/yolov5_openvino.hpp"

namespace yolov5
{

DetectorOpenVINO::DetectorOpenVINO(const std::string& model_path_,
        const std::string& class_list_path,
        const bool& is_GPU_,
        const int& resolution_){
    std::vector<std::string> availableDevices = this->ie.get_available_devices();
    for (size_t i = 0; i < availableDevices.size(); i++) {
        printf("supported device name : %s \n", availableDevices[i].c_str());
    }

    network = ie.read_model(model_path_);

    this->networkInfo(*network);

    ov::preprocess::PrePostProcessor network_ppp(network);
    network_ppp.input().tensor().set_element_type(ov::element::f32);
    network_ppp.input().model().set_layout("NCHW");
    this->network = network_ppp.build();

    // Loading a model to the device
    if (is_GPU_)
    {
        auto device_name = "GPU";
        executable_network = ie.compile_model(network, device_name);
    }
    else
    {
        executable_network = ie.compile_model(network, "CPU");
    }

    // load class_list
    std::ifstream ifs(class_list_path);
    std::string line;
    while (getline(ifs, line))
    {
        this->class_list.push_back(line);
    }

    this->resolution = resolution_;
}

std::vector<yolov5::Detection> DetectorOpenVINO::detect(const cv::Mat &src_, 
        const float & confidence_th_,
        const float& iou_th_){
    ov::Shape input_shape = {1, 3, 
    static_cast<unsigned long>(this->resolution), static_cast<unsigned long>(this->resolution)};

    float* blob = nullptr;

    std::vector<int64_t> input_tensor_shape{ 1, 3, -1, -1 };

    // format input image
    cv::Mat fmt_img = this->format_img(src_);
    
    cv::Mat resize_img, resize_img_float;

    float resize_scale = (float)fmt_img.size().width / (float)this->resolution;

    cv::resize(fmt_img, resize_img, cv::Size(this->resolution, this->resolution));
    cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);

    input_tensor_shape[2] = resize_img.rows;
    input_tensor_shape[3] = resize_img.cols;

    resize_img.convertTo(resize_img_float, CV_32FC3, 1.0 /255.0);
    blob = new float[resize_img_float.cols * resize_img_float.rows * resize_img_float.channels()];
    cv::Size img_f_size{resize_img_float.cols, resize_img_float.rows };

    // // hwc -> chw
    // std::vector<cv::Mat> chw(resize_img_float.channels());
    // for (int i = 0; i < resize_img_float.channels(); ++i)
    // {
    //     chw[i] = cv::Mat(resize_img_float, CV_32FC1, blob + i * img_f_size.width * img_f_size.height);
    // }
    // cv::split(resize_img_float, chw);
    ov::element::Type input_type = ov::element::f32;
    ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, blob);

    ov::InferRequest infer_request = executable_network.create_infer_request();

    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    // cv::Size resized_shape = cv::Size((int)input_tensor_shape[3], (int)input_tensor_shape[2]);

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    const ov::Tensor output_tensor = infer_request.get_output_tensor(0);
    const ov::Shape output_shape = infer_request.get_output_tensor(0).get_shape();
    auto *batch_data = output_tensor.data<const float>();

    std::vector<float> output(batch_data, batch_data + output_tensor.get_size());

    int num_classes = (int)output_shape[2] - 5;
    int elements_in_batch = (int)(output_shape[1] * output_shape[2]);

    for (auto it = output.begin(); it != output.begin() + elements_in_batch; it += output_shape[2]){
        float class_confidence = it[4];

        if (class_confidence > confidence_th_){
            int centerX = (int)(it[0]);
            int centerY = (int)(it[1]);
            int width = (int)(it[2]);
            int height = (int)(it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float obj_confidence;
            int class_id;

            this->getBestClassInfo(it, num_classes, obj_confidence, class_id);

            float confidence = class_confidence + obj_confidence;

            boxes.emplace_back(left, top, width, height);
            confidences.emplace_back(confidence);
            class_ids.emplace_back(class_id);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_th_, iou_th_, indices);
    std::cout<<indices.size()<<std::endl;
    std::vector<yolov5::Detection> detections;

    for (int i : indices)
    {
        yolov5::Detection det;
        det.bbox = scaleBBox(cv::Rect(boxes[i]), resize_scale);
        det.confidence = confidences[i];
        det.class_id = class_ids[i];
        detections.emplace_back(det);
    }

    delete[] blob;

    return detections;
}

void DetectorOpenVINO::getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
    float& bestConf, int& bestClassId)
{
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }

}

cv::Rect DetectorOpenVINO::scaleBBox(const cv::Rect &bbox_, const float &scale_){
    return cv::Rect{ static_cast<int>(bbox_.x * scale_), 
                    static_cast<int>(bbox_.y * scale_), 
                    static_cast<int>(bbox_.width * scale_), 
                    static_cast<int>(bbox_.height * scale_)};
}

void DetectorOpenVINO::networkInfo(const ov::Model& network){
    std::cout << "Network inputs:" << std::endl;
    for (auto&& input : network.inputs()) {
        std::cout << "    " << input.get_any_name() << " (node: " << input.get_node()->get_friendly_name()
            << ") : " << input.get_element_type() << " / " << ov::layout::get_layout(input).to_string()
            << std::endl;
    }

    std::cout << "Network outputs:" << std::endl;
    for (auto&& output : network.outputs()) {
        std::string out_name = "***NO_NAME***";
        std::string node_name = "***NO_NAME***";

        // Workaround for "tensor has no name" issue
        try {
            out_name = output.get_any_name();
        }
        catch (const ov::Exception&) {
        }
        try {
            node_name = output.get_node()->get_input_node_ptr(0)->get_friendly_name();
        }
        catch (const ov::Exception&) {
        }

        std::cout << "    " << out_name << " (node: " << node_name << ") : " << output.get_element_type() << " / "
            << ov::layout::get_layout(output).to_string() << std::endl;
    }
}

cv::Mat DetectorOpenVINO::format_img(const cv::Mat &src){
    int format_size = MAX(src.cols, src.rows);
    cv::Mat dst = cv::Mat::zeros(cv::Size(format_size, format_size), CV_8UC3);
    src.copyTo(dst(cv::Rect(0, 0, src.cols, src.rows)));
    return dst;
}

DetectorOpenVINO::~DetectorOpenVINO()
{
}

} // namespace yolov5
