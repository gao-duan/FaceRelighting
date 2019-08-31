#pragma once
#include <fstream>
#include <vector>
#include <map>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

class LandmarkDetectorImpl;

class LandmarkDetectorWrapper {
private:
	LandmarkDetectorImpl* impl;
public:
	LandmarkDetectorWrapper(const std::string& _model_path="", const std::string& _classfier_path="");
	~LandmarkDetectorWrapper();
	std::vector<std::pair<float, float> > detect_landmarks_from_single_img(const std::string& img_path);
	std::vector<std::pair<float, float> > detect_landmarks_from_single_img(const cv::Mat& img);
	std::vector<std::vector<std::pair<float, float> > > detect_landmarks_from_multiple_img(const std::vector<std::string>& imgs_path);
	std::vector<std::vector<std::pair<float, float> > > detect_landmarks_from_multiple_img(const std::vector<cv::Mat>& imgs);
};
