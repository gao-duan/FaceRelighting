#include "landmark_detection.h"

#include "stdafx.h"
#include <LandmarkCoreIncludes.h>

#include <dlib/image_processing/frontal_face_detector.h>
#include <tbb/tbb.h>
#include "common.h"

static void convert_to_grayscale(const cv::Mat& in, cv::Mat& out)
{
	if (in.channels() == 3)
	{
		// Make sure it's in a correct format
		if (in.depth() != CV_8U)
		{
			if (in.depth() == CV_16U)
			{
				cv::Mat tmp = in / 256;
				tmp.convertTo(tmp, CV_8U);
				cv::cvtColor(tmp, out, CV_BGR2GRAY);
			}
		}
		else
		{
			cv::cvtColor(in, out, CV_BGR2GRAY);
		}
	}
	else if (in.channels() == 4)
	{
		cv::cvtColor(in, out, CV_BGRA2GRAY);
	}
	else
	{
		if (in.depth() == CV_16U)
		{
			cv::Mat tmp = in / 256;
			out = tmp.clone();
		}
		else if (in.depth() != CV_8U)
		{
			in.convertTo(out, CV_8U);
		}
		else
		{
			out = in.clone();
		}
	}
}



class LandmarkDetectorImpl {
private:
	LandmarkDetector::FaceModelParameters det_parameters;
	LandmarkDetector::CLNF clnf_model;
	cv::CascadeClassifier classifier;
	dlib::frontal_face_detector face_detector_hog;
public:
	LandmarkDetectorImpl(const string& model_path = "", const string& classifier_path = "");
	std::vector<std::pair<float, float> > detect_landmarks_from_single_img(const string& img_path);
	std::vector<std::pair<float, float> > detect_landmarks_from_single_img(const cv::Mat& img);

	std::vector<std::vector<std::pair<float, float> > > detect_landmarks_from_multiple_img(const std::vector<string>& imgs_path);
	std::vector<std::vector<std::pair<float, float> > > detect_landmarks_from_multiple_img(const std::vector<cv::Mat>& imgs);
};

std::vector<std::pair<float, float>> LandmarkDetectorImpl::detect_landmarks_from_single_img(const string & img_path)
{
	cv::Mat read_image = cv::imread(img_path, -1);
	return detect_landmarks_from_single_img(read_image);
}

std::vector<std::pair<float, float>> LandmarkDetectorImpl::detect_landmarks_from_single_img(const cv::Mat & img)
{

	cv::Mat_<uchar> grayscale_image;
	convert_to_grayscale(img, grayscale_image);
	
	vector<cv::Rect_<double> > face_detections;
	vector<double> confidences;

	std::vector<std::pair<float, float>> res;

	if (det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
	{
		LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, face_detector_hog, confidences);
	}
	else
	{
		LandmarkDetector::DetectFaces(face_detections, grayscale_image, classifier);
	}

	// Detect landmarks around detected faces
	int face_det = 0;
	// perform landmark detection for every face detected

	for (size_t face = 0; face < face_detections.size(); ++face)
	{
		// if there are multiple detections go through them
		bool success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, face_detections[face], clnf_model, det_parameters);
		if (!success) {
			continue;
		}

		int size = (clnf_model.detected_landmarks.size)[0];

		bool valid = true;
		for (int j = 0; j < size; ++j) {
			if (isnan(clnf_model.detected_landmarks.at<double>(j, 0))) {
				valid = false;
				break;
			}
		}
		if (!valid) {
			continue;
		}

		for (int j = 0; j < size / 2; ++j) {
			float x = clnf_model.detected_landmarks.at<double>(j, 0);
			float y = clnf_model.detected_landmarks.at<double >(j + size / 2, 0);
			res.push_back(std::make_pair(x,y));
		}

		break;
	}
	return res;
}

std::vector<std::vector<std::pair<float, float>>> LandmarkDetectorImpl::detect_landmarks_from_multiple_img(const std::vector<string>& imgs_path)
{
	std::vector<std::vector<std::pair<float, float>>> res;
	for (auto path : imgs_path) {
		std::vector<std::pair<float, float> > tmp = detect_landmarks_from_single_img(path);
		res.push_back(tmp);
	}
	return res;
}

std::vector<std::vector<std::pair<float, float>>> LandmarkDetectorImpl::detect_landmarks_from_multiple_img(const std::vector<cv::Mat>& imgs)
{
	std::vector<std::vector<std::pair<float, float>>> res;
	for (auto img : imgs) {
		std::vector<std::pair<float, float> > tmp = detect_landmarks_from_single_img(img);
		res.push_back(tmp);
	}
	return res;
}

LandmarkDetectorImpl::LandmarkDetectorImpl(const string & _model_path, const string & _classifier_path)
{
	string default_model_path = solution_dir + "data/model/main_clnf_general.txt";
	string default_classifier_path = solution_dir + "data/classifiers/haarcascade_frontalface_alt.xml";

	string model_path = (_model_path.empty()) ? default_model_path : _model_path;
	string classifier_path = (_classifier_path.empty()) ? default_classifier_path : _classifier_path;

	det_parameters = LandmarkDetector::FaceModelParameters();
	det_parameters.validate_detections = false;
	det_parameters.model_location = model_path;
	det_parameters.face_detector_location = classifier_path;

	clnf_model = LandmarkDetector::CLNF(det_parameters.model_location);
	classifier = cv::CascadeClassifier(det_parameters.face_detector_location);
	face_detector_hog = dlib::get_frontal_face_detector();
}

LandmarkDetectorWrapper::LandmarkDetectorWrapper(const string & _model_path, const string & _classfier_path)
{
	impl = new LandmarkDetectorImpl(_model_path, _classfier_path);
}

LandmarkDetectorWrapper::~LandmarkDetectorWrapper()
{
	delete impl;
}

std::vector<std::pair<float, float>> LandmarkDetectorWrapper::detect_landmarks_from_single_img(const string & img_path)
{
	return impl->detect_landmarks_from_single_img(img_path);
}

std::vector<std::pair<float, float>> LandmarkDetectorWrapper::detect_landmarks_from_single_img(const cv::Mat & img)
{
	return impl->detect_landmarks_from_single_img(img);
}

std::vector<std::vector<std::pair<float, float>>> LandmarkDetectorWrapper::detect_landmarks_from_multiple_img(const std::vector<string>& imgs_path)
{
	return impl->detect_landmarks_from_multiple_img(imgs_path);
}

std::vector<std::vector<std::pair<float, float>>> LandmarkDetectorWrapper::detect_landmarks_from_multiple_img(const std::vector<cv::Mat>& imgs)
{
	return impl->detect_landmarks_from_multiple_img(imgs);
}
