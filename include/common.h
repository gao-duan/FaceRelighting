#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <map>
#include <iomanip> 
#include <limits>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/LU"
#include "Eigen/SparseQR"
#include "Eigen/SparseLU"
#include "Eigen/IterativeLinearSolvers"
#include "Eigen/SparseCholesky"

#include <Windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif 

using std::vector;
using cv::Mat;
using cv::imread;
using cv::imwrite;
using cv::IMREAD_COLOR;
using cv::Vec3b;
using cv::namedWindow;
using cv::WINDOW_AUTOSIZE;
using cv::waitKey;
using cv::Vec2i;
using cv::Vec4f;
using cv::Vec3f;
using cv::Vec2f;
using cv::Vec4b;
using cv::Mat1f;

using namespace std;


using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::SparseQR;
using Eigen::SparseLU;
using Eigen::SparseMatrix;
using Eigen::ConjugateGradient;

const float FloatMax = std::numeric_limits<float>::max();
const float FloatMin = std::numeric_limits<float>::min();
const float FloatInfity = std::numeric_limits<float>::infinity();

const float PI = 3.14159265358979323846;
extern string solution_dir;
extern std::string res_path;

bool file_exists(const std::string& name);
template<typename T>
T Clamp(T x, T a, T b) {
	if (x < a) return a;
	if (x > b)return b;
	else return x;
}
template<typename T>
T random(T begin, T end) {
	return ((T)rand()) / (T)RAND_MAX * (end - begin) + begin;
}




