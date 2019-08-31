#pragma once
#include "math_vec.h"
#include "BFM.h"
#include "BFM2Pixel.h"

class Landmarks {
public:
	vector<int> landmarks_indices;
	vector<vec4> landmarks_3d;
	vector<vec2i> landmarks_2d;
	vector<string> landmarks_name;
	int size;

	bool load(BFM* bfm, std::vector<vec3i>& data);
	bool load(BFM* bfm, const string& name);
};

class Landmarks3D {
public:
	vector<int> landmarks_indices;
	vector<vec4> landmarks_bfm;
	vector<vec3> landmarks_obj;
	vector<string> landmarks_name;
	int size;

	bool load(BFM* bfm, const string& name, const string& obj_name);
};

std::vector<vec2i> get_valid_points_from_obj(const string & name);
std::vector<vec2i> get_valid_points_from_matte(const Texture& matte);

std::vector<vec3> get_points_from_obj(const string& name);
std::vector<vec3> cal_normals_from_obj(const string & name);
std::vector<vec3> load_normals_from_obj(const string& name);
std::vector<vec3> get_normals_from_bfm(BFM* bfm, const vector<FacePoint>&);
Image depth_map(const string& name);
Image matte_map(const string& name);

std::vector<int> get_invalids_normal(const string & name);
std::vector<vec3> correct_normals(const vector<vec2i>& valid_points, const vector<vec3>& normals_bfm, vector<vec3>& normals, const string&);
std::vector<vec3> get_normal_from_normalmap(const Texture& normal_map, const vector<vec2i> &valid_points);

Texture generate_matte_from_landmarks(std::vector<cv::Point>& points);
Texture generate_matte_from_landmarks(const string& name);
Texture generate_eye_mouth_matte_from_landmarks(const string& name);
Texture generate_eye_mouth_matte_from_landmarks(const std::vector<cv::Point>& points);

Texture generate_nose_matte_from_landmarks(const string& name);