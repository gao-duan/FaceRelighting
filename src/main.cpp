#define _CRTDBG_MAP_ALLOC  
#include <stdlib.h>  
#include <crtdbg.h>  

#include <opencv2/photo.hpp>
#include "BFM2Pixel.h"
#include "image.h"
#include "math_vec.h"
#include "landmark_detection.h"
#include "landmark.h"
#include "geometry_fit.h"
#include "tex_light_fit.h"
#include "specular_removal_grad.h"


void optimize_single_image(const string& src_path, Texture& optmize, Texture& matte,
	Texture& orig_albedo, Texture& optmize_albedo, Texture& specular_remove_albedo, std::vector<vec3i> landmarks,
	BFM2Pixel& bfm2pixel, std::vector<float>& geo_params, std::vector<float>& render_params, bool relit, bool prop)
{
	const int ALPHA_SIZE = 99, BETA_SIZE = 99, GAMMA_SIZE = 29;
	std::vector<float> land_3d = { 22117, 22009, 21903, 21279, 42698, 44354, 45933, 46940, 47915,
		48566, 49444, 51017, 52827, 33416, 32603, 32323, 32429, 38694, 39286, 39751, 40015, 40182,
		40840, 41088, 41298, 41696, 42235, 8149, 8174, 8186, 8191, 6634, 7122, 8202, 9282, 9638,
		2600, 4014, 5047, 6466, 4804, 3644, 9833, 11093, 12383, 13420, 12527, 11495, 5263, 5898,
		7375, 8215, 9055, 10523, 11435, 9675, 8835, 8236, 7635, 6672, 5650, 7624, 8224, 8824, 10664,
		8708, 8228, 7749 };

	// 1. initial BFM and landmark detector
	BFM bfm;
	bfm.load(solution_dir + "data/BFM/bfm_data", solution_dir + "data/BFM/bfm_exp", solution_dir + "data/BFM/bfm_attrib.txt");
	LandmarkDetectorWrapper detector;

	cout << "Begin optmize" << endl;

	DWORD tm_beg = GetTickCount();
	// 2. detect landmarks of image, generate matte
	std::vector<std::pair<float, float>>  res = detector.detect_landmarks_from_single_img(src_path);
	vector<cv::Point> points;
	landmarks.clear();
	for (size_t i = 0; i < res.size(); ++i) {
		points.push_back(cv::Point(res[i].first, res[i].second));
		landmarks.push_back(vec3i(res[i].first, res[i].second, land_3d[i]));
	}
	matte = generate_matte_from_landmarks(points);

	Landmarks lm;
	bool ret = lm.load(&bfm, landmarks);
	PoseOptimizer geo_opt(9 + ALPHA_SIZE + GAMMA_SIZE, lm.size, 40, &bfm, &lm, 250, 250);
	geo_opt.optimize();
	geo_params = geo_opt.get_params();
	geo_opt.save_parameter_to_file(res_path + "geo_params.txt");

	{
		bfm.update_vertices();

		auto fp = std::bind(&PoseOptimizer::_transform, &geo_opt, _1);

		bfm.update_by_pose_optimize(fp);
	}

	vector<vec2i> valid_points = get_valid_points_from_matte(matte);
	bfm2pixel = BFM2Pixel(&bfm, valid_points, bfm.vertices, bfm.tl, "");
	bfm2pixel.map(true);

	vector<vec3> normals;
	normals = get_normals_from_bfm(&bfm, bfm2pixel.face_points);

	Texture ref_img(src_path);

	vector<vec2i> valid_points_sampled;
	vector<vec3> normals_sampled;
	vector<FacePoint> face_points_sampled;

	for (int i = 0; i < valid_points.size(); i += 10) {
		valid_points_sampled.push_back(valid_points[i]);
		normals_sampled.push_back(normals[i]);
		face_points_sampled.push_back(bfm2pixel.face_points[i]);
	}

	TexLightOptimizer render_opt(&bfm, valid_points, bfm2pixel.face_points,
		normals, ref_img, Texture(), matte, 15);

	render_opt.optimize();
	render_opt.save_parameter_to_file(res_path + "render_params.txt");

	DWORD tm_end = GetTickCount();

	cout << "Time: " << (tm_end - tm_beg) / 1000.0 << " s" << endl;

	render_params = render_opt.get_params();

	optmize_albedo = render_opt.generate_final_texture();
	orig_albedo = render_opt.generate_original_texture();
	optmize = render_opt.generate_optmize_image();

	specular_remove_albedo = specular_removal(optmize_albedo, orig_albedo, matte, points, 1.0f);
	if(relit)
		render_opt.relighting_video(specular_remove_albedo, prop);
}

std::string res_path;
std::string solution_dir;

int main(int argc, char** argv) {
	if (argc < 2) {
		std::cout << "missing parameters." << std::endl;
		return 0;
	}

	Eigen::setNbThreads(6);
	cout <<"Threads: "<< Eigen::nbThreads() << endl;

	solution_dir = std::string(argv[1]);
	res_path = std::string(argv[2]);
	string src_path = std::string(argv[3]);
	bool relit = true;
	bool prop = false; // a little bit slow for propagation process.

	Texture optmize, matte, orig_albedo, optmize_albedo, specular_remove_albedo;
	std::vector<vec3i> landmarks;
	BFM2Pixel bfm2pixel;
	std::vector<float> geo_params, render_params;
	optimize_single_image(src_path, optmize, matte, orig_albedo, optmize_albedo, 
		specular_remove_albedo, landmarks, bfm2pixel, geo_params, render_params,
		relit, prop);
	
	specular_remove_albedo.save(res_path + "albedo_high.bmp");
	orig_albedo.save(res_path + "albedo_low.bmp");
	optmize.save(res_path + "reconstruct.bmp");

	
	system("pause");

	return 0;
}
