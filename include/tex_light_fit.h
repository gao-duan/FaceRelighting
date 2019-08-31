#pragma once

#include "common.h"
#include "image.h"
#include "BFM.h"
#include "BFM2Pixel.h"
#include "parameter.h"


vec3 SH(const vector<vec3>& sh, const vec3& normal);


struct TexLightOptimizer {
	std::vector<Parameter> params; // cur_params
	int params_num, constraits_num;
	float lr;
	int n_iteration;
	MatrixXd jaccobian;
	VectorXd R;
	vector<float> beta;
	vector<vec3> sh;
	float lambda; // regulization factor

	BFM* bfm;
	vector<vec2i> valid_points;
	vector<FacePoint> face_points;
	vector<vec3> normals;
	vector<bool> valids;

	Texture ref_img;
	Texture depth_img;
	Texture matte_img;
	Texture seg;

	Texture _albedo;

	int n_beta;
	int cur_iter = 0;

	TexLightOptimizer(BFM* _bfm, const vector<vec2i>& _valid_points,
		const vector<FacePoint>& _face_points, const vector<vec3>& _normals,
		const Texture& _ref_img, const Texture& _depth_img, const Texture&, int nt);

	float get_loss(bool flag);

	vector<bool> GD;

	// p: beta and sh, calculate ith point's color
	float f(const std::vector<Parameter>& p, int idx);
	vector<float> f_col(const std::vector<Parameter>&p, bool flag);

	VectorXd get_grad(int i);

	void get_jaccobian(bool flag);

	void get_r(bool flag);

	void update_parameter(const VectorXd& delta);

	void show_result(const std::string&, bool);

	void optimize();

	void save_parameter_to_file(const string& name);
	void load_parameter_from_file(const string& name);
	void load_parameter_from_vector(const vector<float>& _params);


	Texture render_by_parameters(const std::vector<vec3>& new_sh,bool flag = false);
	Texture render_by_parameters(const vec3& dir, const vec3& color);


	Texture generate_final_texture();
	Texture generate_original_texture();
	Texture generate_optmize_image();

	void relighting(const std::vector<vec3>& new_sh, const Texture& albedo);
	void relighting_video(const Texture& albedo, bool prop);

	vector<float> get_params() const;
};