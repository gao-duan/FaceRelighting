#pragma once
#include "BFM.h"
#include "landmark.h"
#include "parameter.h"

class PoseOptimizer  {
private:
	// optimizer data
	std::vector<Parameter> params; // cur_params
	int params_num, constraits_num;
	float lr;
	int n_iteration;
	MatrixXd jaccobian;
	VectorXd R;
	float lambda; // regulization factor
	
	// pose related data
	BFM* bfm;
	Landmarks* lm;
	vector<vec3> ref_normals;

	int width, height;
	float scale;
	float translate_x;
	float translate_y;
	float translate_z;
	float rotate_x;
	float rotate_y;
	float rotate_z;
	float camera_pos;
	float camera_fov;
	float ss;
	vector<float> alpha;
	vector<float> gamma;
	
	void initial_params();
public:
	PoseOptimizer(int _params_num, int _constraits_num, int nt, BFM* bfm, Landmarks* lm, int w, int h);
	vec3 _transform(const vec3& pos)const;

	float get_loss();

	VectorXd get_grad(int i);

	void get_jaccobian();

	void get_r();

	void update_parameter(const VectorXd& delta);

	void show_result(const std::string& path);

	// loss function: i is ith landmark
	virtual float f(const std::vector<Parameter>& p, int i);

	void optimize();
	vector<float> get_final_camera_pose();
	
	vec2 transform_to_2d(const vec3& pos) const;

	void final_transform(vector<float>& vertices, vector<int>& indices);
	void save_parameter_to_file(const string& name);
	bool load_parameter_from_file(const string& name);
	bool load_parameter_from_vector(const vector<float>& _params);
	vector<float> get_params() const;
};