#include "geometry_fit.h"

#define ALPHA_SIZE 99
#define GAMMA_SIZE 29

vec2 PoseOptimizer::transform_to_2d(const vec3 & pos) const
{
	mat4 center_trans = Translate(-bfm->center);
	mat4 scale_trans = Scale(scale * ss);
	mat4 rotate = get_rotation_mat(rotate_x, rotate_y, rotate_z);
	mat4 translate = Translate(translate_x, translate_y, translate_z);
	mat4 transform = translate * rotate * scale_trans * center_trans;
	vec3 p = (transform * pos).div_w().xyz();

	mat4 perspec = Perspective(camera_fov, float(width) / height,  -1,  1);
	p = (perspec  * p).xyz(); // x, y, 1
	vec2 screen_pos = vec2(width * 0.5f * p.x + width * 0.5f * 1.0,
	-height * 0.5f * p.y + height * 0.5f * 1.0);
	return screen_pos;
}

vec3 PoseOptimizer::_transform(const vec3 & pos) const
{
	mat4 center_trans = Translate(-bfm->center);
	mat4 scale_trans = Scale(scale * ss);
	mat4 rotate = get_rotation_mat(rotate_x, rotate_y, rotate_z);
	mat4 translate = Translate(translate_x, translate_y, translate_z);
	mat4 transform = translate * rotate * scale_trans * center_trans;
	vec3 p = (transform * pos).div_w().xyz();
	
	
	mat4 perspec = Perspective(camera_fov, float(width) / height, -1, 1);
	vec4 _p = (perspec  * p); // x, y, 1
	p.x = _p.x; p.y = _p.y; p.z = _p.w;
	float depth = 0.902697861194611 * 200;
	vec3 screen_pos = vec3(width * 0.5f * p.x + width * 0.5f,
		-height * 0.5f * p.y + height * 0.5, depth * 0.5f * p.z + depth * 0.5f);

	return screen_pos;
}

void PoseOptimizer::initial_params()
{
	vec3 diag = bfm->pMax - bfm->center;
	ss = 1.0 / diag.maxComponent();
	params[0].value = 1;
	params[1].value = 0;
	params[2].value = 0;
	params[3].value = 0;
	params[4].value = 0;
	params[5].value = 0;
	params[6].value = 0;
	params[7].value = 6;
	params[8].value = 1;
	params[7].lr = 0.5;
	
	alpha.resize(ALPHA_SIZE, 0);
	gamma.resize(GAMMA_SIZE);
}

PoseOptimizer::PoseOptimizer(int _params_num, int _constraits_num, int nt, BFM * bfm, Landmarks * lm, int w, int h)
	:params_num(_params_num), constraits_num(_constraits_num), n_iteration(nt),
	bfm(bfm), lm(lm), width(w), height(h) {
	params.resize(params_num);
	jaccobian = MatrixXd(constraits_num, params_num);
	R = VectorXd(constraits_num);
	lr = 0.5;
	lambda = 0.1;

	initial_params();
}

float PoseOptimizer::get_loss()
{
	float loss = 0.0f;
	for (int i = 0; i < constraits_num; ++i) {
		float t = f(params, i) * std::sqrt(10.0 / constraits_num / 250);
		loss += t * t;
	}
	return loss;
}

VectorXd PoseOptimizer::get_grad(int i)
{
	VectorXd grad(params_num);
	float delta = 1e-5;
	float loss_cur = f(params, i);
	for (int j = 0; j < params_num; ++j) {
		std::vector<Parameter> new_params(params);
		new_params[j].update(delta);
		float loss_new = f(new_params, i);
		grad[j] = (loss_new - loss_cur) / delta;
	}
	return grad;
}

void PoseOptimizer::get_jaccobian()
{
	for (int i = 0; i < constraits_num; ++i) {
		VectorXd grad = get_grad(i);
		for (int j = 0; j < params_num; ++j) {
			jaccobian(i, j) = grad(j, 0);
		}
	}
	jaccobian *= (sqrt(10.0f / constraits_num / 250.0f));
}

void PoseOptimizer::get_r()
{
	for (int i = 0; i < constraits_num; ++i) {
		R(i, 0) = double(f(params, i)) * sqrt(10.0f / constraits_num / 250.0f);
	}
}

void PoseOptimizer::update_parameter(const VectorXd & delta)
{
	for (int i = 0; i < params_num; ++i) {
		params[i].update(delta(i, 0));
	}
}

void PoseOptimizer::show_result(const std::string& path)
{
	Image img(path);
	for (size_t i = 0; i < lm->landmarks_2d.size(); ++i) {
		img.setPixel(lm->landmarks_2d[i].x, lm->landmarks_2d[i].y, Vec3b(0, 255, 0));
		int vertex_idx = lm->landmarks_indices[i];
		vec3 world_pos = bfm->get_vertex(vertex_idx);
		vec2 project_pos = transform_to_2d(world_pos);
		img.setPixel(project_pos.x, project_pos.y, Vec3b(0, 0, 255));
	}
	img.show();
}

float PoseOptimizer::f(const std::vector<Parameter>& p, int i)
{
	scale = p[0].value;
	translate_x = p[1].value;
	translate_y = p[2].value;
	translate_z = p[3].value;
	rotate_x = p[4].value;
	rotate_y = p[5].value;
	rotate_z = p[6].value;
	camera_pos = p[7].value;
	camera_fov = p[8].value;

	for (int i = 9; i < 9 + alpha.size(); ++i) {
		alpha[i - 9] = p[i].value;
		bfm->alpha[i - 9] = p[i].value;
	}
	for (int i = 9 + alpha.size(); i < p.size(); ++i) {
		gamma[i - 9 - alpha.size()] = p[i].value;
		bfm->gamma[i - 9 - alpha.size()] = p[i].value;
	}

	int vertex_idx = lm->landmarks_indices[i];
	vec3 world_pos = bfm->get_vertex(vertex_idx);;
	vec2 ref_pos = lm->landmarks_2d[i];

	vec2 project_pos = transform_to_2d(world_pos);
	
	return (ref_pos - project_pos).length();
}

void PoseOptimizer::optimize()
{
	VectorXd update_params(params_num);
	

	for (int iter = 0; iter < n_iteration; ++iter) {
		if (iter == 0) {
			for (int i = 0; i < 9; ++i) {
				params[i].lr = 0.15;
			}
			for (int i = 9; i < params.size(); ++i) {
				params[i].lr = 0;
			}
		}
		else if (iter == 10) {
			for (int i = 0; i < 9; ++i) {
				params[i].lr = 0.1;
			}
			params[4].lr = 0.2;

			for (int i = 9; i < params.size(); ++i) {
				params[i].lr = 1.5;
			}
		}
		get_jaccobian();
		get_r();


		MatrixXd jt = jaccobian.transpose();
		MatrixXd jtj = jt * jaccobian + lambda * Eigen::MatrixXd::Identity(params_num, params_num);

		MatrixXd jtj_inv = jtj.inverse();
		MatrixXd jtr = jt * R;
		MatrixXd res = jtj_inv * jtr;
		update_params = -lr * res;

		update_parameter(update_params);
	}
}
vector<float> PoseOptimizer::get_final_camera_pose() {
	vector<float> res = { params[0].value, params[1].value,
		params[2].value, params[3].value,
		params[4].value, params[5].value,
		params[6].value, params[7].value,
		params[8].value
	};
	return res;
}
void PoseOptimizer::final_transform(vector<float>& vertices, vector<int>& indices)
{
	scale = params[0].value;
	translate_x = params[1].value;
	translate_y = params[2].value;
	translate_z = params[3].value;
	rotate_x = params[4].value;
	rotate_y = params[5].value;
	rotate_z = params[6].value;
	camera_pos = params[7].value;
	camera_fov = params[8].value;

	vertices.resize(3 * bfm->n_vertex);
	indices.resize(3 * bfm->n_triangle);

	for (int i = 0; i < bfm->n_triangle * 3; ++i) indices[i] = bfm->tl[i];
	for (int i = 0; i < bfm->n_vertex; ++i) {
		vec3 p(bfm->vertices[3 * i], bfm->vertices[3 * i + 1], bfm->vertices[3 * i + 2]);
		p = _transform(p);
		vertices[3 * i] = p.x;
		vertices[3 * i + 1] = p.y;
		vertices[3 * i + 2] = p.z;
	}
}

void PoseOptimizer::save_parameter_to_file(const string & name)
{
	ofstream out(name);

	if (!out) {
		cout << "cannot open file " << name << endl;
		return;
	}
	int idx = 0;
	for (auto p : params) {
		out << p.value << " ";
		if (idx == 8) {
			out << endl;
		}
		idx++;
	}
	out << endl;
	out.close();
}

bool PoseOptimizer::load_parameter_from_file(const string & name)
{
	ifstream in(name);
	if (!in) {
		cout << "cannot open file " << name << endl;
		return false;
	}
	float v;
	for (size_t i = 0; i < params.size(); ++i) {
		in >> v;
		params[i].value = v;
	}
	in.close();

	scale = params[0].value;
	translate_x = params[1].value;
	translate_y = params[2].value;
	translate_z = params[3].value;
	rotate_x = params[4].value;
	rotate_y = params[5].value;
	rotate_z = params[6].value;
	camera_pos = params[7].value;
	camera_fov = params[8].value;
	for (int i = 9; i < 9 + alpha.size(); ++i) {
		alpha[i - 9] = params[i].value;
		bfm->alpha[i - 9] = params[i].value;
	}
	for (int i = 9 + alpha.size(); i < params.size(); ++i) {
		gamma[i - 9 - alpha.size()] = params[i].value;
		bfm->gamma[i - 9 - alpha.size()] = params[i].value;
	}
	return true;
}

bool PoseOptimizer::load_parameter_from_vector(const vector<float>& _params)
{
	for (size_t i = 0; i < params.size(); ++i) {
		params[i].value = _params[i];
	}
	
	scale = params[0].value;
	translate_x = params[1].value;
	translate_y = params[2].value;
	translate_z = params[3].value;
	rotate_x = params[4].value;
	rotate_y = params[5].value;
	rotate_z = params[6].value;
	camera_pos = params[7].value;
	camera_fov = params[8].value;
	for (int i = 9; i < 9 + alpha.size(); ++i) {
		alpha[i - 9] = params[i].value;
		bfm->alpha[i - 9] = params[i].value;
	}
	for (int i = 9 + alpha.size(); i < params.size(); ++i) {
		gamma[i - 9 - alpha.size()] = params[i].value;
		bfm->gamma[i - 9 - alpha.size()] = params[i].value;
	}
	return true;
}

vector<float> PoseOptimizer::get_params() const
{
	vector<float> res;
	for (size_t i = 0; i < params.size(); ++i) {
		res.push_back(params[i].value);
	}
	return res;
}
