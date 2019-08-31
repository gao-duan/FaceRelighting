#include "tex_light_fit.h"
#include "color_propagation.h"

#define BETA_SIZE 99
vec3 SH(const vector<vec3>& l, const vec3 & N) {
	float c1 = 1.0 / sqrt(4.0 * PI);
	float c2 = 2.0*PI / 3.0 * sqrt(3.0 / (4.0 * PI));
	float c3 = PI / 8.0 * sqrt(5.0 / (4.0 * PI));
	float c4 = 3.0 * PI / 4.0 * sqrt(5.0 / (12.0 * PI));
	float c5 = 3.0 * PI / 8.0 * sqrt(5.0 / (12.0 * PI));

	
	float h1 = c1;
	float h2 = c2 * N.z;
	float h3 = c2 * N.y;
	float h4 = c2 * N.x;
	float h5 = c3 * (3.0 * N.z * N.z - 1.0);
	float h6 = c4 * N.y * N.z;
	float h7 = c4 * N.x * N.z;
	float h8 = c4 * N.x * N.y;
	float h9 = c5 * (N.x * N.x - N.y * N.y);
	
	vec3 I1 = Clamp(h1 * l[0]);
	vec3 I2 = Clamp(h2 * l[1]);
	vec3 I3 = Clamp(h3 * l[2]);
	vec3 I4 = Clamp(h4 * l[3]);
	vec3 I5 = Clamp(h5 * l[4]);
	vec3 I6 = Clamp(h6 * l[5]);
	vec3 I7 = Clamp(h7 * l[6]);
	vec3 I8 = Clamp(h8 * l[7]);
	vec3 I9 = Clamp(h9 * l[8]);

	return I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9;
}
vec3 Blingphong(vec3 normal, vec3 dir, vec3 color) {
	vec3 view(0, 0, 1);
	vec3 H = (dir + view);
	H.normalize();
	float glossy = 8;
	float specular = pow(max(0.0f, Dot(normal, H)), glossy);
	float diffuse = max(0.0f, Dot(normal, dir));
	float ambient = 0.4;
	return color * (ambient + 0.7 * diffuse + 0.001 * specular);
}

TexLightOptimizer::TexLightOptimizer(BFM * _bfm, const vector<vec2i>& _valid_points,
	const vector<FacePoint>& _face_points, const vector<vec3>& _normals, const Texture & _ref_img,
	const Texture& _depth_img, const Texture& _matte_img, int nt)
	: bfm(_bfm), valid_points(_valid_points), face_points(_face_points),
	normals(_normals), ref_img(_ref_img), params_num(BETA_SIZE + 9 * 3),

	constraits_num(_valid_points.size() / 10), depth_img(_depth_img), matte_img(_matte_img), n_iteration(nt) {
	n_beta = BETA_SIZE;
	params.resize(params_num);
	for (int i = n_beta; i < n_beta + 3; ++i) {
		params[i].value = 0.5f;
	}
	for (int i = n_beta + 3; i < params_num; ++i) {
		params[i].value = 0.0f;
	}

	jaccobian = MatrixXd(constraits_num, params_num);
	R = VectorXd(constraits_num);
	lr = 0.5;
	lambda = 0.5;
	beta.resize(n_beta);
	sh.resize(9);

	for (auto p : valid_points) {
		vec3 c = ref_img.getPixel(p.x, p.y);
		float lumiance = (c[0] + c[1] + c[2]) / 3.0;
		if (lumiance > 0.8 || lumiance < 0.4) {
			valids.push_back(true);
		}
		else valids.push_back(true);
	}
	_albedo = Texture(250, 250);
	seg = Texture(250, 250);
	for (size_t i = 0; i < valid_points.size(); ++i) {
		vec2i p = valid_points[i];
		FacePoint fp = face_points[i];
		int _seg;
		if (fp.face_idx == -1) {
			_seg = 3;
		}
		else {
			_seg = bfm->get_seg(fp.face_idx, fp.u, fp.v);
		}
		vec3 color;
		switch (_seg)
		{
			case 0:color = vec3(0, 0, 0); break;
			case 1:color = vec3(1, 0, 0); break;
			case 2:color = vec3(2, 0, 0); break;
			case 3:color = vec3(3, 0, 0); break;
			default:
				break;
		}
		seg.setPixel(p.x, p.y, color);
	}
}

float TexLightOptimizer::get_loss(bool flag)
{
	float loss = 0.0f;
	vector<float> ts = f_col(params, flag);
	for (int i = 0; i < constraits_num; ++i) {
		float t = ts[i] * std::sqrt(10.0 / constraits_num);
		loss += t * t;
	}
	return loss;
}

float TexLightOptimizer::f(const std::vector<Parameter>& p, int idx)
{
	idx *= 10;

	// 1. update bfm color
	for (int k = 0; k < n_beta; ++k) {
		beta[k] = p[k].value;
		bfm->beta[k] = p[k].value;
	}
	// 2. get valid_points albedo
	vec2i pos = valid_points[idx];
	FacePoint fp = face_points[idx];
	vec3 albedo;
	if (fp.face_idx != -1)
		albedo = bfm->get_color(fp.face_idx, fp.u, fp.v,n_beta);
	else {
		albedo = vec3(1, 0, 0);
	}

	vec3 _tmp = ref_img.getPixel(pos.x, pos.y);
	vec3 ref_color = vec3(_tmp[0], _tmp[1], _tmp[2]); // rgb

	// 3. use sh to calculate light(needs normal of pixels)
	for (int i = 0; i < sh.size(); ++i) {
		sh[i] = vec3(p[n_beta + 3 * i].value, p[n_beta + 3 * i + 1].value, p[n_beta + 3 * i + 2].value);
	}
	vec3 light = SH(sh, normals[idx]);
	vec3 fit_color = vec3(albedo.x * light.x, albedo.y * light.y, albedo.z * light.z);

	float w = 1.0f;

	return (ref_color - fit_color / 255.0).length() * w;
}

vector<float> TexLightOptimizer::f_col(const std::vector<Parameter>& p,bool flag)
{
	vector<float> res(constraits_num);

	//1. update bfm color and sh
	for (int k = 0; k < n_beta; ++k) {
		beta[k] = p[k].value;
		bfm->beta[k] = p[k].value;
	}

	for (int i = 0; i < sh.size(); ++i) {
		sh[i] = vec3(p[n_beta + 3 * i].value, p[n_beta + 3 * i + 1].value, p[n_beta + 3 * i + 2].value);
	}

	int width = ref_img.width, height = ref_img.height;
	int tex2idx[250][250];
	memset(tex2idx, 0, 250 * 250 * sizeof(int));
	Texture fit_img(width, height);
	vector<vec3> ref_color(constraits_num), fit_color(constraits_num);

	for (size_t idx = 0; idx < valid_points.size(); idx += 10) {
		tex2idx[valid_points[idx].x][valid_points[idx].y] = idx;
		if (idx / 10 > constraits_num - 1) continue;
	
		// 2. get valid_points albedo
		vec2i pos = valid_points[idx];
		FacePoint fp = face_points[idx];
		vec3 albedo;
		if(flag) albedo= (fp.face_idx == -1) ? vec3(1, 0, 0) : bfm->get_color(fp.face_idx, fp.u, fp.v, n_beta)/255.0;
		else {
			albedo = _albedo.getPixel(pos);
		}
		// 3. get ref color
		vec3 _tmp = ref_img.getPixel(pos.x, pos.y);
		vec3 _ref_color = vec3(_tmp[0], _tmp[1], _tmp[2]); // rgb

		
		// 4. get fit color
		vec3 light = SH(sh, normals[idx]);
		vec3 _fit_color = vec3(albedo.x * light.x, albedo.y * light.y, albedo.z * light.z);
		if (flag) _fit_color = albedo;

		ref_color[idx / 10] = _ref_color;
		fit_img.setPixel(pos.x, pos.y, _fit_color);
		fit_color[idx / 10] = _fit_color;

	}

	// 5. blur fit_color
	//float sssWidth = 0.025 / 20;
	//fit_img = UnionTexture(ref_img, fit_img, matte_img);
	//Texture blur_img = Blur(fit_img, depth_img, sssWidth, matte_img);

	//for (int i = 0; i < width; ++i) {
	//	for (int j = 0; j < height; ++j) {
	//		int idx = tex2idx[i][j];
	//		fit_color[idx] = blur_img.getPixel(i, j);
	//	}
	//}
	// 6. convert fit_color to vector<vec3> and get loss
	for (size_t i = 0; i < constraits_num; ++i) {
		res[i] = (ref_color[i] - fit_color[i]).length();
		
	}
	return res;
}

VectorXd TexLightOptimizer::get_grad(int i)
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

void TexLightOptimizer::get_jaccobian(bool flag)
{
	vector<float> old_loss(constraits_num), new_loss;
	float delta = 1e-5;
	old_loss = f_col(params, flag);

	for (int j = 0; j < params_num; ++j) {
		std::vector<Parameter> new_params(params);

		new_params[j].update(delta);

		new_loss = f_col(new_params, flag); // update parameter, get loss for every pixel
		for (int i = 0; i < constraits_num; ++i) {
			jaccobian(i, j) = (new_loss[i] - old_loss[i]) / delta;
		
		}
	}
	jaccobian *= sqrt(10.0f / constraits_num);
}

void TexLightOptimizer::get_r(bool flag)
{
	vector<float> res = f_col(params, flag);
	for (int i = 0; i < constraits_num; ++i) {
		R(i, 0) = double(res[i]) * sqrt(10.0f / constraits_num);
	}
}

void TexLightOptimizer::update_parameter(const VectorXd & delta)
{
	for (int i = 0; i < params_num; ++i) {
		params[i].update(delta(i, 0));
	}
}

void TexLightOptimizer::show_result(const std::string& dir, bool flag)
{
	Texture res = render_by_parameters(sh, flag);

	res.save(dir + to_string(cur_iter) + ".bmp");
}

void TexLightOptimizer::optimize()
{
	VectorXd update_params(params_num);

	for (int i = 0; i < n_beta; ++i) {
		params[i].lr = 10.0;
	}

	for (int i = n_beta; i < params.size(); ++i) {
		params[i].lr = 0;
	}

	bool flag = true;
	for (int iter = 0; iter < n_iteration; ++iter) {
		if (iter == 4) {
			for (int i = 0; i < n_beta; ++i) {
				params[i].lr = 0.0;
			}

			for (int i = n_beta; i < params.size(); ++i) {
				params[i].lr = 1.0;
			}
			flag = false;
			for (size_t idx = 0; idx < valid_points.size(); ++idx) {
				vec2i p = valid_points[idx];
				FacePoint fp = face_points[idx];
				vec3 a = (fp.face_idx == -1) ? vec3(0, 0, 0) : bfm->get_color(fp.face_idx, fp.u, fp.v, n_beta) / 255.0;
				_albedo.setPixel(p.x, p.y, a);
			}
		}
		get_jaccobian(flag);
		get_r(flag);
		cur_iter = iter;

		MatrixXd jt = jaccobian.transpose();
		MatrixXd jtj = jt * jaccobian + lambda * Eigen::MatrixXd::Identity(params_num, params_num);

		MatrixXd jtj_inv = jtj.inverse();
		MatrixXd jtr = jt * R;
		MatrixXd res = jtj_inv * jtr;
		update_params = -lr * res;

		update_parameter(update_params);

	//	show_result(res_path, flag);
	}
}

void TexLightOptimizer::save_parameter_to_file(const string & name)
{
	ofstream out(name);

	if (!out) {
		cout << "cannot open file " << name << endl;
		return;
	}
	int idx = 0;
	for (auto p : params) {
		out << p.value << " ";
		if (idx == 10) {
			out << endl;
		}
		idx++;
	}
	out << endl;
	out.close();
}

void TexLightOptimizer::load_parameter_from_file(const string & name)
{
	ifstream in(name);
	if (!in) {
		cout << "cannot open file " << name << endl;
		return;
	}
	float v;
	for (size_t i = 0; i < params.size(); ++i) {
		in >> v;
		params[i].value = v;
	}
	in.close();

	for (int k = 0; k < n_beta; ++k) {
		beta[k] = params[k].value;
		bfm->beta[k] = params[k].value;
	}

	for (int i = 0; i < sh.size(); ++i) {
		sh[i] = vec3(params[n_beta + 3 * i].value, params[n_beta + 3 * i + 1].value, params[n_beta + 3 * i + 2].value);
	}

}

void TexLightOptimizer::load_parameter_from_vector(const vector<float>& _params)
{
	for (size_t i = 0; i < params.size(); ++i) {
		params[i].value = _params[i];
	}

	for (int k = 0; k < n_beta; ++k) {
		beta[k] = params[k].value;
		bfm->beta[k] = params[k].value;
	}

	for (int i = 0; i < sh.size(); ++i) {
		sh[i] = vec3(params[n_beta + 3 * i].value, params[n_beta + 3 * i + 1].value, params[n_beta + 3 * i + 2].value);
		cout << sh[i] << endl;
	}

}

Texture TexLightOptimizer::render_by_parameters(const std::vector<vec3>& _sh, bool flag) {
	Texture res(250, 250);

	for (size_t idx = 0; idx < valid_points.size(); ++idx) {
		vec2i pos = valid_points[idx];
		FacePoint fp = face_points[idx];
		vec3 albedo = (fp.face_idx == -1) ? vec3(0, 0, 0) : bfm->get_color(fp.face_idx, fp.u, fp.v, n_beta);
		vec3 light = SH(_sh, normals[idx]);
		vec3 _fit_color = vec3(albedo.x * light.x, albedo.y * light.y, albedo.z * light.z) / 255.0;
		if (flag) _fit_color = albedo / 255.0f;
		
		res.setPixel(pos.x, pos.y, _fit_color);
	}

	res = UnionTexture(ref_img, res, matte_img);
	return res;
}

Texture TexLightOptimizer::render_by_parameters(const vec3 & dir, const vec3 & color)
{
	Texture res(250, 250);

	for (size_t idx = 0; idx < valid_points.size(); ++idx) {
		vec2i pos = valid_points[idx];
		FacePoint fp = face_points[idx];
		vec3 albedo = (fp.face_idx == -1) ? vec3(0, 0, 0) : bfm->get_color(fp.face_idx, fp.u, fp.v, n_beta);
		vec3 light = Blingphong(dir, color, normals[idx]);

		vec3 _fit_color = vec3(albedo.x * light.x, albedo.y * light.y, albedo.z * light.z) / 255.0;
		res.setPixel(pos.x, pos.y, _fit_color);
	}

	res = UnionTexture(ref_img, res, matte_img);
	return res;
}

void blur(Texture& img, vec2i beg, vec2i end) {
	int width = end.x - beg.x;
	int height = end.y - beg.y;
	Texture slice(width, height);

	if (beg.x < 0 || beg.y < 0 || end.x > img.width || end.y > img.height) {
		return;
	}

	for (int x = 0; x < width; ++x) {
		for (int y = 0; y < height; ++y) {
			slice.setPixel(x, y, img.getPixel(beg.x + x, beg.y + y));
		}
	}
	Mat res;
	cv::GaussianBlur(slice.convert2opencv(), res, cv::Size(3, 3), 0, 0);

	slice = Texture(res);

	for (int x = 0; x < width; ++x) {
		for (int y = 0; y < height; ++y) {
			img.setPixel(beg.x + x, beg.y + y, slice.getPixel(x, y));
		}
	}
}

Texture TexLightOptimizer::generate_final_texture()
{
	Texture albedo(250, 250);
	Texture orig(250, 250);

	Texture res_fit = render_by_parameters(sh);
	
	for (size_t idx = 0; idx < valid_points.size(); ++idx) {
		vec2i p = valid_points[idx];
		FacePoint fp = face_points[idx];
		vec3 a = (fp.face_idx == -1) ? vec3(0, 0, 0) : bfm->get_color(fp.face_idx, fp.u, fp.v, n_beta) / 255.0;
		vec3 c1 = ref_img.getPixel(p);
		vec3 c3 = res_fit.getPixel(p);
		albedo.setPixel(p.x, p.y, c1 / c3 * a);
	}

	return albedo;
}

Texture TexLightOptimizer::generate_original_texture()
{
	Texture orig(250, 250);
	for (size_t idx = 0; idx < valid_points.size(); ++idx) {
		vec2i p = valid_points[idx];
		FacePoint fp = face_points[idx];
		vec3 a = (fp.face_idx == -1) ? vec3(0, 0, 0) : bfm->get_color(fp.face_idx, fp.u, fp.v, n_beta) / 255.0;
		orig.setPixel(p.x, p.y, a);
	}
	return orig;
}

Texture TexLightOptimizer::generate_optmize_image()
{
	return render_by_parameters(sh);
}


void TexLightOptimizer::relighting(const std::vector<vec3>& new_sh, const Texture& albedo_img)
{
	Texture orig = render_by_parameters(new_sh);
	orig.save(res_path + "relit_orig.bmp");
	orig.saveHDR(res_path + "relit_orig.exr");

	Texture tmp(250, 250);
	float lumiance = 0;
	for (size_t idx = 0; idx < valid_points.size(); ++idx) {
		FacePoint fp = face_points[idx];
		vec3 albedo = (fp.face_idx == -1) ? vec3(0, 0, 0) : bfm->get_color(fp.face_idx, fp.u, fp.v, n_beta) / 255.0;
		vec3 color = albedo_img.getPixel(valid_points[idx]) * albedo;
		tmp.setPixel(valid_points[idx].x, valid_points[idx].y, color);
		lumiance = lumiance + (color.R + color.G + color.B) / 3.0;
	}
	tmp.save(res_path + "relit_albedo_color1.bmp");

	lumiance = lumiance / valid_points.size();
	
	for (size_t idx = 0; idx < valid_points.size(); ++idx) {
		vec3 color = tmp.getPixel(valid_points[idx]);
		vec3 l = (color.R + color.G + color.B);
		color = color / l;
		tmp.setPixel(valid_points[idx].x, valid_points[idx].y, color);
	}
	tmp.save(res_path + "relit_albedo_color2.bmp");


	Texture res(250, 250);

	for (size_t idx = 0; idx < valid_points.size(); ++idx) {
		res.setPixel(valid_points[idx].x, valid_points[idx].y, orig.getPixel(valid_points[idx]) * albedo_img.getPixel(valid_points[idx]));
	}
	res = UnionTexture(ref_img, res, matte_img);

	res.save(res_path + "relit.bmp");
	res.saveHDR(res_path + "relit_hdr.exr");

}

void TexLightOptimizer::relighting_video(const Texture & albedo_img, bool prop)
{
	const int N = 20;

	vec3 light_dir(-1, 0, 0);
	vec3 light_color(1.2, 1.2, 1.2);

	float step = 180.0 / N;
	for (int i = 0; i <= N; ++i) {
		vec3 dir = Rotate(0, -step * i, 0) * light_dir;
		Texture res(250, 250);

		for (size_t idx = 0; idx < valid_points.size(); ++idx) {
			vec2i pos = valid_points[idx];
			vec3 light = Blingphong(normals[idx], dir, light_color);
			res.setPixel(pos.x, pos.y, light);
		}
		
		res = UnionTexture(ref_img, res, matte_img);

		for (size_t idx = 0; idx < valid_points.size(); ++idx) {
			res.setPixel(valid_points[idx].x, valid_points[idx].y, res.getPixel(valid_points[idx]) * albedo_img.getPixel(valid_points[idx]));
		}
		
		res.saveHDR(res_path + "relit/relight_hdr_" + to_string(i) + ".exr");
		res.save(res_path + "relit/relight_hdr_" + to_string(i) + ".bmp");

		if (prop) {
			ColorPropagaton solver(ref_img, res, matte_img, valid_points);
			solver.matte_img = matte_img;
			res = solver.solve(50.0);
			res.save(res_path + "prog/" + to_string(i) + ".bmp");
		}

	}
}

vector<float> TexLightOptimizer::get_params() const
{
	vector<float> res;
	for (size_t i = 0; i < params.size(); ++i) {
		res.push_back(params[i].value);
	}
	return res;
}