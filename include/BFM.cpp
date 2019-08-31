#include "BFM.h"
#include "landmark.h"

template <typename T>
vector<T> load_util(FILE* stream)
{
	int len1, len2;
	size_t temp;
	temp = fread(&len1, sizeof(int), 1, stream);
	temp = fread(&len2, sizeof(int), 1, stream);
	int len = int(len1) * int(len2);


	T* res = new T[len];
	temp = fread(res, sizeof(T), len, stream);

	vector<T> r;
	for (int i = 0; i < len; ++i) {
		r.push_back(res[i]);
	}
	delete[] res;
	return r;

}

bool BFM::load(const string & name, const string& exp_name, const string attribute_name)
{
	FILE *stream;
	if ((stream = fopen(name.c_str(), "rb")) != NULL)
	{
		shapeEV = load_util<float>(stream);	
		shapeMU = load_util<float>(stream);
		shapePC = load_util<float>(stream);
		texEV = load_util<float>(stream);
		texMU = load_util<float>(stream);
		texPC = load_util<float>(stream);
		tl = load_util<int>(stream);
		segments = load_util<int>(stream);
		fclose(stream);
	}
	else {
		cout << "can not open bfm file: " << name << endl;
		return false;
	}
	if ((stream = fopen(exp_name.c_str(), "rb")) != NULL)
	{
		expEV = load_util<float>(stream);
		expMU = load_util<float>(stream);
		expPC = load_util<float>(stream);
		fclose(stream);
	}
	else {
		cout << "can not open bfm exp file: " << name << endl;
	}
	for (size_t i = 0; i < expEV.size(); ++i) {
		expEV[i] *= 0.02f;
	}
	
	n_parameter = shapeEV.size();
	n_vertex = shapeMU.size() / 3;
	n_triangle = tl.size() / 3;
	n_segments = segments.size();

	vertices.resize(n_vertex * 3);
	colors.resize(n_vertex * 3);
	
	normals.resize(n_vertex * 3);

	initial_mean();
	calculate_bbox();

	printf("[info]load bfm: n_parameter: %d, n_vertex: %d, n_triangle: %d\n", n_parameter, n_vertex, n_triangle);
	
	
	m_tex_pc.resize(n_vertex * 3, 199);
	
	for (int i = 0; i < n_vertex * 3; ++i) {
		for (int j = 0; j < 199; ++j) {
			m_tex_pc(i, j) = texPC[n_parameter * i + j];
		}
	}
	
	for (int i = 0; i < n_vertex * 3; ++i) {
		for (int j = 0; j < 99; ++j) {
			_texPC.push_back(m_tex_pc(i, j));
		}
	}
	bool ret = true;
	if (!attribute_name.empty()) {
		ret &= load_attribute(attribute_name);
	}
	return ret;
}
#define LOAD_ATTRIBUTE_PART(var) \
	for (int i = 0; i < 199; ++i) {float _t;in >> _t;var.push_back(_t);} in>>_b;

bool BFM::load_attribute(const string & name)
{
	ifstream in(name);
	if (!in) {
		cout << "cannot open " << name << endl;
		return false;
	}
	float _b;
	LOAD_ATTRIBUTE_PART(age_shape);
	LOAD_ATTRIBUTE_PART(age_tex);
	LOAD_ATTRIBUTE_PART(gender_shape);
	LOAD_ATTRIBUTE_PART(gender_tex);
	LOAD_ATTRIBUTE_PART(height_shape);
	LOAD_ATTRIBUTE_PART(height_tex);
	LOAD_ATTRIBUTE_PART(weight_shape);
	LOAD_ATTRIBUTE_PART(weight_tex);
	return true;
}

bool BFM::set_landmark_indices(const vector<int>& lm_idx)
{
	landmark_indices = lm_idx;
	return true;
}

bool BFM::initial_mean()
{
	alpha.clear();
	beta.clear();
	gamma.clear();
	
	for (int i = 0; i < n_parameter; ++i) {
		alpha.push_back(0);
		beta.push_back(0);
	}
	for (int i = 0; i < expEV.size(); ++i) {
		gamma.push_back(0);
	}
	update();
	return true;
}

bool BFM::initial_random()
{
	alpha.clear();
	beta.clear();
	gamma.clear();

	for (int i = 0; i < n_parameter; ++i) {
		alpha.push_back(random<float>(-1, 1));
		beta.push_back(random<float>(-1, 1));
	}
	for (int i = 0; i < expEV.size(); ++i) {
		gamma.push_back(random<float>(-1, 1));
	}
	update();
}


void BFM::update()
{
	for (size_t i = 0; i < n_vertex * 3; ++i) {
		vertices[i] = shapeMU[i];
	}

	for (size_t i = 0; i < n_vertex * 3; ++i) {
		colors[i] = texMU[i];
	}
	
	int len = texEV.size();
	// vertices and color
	for (int i = 0; i < alpha.size(); ++i) { 
		for (int j = 0; j < n_vertex * 3; ++j) {
			vertices[j] += alpha[i] * shapeEV[i] * shapePC[n_parameter * j + i];
			colors[j] += beta[i] * texEV[i] * texPC[n_parameter * j + i];
		}
	}	
	
	//exp
	for (int i = 0; i < expEV.size(); ++i) { 
		for (int j = 0; j < n_vertex * 3; ++j) {
			vertices[j] += gamma[i] * expEV[i] * expPC[expEV.size() * j + i];
		}
	}

}

void BFM::update_vertices()
{
	for (size_t i = 0; i < n_vertex * 3; ++i) {
		vertices[i] = shapeMU[i];
	}
	for (int i = 0; i < alpha.size(); ++i) { 
		for (int j = 0; j < n_vertex * 3; ++j) {
			vertices[j] += alpha[i] * shapeEV[i] * shapePC[n_parameter * j + i];
		}
	}
	for (int i = 0; i < expEV.size(); ++i) { 
		for (int j = 0; j < n_vertex * 3; ++j) {
			vertices[j] += gamma[i] * expEV[i] * expPC[expEV.size() * j + i];
		}
	}
}

void BFM::update_normals()
{
	int* indices = new int[n_triangle * 3];
	float* _vertices = new float[n_vertex * 3];
	float* _normals = new float[n_vertex * 3];

	update_vertices();
	for (int i = 0; i < n_triangle * 3; ++i) indices[i] = tl[i];
	for (int i = 0; i < n_vertex * 3; ++i) _vertices[i] = vertices[i];

	calculate_normals(indices, _vertices, n_triangle, _normals, false);

	for (int i = 0; i < n_vertex * 3; ++i) {
		normals[i] = _normals[i];
	}
	delete[] indices;
	delete[] _vertices;
	delete[] _normals;
}

void BFM::calculate_bbox()
{
	pMin = vec3(FloatMax, FloatMax, FloatMax);
	pMax = vec3(FloatMin, FloatMin, FloatMin);

	for (size_t i = 0; i < n_vertex; ++i) {
		pMin.x = std::min(pMin.x, vertices[3 * i]);
		pMax.x = std::max(pMax.x, vertices[3 * i]);

		pMin.y = std::min(pMin.y, vertices[3 * i + 1]);
		pMax.y = std::max(pMax.y, vertices[3 * i + 1]);

		pMin.z = std::min(pMin.z, vertices[3 * i + 2]);
		pMax.z = std::max(pMax.z, vertices[3 * i + 2]);
	}
	center = (pMin + pMax) * 0.5f;
}

vec3 BFM::get_vertex(int index, int n) const
{
	vec3 res;
	int index_1 = n_parameter * (3 * index);
	int index_2 = n_parameter * (3 * index + 1);
	int index_3 = n_parameter * (3 * index + 2);

	res.x = shapeMU[3 * index];
	res.y = shapeMU[3 * index + 1];
	res.z = shapeMU[3 * index + 2];

	if (n == -1 || n > alpha.size()) n = alpha.size();
	for (int i = 0; i < n; ++i) { //199
		float tmp = shapeEV[i] * alpha[i];
		res.x += tmp * shapePC[index_1 + i];
		res.y += tmp * shapePC[index_2 + i];
		res.z += tmp * shapePC[index_3 + i];
	}
	index_1 = gamma.size() * (3 * index);
	index_2 = gamma.size() * (3 * index + 1);
	index_3 = gamma.size() * (3 * index + 2);

	for (int i = 0; i < gamma.size(); ++i) { //29
		float tmp = expEV[i] * gamma[i];
		res.x += tmp * expPC[index_1 + i];
		res.y += tmp * expPC[index_2 + i];
		res.z += tmp * expPC[index_3 + i];
	}
	return res;
}

vec3 BFM::get_vertex(int face_idx, float u, float v, int n) const
{
	int idx_p1 = tl[3 * face_idx];
	int idx_p2 = tl[3 * face_idx + 1];
	int idx_p3 = tl[3 * face_idx + 2];

	vec3 c1 = get_vertex(idx_p1, n);
	vec3 c2 = get_vertex(idx_p2, n);
	vec3 c3 = get_vertex(idx_p3, n);

	return  (1 - u - v) * c1 + u * c2 + v * c3;
}

vec3 BFM::get_color(int idx, int n) const
{
	vec3 res(texMU[3 * idx], texMU[3 * idx + 1], texMU[3 * idx + 2]);
	int index_1 = 99 * (3 * idx);
	int index_2 = 99 * (3 * idx + 1);
	int index_3 = 99 * (3 * idx + 2);

	if (n == -1 || n > beta.size()) n = beta.size();

	for(int i = 0; i < n; ++i) { 
		float tmp = beta[i] * texEV[i];
		res[0] += tmp * _texPC[index_1 + i];
		res[1] += tmp * _texPC[index_2 + i];
		res[2] += tmp * _texPC[index_3 + i];
	}

	return res;
}

vec3 BFM::get_color(int face_idx, float u, float v, int n) const
{
	int idx_p1 = tl[3 * face_idx];
	int idx_p2 = tl[3 * face_idx + 1];
	int idx_p3 = tl[3 * face_idx + 2];

	vec3 c1 = get_color(idx_p1, n);
	vec3 c2 = get_color(idx_p2, n);
	vec3 c3 = get_color(idx_p3, n);
	
	return  (1 - u - v) * c1 + u * c2 + v * c3;
}

// 0: nose; 1: eye; 2: mouth; 3: others
int BFM::get_seg(int idx) const
{
	vec4i seg(segments[4 * idx], segments[4 * idx + 1], segments[4 * idx + 2], segments[4*idx+3]);
	int ret = -1;
	int sum = seg[0] + seg[1] + seg[2] + seg[3];
	
	for (int i = 0; i < 4; ++i) {
		if (seg[i] != 0) {
			ret = i;
			break;
		}
	}
	return ret; 
}

template<typename T>
int mode(const std::vector<T>& _data) {
	int max_counts = 0, cur_counts = 0;
	
	std::vector<T> data(_data.begin(), _data.end());
	std::sort(data.begin(), data.end());
	T res = -1;
	T prev = data[0];
	for (size_t i = 1; i < data.size();++i) {
		if (data[i] == prev) {
			cur_counts++;
		}
		else {
			cur_counts = 1;
			prev = data[i];
		}
		if (cur_counts > max_counts) {
			max_counts = cur_counts;
			res = data[i];
		}
	}
	return res;
}
int BFM::get_seg(int face_idx, float u, float v) const
{
	int idx_p1 = tl[3 * face_idx];
	int idx_p2 = tl[3 * face_idx + 1];
	int idx_p3 = tl[3 * face_idx + 2];

	int c1 = get_seg(idx_p1);
	int c2 = get_seg(idx_p2);
	int c3 = get_seg(idx_p3);
							
	return  mode(vector<int>{c1,c2,c3});
}

vec3 BFM::get_normal(int idx) const
{
	vec3 normal(normals[3 * idx], normals[3 * idx + 1], normals[3 * idx + 2]);
	return normal;
}

vec3 BFM::get_normal(int face_idx, float u, float v) const
{
	int idx_p1 = tl[3 * face_idx];
	int idx_p2 = tl[3 * face_idx + 1];
	int idx_p3 = tl[3 * face_idx + 2];

	vec3 c1 = get_normal(idx_p1);
	vec3 c2 = get_normal(idx_p2);
	vec3 c3 = get_normal(idx_p3);
	
	c1 = normalize_normal(c1);
	c2 = normalize_normal(c2);
	c3 = normalize_normal(c3);

	return  (1 - u - v) * c1 + u * c2 + v * c3;
}

bool BFM::write_to_ply(const string & name, bool inverse_y)
{
	ofstream out(name);
	if (!out) return false;

	out << "ply\nformat ascii 1.0\n";
	out << "element vertex " << n_vertex << "\n";
	out << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
	out << "element face " << n_triangle << "\n";
	out << "property list uchar int vertex_index\nend_header\n";

	vector<float> c;
	for (auto i : colors) {
		c.push_back(i);
	}
	for (int i = 0; i < landmark_indices.size(); ++i) {
		int idx = landmark_indices[i];
		c[idx * 3] = 255.0;
		c[idx * 3 + 1] = 0.0;
		c[idx * 3 + 2] = 0.0;
	}

	for (int i = 0; i < n_vertex; ++i) {
		if(inverse_y) {
			out << std::fixed << std::setprecision(8) << vertices[i * 3] << " " << 250 - vertices[i * 3 + 1] << " " << vertices[i * 3 + 2] << " "
				<< int(c[i * 3]) << " " << int(c[i * 3 + 1]) << " " << int(c[i * 3 + 2]) << endl;
		}
		else {
			
			out << std::fixed << std::setprecision(8) << vertices[i * 3] << " " << vertices[i * 3 + 1] << " " << vertices[i * 3 + 2] << " "
				<< int(c[i * 3]) << " " << int(c[i * 3 + 1]) << " " << int(c[i * 3 + 2]) << endl;
		}
	}

	for (int i = 0; i < n_triangle; ++i) {
		out << "3 " << tl[i * 3] << " " << tl[i * 3 + 1] << " " << tl[i * 3 + 2] << endl;
	}
	out.close();
	return true;
}
map<int ,set<int> > BFM::get_neighbors() {
	map<int, set<int> > res;
	for (int i = 0; i < n_triangle; ++i) {
		int idx1 = tl[3 * i];
		int idx2 = tl[3 * i + 1];
		int idx3 = tl[3 * i + 2];
		res[idx1].insert(idx2);
		res[idx1].insert(idx3);
		res[idx2].insert(idx1);
		res[idx2].insert(idx3);
		res[idx3].insert(idx1);
		res[idx3].insert(idx2);
	}
	return res;
}

void BFM::update_by_pose_optimize(std::function<vec3(const vec3&)> transform)

{
	for (int i = 0; i < n_vertex; ++i) {
		vec3 _p = vec3(vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]);
		_p = transform(_p);
		vertices[3 * i] = _p.x;
		vertices[3 * i + 1] = _p.y;
		vertices[3 * i + 2] = _p.z;
	}
}
void BFM::deformation(const Landmarks3D & lm3d, float weight)
{
	int n = n_vertex;
	int k = lm3d.landmarks_indices.size();
	SparseMatrix<double> A(n + k, n);
	cout << A.rows() <<" " <<A.cols() << endl;

	MatrixXd b(n + k, 3);
	MatrixXd vpos(n, 3);
	for (int i = 0; i < n_vertex; ++i) {
		vpos(i, 0) = vertices[3 * i];
		vpos(i, 1) = vertices[3 * i + 1];
		vpos(i, 2) = vertices[3 * i + 2];
	}
	cout << "fill A" << endl;
	/*fill A*/
	vector<int> I;
	vector<int> J;
	vector<double> V;
	map<int, set<int> > all_neighbors = get_neighbors();
	for (int i = 0; i < n; ++i) {
		set<int>& neighbor_indices = all_neighbors[i];
		int z = neighbor_indices.size();
		for (int j = 0; j < z + 1; ++j) 
			I.push_back(i);
		for (auto _idx : neighbor_indices) 
			J.push_back(_idx);
		J.push_back(i);
		for (int j = 0; j < z; ++j) V.push_back(-1);
		V.push_back(z);
	}

	for (int i = 0; i < k; ++i) {
		I.push_back(n + i);
		J.push_back(lm3d.landmarks_indices[i]);
		V.push_back(weight);
	}
	int idx = lm3d.landmarks_indices[60];
	

	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	tripletList.reserve(I.size());

	for (int idx = 0; idx < I.size(); ++idx) {
		
		tripletList.push_back(T(I[idx], J[idx], V[idx]));
	}
	A.setFromTriplets(tripletList.begin(), tripletList.end());

	/*fill b, split x,y,z*/
	b = A * vpos;
	for (int i = 0; i < k; ++i) {
		b(n + i, 0) = weight * lm3d.landmarks_obj[i].x;
		b(n + i, 1) = weight * lm3d.landmarks_obj[i].y;
		b(n + i, 2) = weight * lm3d.landmarks_obj[i].z;
	}
	VectorXd x(n, 1);
	VectorXd guess[3];
	for (int j = 0; j < 3; j++) {
		guess[j] = VectorXd(n, 1);
		for (int i = 0; i < n; ++i) {
			guess[j](i, 0) = vertices[3 * i + j];
		}
	}

	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;

	Eigen::SparseMatrix<double> _A = A.transpose() * A;

	solver.compute(_A);

	if (solver.info() != Eigen::Success)
	{
		std::cerr << "Waring: Eigen decomposition failed" << std::endl;
	}
	//solver.setMaxIterations(1000);
	solver.setTolerance(8e-8);
	for (int i = 0; i < 2; ++i) {
		
		Eigen::MatrixXd _b = A.transpose() * b.col(i);

		x = solver.solveWithGuess(_b, guess[i]);
	
		for (int j = 0; j < n_vertex; ++j) {
			vertices[3 * j + i] = x(j, 0);
		}
	}
	
}

void BFM::applyAttributes(Attribute attrib, float factor)
{
	vector<float>* attrib_shape = nullptr, *attrib_tex = nullptr;
	switch (attrib)
	{
	case BFM::Attribute::AGE:
		attrib_shape = &age_shape;
		attrib_tex = &age_tex;
		break;
	case BFM::Attribute::GENDER:
		attrib_shape = &gender_shape;
		attrib_tex = &gender_tex;
		break;
	case BFM::Attribute::HEIGHT:
		attrib_shape = &height_shape;
		attrib_tex = &height_tex;
		break;
	case BFM::Attribute::WEIGHT:
		attrib_shape = &weight_shape;
		attrib_tex = &weight_tex;
		break;
	default:
		return;
	}

	for (int i = 0; i < alpha.size(); ++i) {
	//	alpha[i] += factor * (*attrib_shape)[i];
		beta[i] += factor * (*attrib_tex)[i];
	}
	update();
}

void BFM::update_color_by_uvmap(const Texture & uvmap)
{
	float aspect = std::max(std::max(pMax.x - center.x, pMax.y - center.y), pMax.z - center.z);

	for (int i = 0; i < n_vertex; ++i) {
		vec3 pos(vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]);
		pos = (pos - center) / aspect; // [-1,1] x [-1,1] x [-1,1]
		vec2 uv = pos2uv(pos); 
		uv = (uv + vec2(1, 1)) / 2.0f;
		
		uv.x = Clamp<float>(uv.x, 0.0, 1.0f);
		uv.y = Clamp<float>(uv.y, 0.0, 1.0f);

		vec2i coord = vec2i(uv.x * uvmap.width, uvmap.height - 1 - uv.y * uvmap.height);
		vec3i color = uvmap.getPixel(coord) * 255.0;
		colors[3 * i] = color.R;
		colors[3 * i + 1] = color.G;
		colors[3 * i + 2] = color.B;
	}
}

vec2 pos2uv(const vec3 & v)
{
	float x = 0.0f;
	if (v.z < 0 && v.x < 0) {
		x = -atan(v.z / v.x) - PI / 2.0f;
	}
	if (v.z >= 0 && v.x < 0) {
		x = -PI / 2.0f + atan(-v.z / v.x);
	}
	if (v.x == 0) {
		x == 0;
	}
	if (v.z >= 0 && v.x > 0) {
		x = PI / 2.0f - atan(v.z / v.x);
	}
	if (v.z < 0 && v.x > 0) {
		x = atan(-v.z / v.x) + PI / 2.0f;
	}
	x = x / PI;
	float y = v.y /1.1f;
	return vec2(x, y);
}

