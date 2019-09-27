#include "landmark.h"
//#include "obj.h"

std::vector<vec2i> get_valid_points_from_obj(const string & name)
{
	ifstream in(name);
	std::vector<vec2i> points;

	string line;
	while (getline(in, line)) {
		stringstream ss(line);
		string type;
		ss >> type;
		if (type == "v") {
			int x, y; float z;
			ss >> x >> y >> z;
			points.push_back(vec2i(x, IMAGE_SIZE - y));
		}
	}

	return points;
}

std::vector<vec2i> get_valid_points_from_matte(const Texture & img)
{
	std::vector<vec2i> points;

	for (int x = 0; x < img.width; ++x) {
		for (int y = 0; y < img.height; ++y) {
			if (img.getPixel(x, y).R != 0) {
				points.push_back(vec2i(x, y));
			}
		}
	}

	return points;
}

std::vector<vec3> get_points_from_obj(const string & name)
{
	ifstream in(name);
	std::vector<vec3> points;

	string line;
	while (getline(in, line)) {
		stringstream ss(line);
		string type;
		ss >> type;
		if (type == "v") {
			int x, y; float z;
			ss >> x >> y >> z;
			points.push_back(vec3(x, IMAGE_SIZE - y, z));
		}
	}

	return points;
}

std::vector<int> get_invalids_normal(const string & name) {
	std::vector<float> vs;
	std::vector<int> is;

	ifstream in(name);
	string line;
	while (getline(in, line)) {
		stringstream s(line);
		string type;
		s >> type;
		if (type == "v") {
			float x, y, z;
			s >> x >> y >> z;
			vs.push_back(x); vs.push_back(y); vs.push_back(z);
		}
		else if (type == "f") {
			int v1, v2, v3;
			s >> v1 >> v2 >> v3;
			is.push_back(v1 - 1); is.push_back(v3 - 1); is.push_back(v2 - 1);
		}
	}

	// remove triangles inside nose
	float avg_z = 0;
	for (int i = 2; i < vs.size(); i += 3) {
	avg_z += vs[i];
	}
	avg_z /= vs.size() / 3;

	ofstream out("stat_p_a.txt");
	vector<int> _is;
	vector<int> invalid;
	for (int i = 0; i < is.size() / 3; ++i) {
		int idx1 = is[3 * i];
		int idx2 = is[3 * i + 1];
		int idx3 = is[3 * i + 2];
		vec3 p1(vs[3 * idx1], vs[3 * idx1 + 1], vs[3 * idx1 + 2]);
		vec3 p2(vs[3 * idx2], vs[3 * idx2 + 1], vs[3 * idx2 + 2]);
		vec3 p3(vs[3 * idx3], vs[3 * idx3 + 1], vs[3 * idx3 + 2]);

		float _avg_z = (p1.z + p2.z + p3.z) / 3;
		float area = Cross(p2 - p1, p3 - p1).length() * 0.5f;
		if (_avg_z > avg_z && area > 5) {
			invalid.push_back(idx1);
			invalid.push_back(idx2);
			invalid.push_back(idx3);
			continue;
		}
		else {
			_is.push_back(is[3 * i]);
			_is.push_back(is[3 * i + 1]);
			_is.push_back(is[3 * i + 2]);
		}
		out << area << endl;
	}
	out.close();
	//write_to_ply("remove_p1.ply", vs, _is, {});
	return invalid;

}
std::vector<vec3> correct_normals(const vector<vec2i>& valid_points, const vector<vec3>& normals_bfm, vector<vec3>& normals, const string& name)
{
	normals.resize(valid_points.size());

	set<vec2i> invalids; // invalid pixels coordinates

	vector<int> invalid_normals; // invalid normals indices
	for (int i = 0; i < valid_points.size(); ++i) {
		vec2i p = valid_points[i];
		if (invalids.find(p) != invalids.end()) {
			invalid_normals.push_back(i);
		}
	}

	// 1. convert invalid normal to bfm normal
	for (int i = 0; i < invalid_normals.size(); ++i) {
		normals[invalid_normals[i]] = normals_bfm[invalid_normals[i]];
	}
	
	// [visualization] normal_map initial
	Texture normal_map(IMAGE_SIZE, IMAGE_SIZE);
	for (size_t i = 0; i < valid_points.size(); ++i) {
		vec2i p = valid_points[i];
		vec3 c = normals[i];
		normal_map.setPixel(p.x, p.y, c);
	}

	// blur normals in invalid area
	for (int i = 0; i < invalid_normals.size(); ++i) {
		int idx = invalid_normals[i];
		vec2i p = valid_points[idx];
		vec3 c = normal_map.getPixel(p.x, p.y);
		c = c + normal_map.getPixel(p.x - 1, p.y);
		c = c + normal_map.getPixel(p.x + 1, p.y);
		c = c + normal_map.getPixel(p.x, p.y - 1);
		c = c + normal_map.getPixel(p.x, p.y + 1);
		c = c / 5;
		normal_map.setPixel(p.x, p.y, c);
		normals[invalid_normals[i]] = c;
	}
	for (int i = 0; i < invalid_normals.size(); ++i) {
		int idx = invalid_normals[i];
		vec2i p = valid_points[idx];
		vec3 _c = normal_map.getPixel(p.x, p.y);
		float c = normal_map.getPixel(p.x, p.y).G;
		c = c + normal_map.getPixel(p.x - 1, p.y).G;
		c = c + normal_map.getPixel(p.x + 1, p.y).G;
		c = c + normal_map.getPixel(p.x, p.y - 1).G;
		c = c + normal_map.getPixel(p.x, p.y + 1).G;
		c = c / 5;
		normal_map.setPixel(p.x, p.y, vec3(_c.R, c, _c.B));
		normals[invalid_normals[i]] = vec3(_c.R, c, _c.B);
	}

	// [visualization] convert normal value from [-1,1] to [0,1]
	for (size_t i = 0; i < valid_points.size(); ++i) {
		vec3 c = normal_map.getPixel(valid_points[i]);
		c[0] = (normals[i].x + 1.0) / 2.0;
		c[1] = (normals[i].y + 1.0) / 2.0;
		c[2] = (normals[i].z + 1.0) / 2.0;
		normal_map.setPixel(valid_points[i].x, valid_points[i].y, c);
	}
	normal_map.save(name);
}
std::vector<vec3> get_normal_from_normalmap(const Texture & normal_map, const vector<vec2i>& valid_points)
{
	vector<vec3> normals;
	for (int i = 0; i < valid_points.size(); ++i) {
		vec2i p = valid_points[i];
		vec3 c = normal_map.getPixel(p.x, p.y);
		c.x =  2.0 * c.x - 1;
		c.y = 2.0 * c.y - 1;
		c.z = 2.0 * c.z - 1;
		normals.push_back(c);
	}
	return normals;
}
Texture generate_matte_from_landmarks(std::vector<cv::Point>& points)
{
	std::vector<cv::Point> left_eye_points(points.begin() + 36, points.begin() + 42);
	std::vector<cv::Point> right_eye_points(points.begin() + 42, points.begin() + 48);
	std::vector<cv::Point> mouth_points(points.begin() + 60, points.begin() + 68);

	vector<vector<cv::Point> >  hull(4);
	Mat drawing = Mat::zeros(cv::Size(IMAGE_SIZE, IMAGE_SIZE), CV_8UC3);
	cv::convexHull(cv::Mat(points), hull[0], false, true);
	cv::convexHull(cv::Mat(left_eye_points), hull[1], false, true);
	cv::convexHull(cv::Mat(right_eye_points), hull[2], false, true);
	cv::convexHull(cv::Mat(mouth_points), hull[3], false, true);

	cv::drawContours(drawing, hull, 0, cv::Scalar(255, 255, 255), CV_FILLED);
	cv::drawContours(drawing, hull, 1, cv::Scalar(0, 0, 0), CV_FILLED);
	cv::drawContours(drawing, hull, 2, cv::Scalar(0, 0, 0), CV_FILLED);
	cv::drawContours(drawing, hull, 3, cv::Scalar(0, 0, 0), CV_FILLED);

	drawing.convertTo(drawing, CV_32FC1); // or CV_32F works (too)
	return Texture(drawing);
}
Texture generate_matte_from_landmarks(const string & name)
{
	ifstream in(name);
	std::vector<cv::Point> points;

	string line;
	while (getline(in, line)) {
		stringstream ss(line);
		int x, y; int z;
		ss >> x >> y >> z;
		points.push_back(cv::Point(x, y));
		
	}
	in.close();
	return generate_matte_from_landmarks(points);
}
Texture generate_nose_matte_from_landmarks(const string & name)
{
	ifstream in(name);
	std::vector<cv::Point> points;

	string line;
	while (getline(in, line)) {
		stringstream ss(line);
		int x, y; int z;
		ss >> x >> y >> z;
		points.push_back(cv::Point(x, y));

	}
	in.close();

	std::vector<cv::Point> nose_points(points.begin() + 29, points.begin() + 36);
	cv::Point ref1, ref2;
	ref1.x = points[31].x; ref1.y = points[30].y;
	ref2.x = points[35].x; ref2.y = points[30].y;

	nose_points.push_back(ref1);
	nose_points.push_back(ref2);

	vector<vector<cv::Point> >  hull(3);
	Mat drawing = Mat::zeros(cv::Size(IMAGE_SIZE, IMAGE_SIZE), CV_8UC3);
	cv::convexHull(cv::Mat(nose_points), hull[0], false, true);

	cv::drawContours(drawing, hull, 0, cv::Scalar(255, 255, 255), CV_FILLED);
	
	drawing.convertTo(drawing, CV_32FC3); // or CV_32F works (too)
	drawing *= 1.0 / 255.0f;
	return Texture(drawing);
}

Texture generate_eye_mouth_matte_from_landmarks(const string & name)
{
	ifstream in(name);
	std::vector<cv::Point> points;

	string line;
	while (getline(in, line)) {
		stringstream ss(line);
		int x, y; int z;
		ss >> x >> y >> z;
		points.push_back(cv::Point(x, y));

	}
	in.close();
	return generate_eye_mouth_matte_from_landmarks(points);
	
}
Texture generate_eye_mouth_matte_from_landmarks(const std::vector<cv::Point>& points)
{
	std::vector<cv::Point> left_eye_points(points.begin() + 17, points.begin() + 22);
	cv::Point ref = left_eye_points[left_eye_points.size() - 1];

	int size = left_eye_points.size() - 1;
	for (int i = 0; i < size; ++i) {
		int x = left_eye_points[i].x;
		int y = ref.y + ref.y - left_eye_points[i].y;
		left_eye_points.push_back(cv::Point(x, y));
	}

	std::vector<cv::Point> right_eye_points(points.begin() + 22, points.begin() + 27);
	ref = right_eye_points[0];
	for (int i = 1; i < size + 1; ++i) {
		int x = right_eye_points[i].x;
		int y = ref.y + ref.y - right_eye_points[i].y;
		right_eye_points.push_back(cv::Point(x, y));
	}
	std::vector<cv::Point> mouth_points(points.begin() + 48, points.begin() + 60);

	vector<vector<cv::Point> >  hull(3);
	Mat drawing = Mat::zeros(cv::Size(IMAGE_SIZE, IMAGE_SIZE), CV_8UC3);
	cv::convexHull(cv::Mat(left_eye_points), hull[0], false, true);
	cv::convexHull(cv::Mat(right_eye_points), hull[1], false, true);
	cv::convexHull(cv::Mat(mouth_points), hull[2], false, true);

	cv::drawContours(drawing, hull, 0, cv::Scalar(255, 255, 255), CV_FILLED);
	cv::drawContours(drawing, hull, 1, cv::Scalar(255, 255, 255), CV_FILLED);
	cv::drawContours(drawing, hull, 2, cv::Scalar(255, 255, 255), CV_FILLED);

	drawing.convertTo(drawing, CV_32FC3); // or CV_32F works (too)
	drawing *= 1.0 / 255.0f;
	return Texture(drawing);
}
std::vector<vec3> cal_normals_from_obj(const string & name)
{
	std::vector<float> vs;
	std::vector<int> is;

	ifstream in(name);
	string line;
	while (getline(in, line)) {
		stringstream s(line);
		string type;
		s >> type;
		if (type == "v") {
			float x, y, z;
			s >> x >> y >> z;
			vs.push_back(x); vs.push_back(y); vs.push_back(z);
		}
		else if (type == "f") {
			int v1, v2, v3;
			s >> v1 >> v2 >> v3;
			is.push_back(v1 - 1); is.push_back(v3 - 1); is.push_back(v2 - 1);
		}
	}

	
	int n_vertex = vs.size() / 3;
	int n_triangle = is.size() / 3;
	float *vertices = new float[vs.size()];
	int* indices = new int[is.size()];
	float* normal = new float[n_vertex * 3];
	for (size_t i = 0; i < vs.size(); ++i) vertices[i] = vs[i];
	for (size_t i = 0; i < is.size(); ++i) indices[i] = is[i];

	calculate_normals(indices, vertices, n_triangle, normal, true);

	vector<vec3> normals;
	for (int i = 0; i < n_vertex; ++i) {
		vec3 n(normal[3 * i], normal[3 * i + 1], normal[3 * i + 2]);

		if (abs(n.x) < 1e-10 && abs(n.y) < 1e-10 && abs(n.z) < 1e-10) {
			n = vec3(0, 0, 0);
		}
		else n.normalize();
		
		normals.push_back(n);
	}
	delete[] vertices;
	delete[] indices;
	delete[] normal;

	return normals;
}

std::vector<vec3> load_normals_from_obj(const string & name)
{
	vector<vec3> normals;
	ifstream in(name);
	string line;

	while (getline(in, line)) {
		stringstream s(line);
		string type;
		s >> type;
		if (type == "vn") {
			float x, y, z;
			s >> x >> y >> z;
			vec3 normal(x, y, z);
			normal.normalize();
			normals.push_back(normal);
		}
	}
	return normals;
}

std::vector<vec3> get_normals_from_bfm(BFM * bfm, const vector<FacePoint>& fps)
{
	vector<vec3> normals;

	int i = 0;
	bfm->update_normals();
	for (auto fp : fps) {
		int face_idx = fp.face_idx;
		float u = fp.u;
		float v = fp.v;
		if (face_idx != -1) {
			vec3 n = bfm->get_normal(face_idx, u, v);
			vec3 on = n;
			n = normalize_normal(n);
			if (isnan(n.x)) {
				cout << n  << on << endl;
			}
			normals.push_back(n);
		}
		else {
			normals.push_back(vec3(0, 0, 0));
		}
		i++;
	}
	return normals;
}

Image depth_map(const string & name)
{
	ifstream in(name);
	
	Image img(IMAGE_SIZE, IMAGE_SIZE);
	img.img  = cv::Scalar(0, 0, 0);

	string line;
	struct Point{
		int x, y;
		float z;
	};
	vector<Point> points;

	while (getline(in, line)) {
		stringstream ss(line);
		string type;
		ss >> type;
		if (type == "v") {
			int x, y; float z;
			ss >> x >> y >> z;
			points.push_back({ x, IMAGE_SIZE - y, z });
		}
	}

	float min_z = FloatMax, max_z = FloatMin;

	for (auto i : points) {
		min_z = std::min(min_z, i.z);
		max_z = std::max(max_z, i.z);
	}

	// [minz, maxz] to [0,1]
	int mininum = 10;
	for (auto i : points) {
		int c =  (255 - mininum) * (i.z - max_z) / (min_z - max_z) + mininum;

		img.setPixel(i.x, i.y, Vec3b(c,c,c));
	}

	return img;
}

Image matte_map(const string & name)
{
	ifstream in(name);

	Image img(IMAGE_SIZE, IMAGE_SIZE);
	img.img = cv::Scalar(0, 0, 0);

	string line;
	
	vector<vec2i> points;

	while (getline(in, line)) {
		stringstream ss(line);
		string type;
		ss >> type;
		if (type == "v") {
			int x, y; float z;
			ss >> x >> y >> z;
			points.push_back({ x, IMAGE_SIZE - y});
		}
	}

	for (auto i : points) {
		img.setPixel(i.x, i.y, Vec3b(255, 255, 255));
	}

	return img;
}

bool Landmarks::load(BFM * bfm, std::vector<vec3i>& data)
{
	for (auto p : data) {
		int a = p.x, b = p.y, c = p.z;
		vec3 p = bfm->get_vertex(c);
		landmarks_indices.push_back(c);
		landmarks_3d.push_back(vec4(p.x, p.y, p.z, 1.0f));
		landmarks_2d.push_back(vec2i(a, b));
	}
	size = landmarks_indices.size();
	return true;
}

bool Landmarks::load(BFM * bfm, const string & name)
{
	ifstream in(name);
	if (!in) {
		cout << "cannot open file" << endl;
		return false;
	}
	std::string line;
	while (getline(in, line)) {

		stringstream ss(line);
		int a, b, c;
		ss >> a >> b >> c;
		string name;
		ss >> name;
		if (name == "right_eye_center" || name == "left_eye_center") {
			continue;
		}
		vec3 p = bfm->get_vertex(c);
		landmarks_indices.push_back(c);
		landmarks_3d.push_back(vec4(p.x, p.y, p.z, 1.0f));
		landmarks_2d.push_back(vec2i(a, b));
		landmarks_name.push_back(name);
	}
	in.close();
	size = landmarks_indices.size();
	return true;
}

bool Landmarks3D::load(BFM * bfm, const string & name, const string & obj_name)
{
	ifstream in(name);
	if (!in) {
		cout << "cannot open file" << endl;
		return false;
	}
	std::string line;
	while (getline(in, line)) {

		stringstream ss(line);
		int a, b, c;
		ss >> a >> b >> c;
		string name;
		ss >> name;
		if (name == "right_eye_center" || name == "left_eye_center") {
			continue;
		}
		vec3 p = bfm->get_vertex(c);
		landmarks_indices.push_back(c);
		landmarks_bfm.push_back(vec4(p.x, p.y, p.z, 1.0f));
		landmarks_obj.push_back(vec3(float(a), float(b), 0.0));
		landmarks_name.push_back(name);
	}
	in.close();
	size = landmarks_indices.size();

	in = ifstream(obj_name);
	std::map<vec2i, float> points;
	float min_z = FloatMax, max_z = FloatMin;
	while (getline(in, line)) {
		stringstream ss(line);
		string type;
		ss >> type;
		if (type == "v") {
			int x, y; float z;
			ss >> x >> y >> z;
			points[vec2i(x, y)] = z;
			min_z = std::min(min_z, z);
			max_z = std::max(max_z, z);
		}
	}
	in.close();
	for (size_t i = 0; i < landmarks_obj.size(); ++i) {
		landmarks_obj[i].z = points[vec2i(landmarks_obj[i].x, landmarks_obj[i].y)];
	}
	return true;
}
