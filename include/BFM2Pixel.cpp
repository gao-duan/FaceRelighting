#include "BFM2Pixel.h"

BFM2Pixel::BFM2Pixel() {

}
BFM2Pixel::BFM2Pixel(BFM * bfm, const vector<vec2i>& _valid_points, const vector<float>& _vertices, const vector<int>& _indices, const std::string& _path)
	:
	bfm(bfm), path(_path)
{
	for (auto i : _valid_points) {
		valid_points.push_back(i);
	}
	for (auto i : _vertices) {
		vertices.push_back(i);
	}
	for (auto i : _indices) {
		indices.push_back(i);
	}
	face_points.resize(valid_points.size());
	n_vertex = vertices.size() / 3;
	n_triangle = indices.size() / 3;
}

void BFM2Pixel::map(bool force)
{
	if (force) {
		map_first_time();
		return;
	}
	else {
		ifstream in(path);
		if (!in) {
			map_first_time();
		}
		else {
			map_load_from_file();
		}
		in.close();
	}
}

void BFM2Pixel::map_first_time()
{
	TriangleMesh mesh(n_vertex, n_triangle, vertices, indices);

	for (size_t i = 0; i < valid_points.size(); ++i) {
		vec2i p = valid_points[i];
		Ray ray(vec3(p.x, p.y, 10000), vec3(0, 0, -1));

		int idx = -1;
		if (mesh.intersect(ray, idx, true)) {
			face_points[i] = { idx, ray.u, ray.v };
		}
		else {
			face_points[i] = { -1, 0, 0 };
		}
		if (idx == -1) {
		//	cerr << endl << "no intersection: " << p << endl;
		}
	}
}

void BFM2Pixel::map_load_from_file()
{
	ifstream in(path);
	if (!in) {
		cerr << "cannot open file" << endl;
		return;
	}
	string line;
	int idx = 0;
	while (getline(in, line)) {
		stringstream s(line);
		s >> face_points[idx].face_idx >> face_points[idx].u >> face_points[idx].v;
		idx++;
	}
	in.close();

}

void BFM2Pixel::test()
{
	ifstream in(path);
	if (!in) {
		cerr << "cannot open file" << endl;
		return;
	}
	string line;
	int idx = 0;
	while (getline(in, line)) {
		stringstream s(line);
		s >> face_points[idx].face_idx >> face_points[idx].u >> face_points[idx].v;
		idx++;
	}
	in.close();

	Image res(IMAGE_SIZE, IMAGE_SIZE);
	for (int i = 0; i < res.height; ++i) {
		for (int j = 0; j < res.width; ++j) {
			res.setPixel(j, i, Vec3b(255, 0, 0));
		}
	}
	for (size_t i = 0; i < valid_points.size(); ++i) {
		vec2i p = valid_points[i];
		FacePoint fp = face_points[i];
		if (fp.face_idx != -1) {
			vec3 color = bfm->get_color(fp.face_idx, fp.u, fp.v);
		
			res.setPixel(p.x, p.y, Vec3b(int(color.z), int(color.y), int(color.x)));
		}
		else {
			res.setPixel(p.x, p.y, Vec3b(0, 0, 255));
		}
	}

}
