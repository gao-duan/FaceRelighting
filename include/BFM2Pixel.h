#pragma once
#include "BFM.h"
#include "mesh.h"
#include "image.h"

struct FacePoint {
	int face_idx;
	float u, v;
};

class BFM2Pixel {
private:
	vector<vec2i> valid_points;
	
	vector<float> vertices;
	vector<int> indices;
	int n_vertex;
	int n_triangle;

	BFM* bfm;

	void map_first_time();
	void map_load_from_file();

	std::string path;

public:
	vector<FacePoint> face_points;
	BFM2Pixel();
	BFM2Pixel(BFM* bfm, const vector<vec2i>& _valid_points, const vector<float>& _vertices,
		const vector<int>& _indices, const std::string& res_name);

	void map(bool force = false);

	void test();
};

