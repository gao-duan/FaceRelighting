#pragma once

#include "math_vec.h"
#include "BFM.h"

struct Ray {
	vec3 orig, dir, inv_dir;
	float dist;
	float u, v;
	Ray(const vec3& o, const vec3& d, float dist = FloatInfity);
};


struct Triangle {
	Triangle();
	int id;
	vec3 V[3];
	vec3 v10, v20, iv10, iv20; // iv10 = (v1 - v0) / length(v1 - v0)^2
	vec3 normal;
	vec3 n0;
	vec3 n10, n20;
	float area;

	Triangle(const vec3 &v0, const vec3 &v1, const vec3 &v2,
		const vec3 &n0, const vec3 &n1, const vec3 &n2);
	Triangle(const Triangle& _t);
	void get_coord(const Ray &ray, double dist, double &coord_u, double &coord_v) const;
	bool intersect(Ray &ray);
};
struct GridNode {
	vector<Triangle> triangles;
	vector<int> triangle_indices;

	bool intersect(Ray& ray, int& face_idx);
	GridNode() {}
};

struct GridAccel {
	GridNode** nodes;
	int width;
	GridAccel();
	GridAccel(const vector<Triangle>& triangles, int);
	bool intersect(Ray& ray, int& face_idx);
	~GridAccel();
};

struct TriangleMesh {
	std::vector<Triangle> triangles;
	GridAccel* accel;
	TriangleMesh(int n_vertex, int n_triangle, const vector<float>& _vertices, const vector<int>& _indices);
	bool intersect(Ray& ray, int& face_idx, bool use_grid_accl);
	~TriangleMesh();
};

