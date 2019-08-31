#include "mesh.h"

Ray::Ray(const vec3 & o, const vec3 & d, float dist)
	:orig(o), dir(d), dist(dist), u(0), v(0) {
	inv_dir = vec3(1, 1, 1) / d;
}

Triangle::Triangle()
{
}

Triangle::Triangle(const vec3 & v0, const vec3 & v1, const vec3 & v2, const vec3 & n0, const vec3 & n1, const vec3 & n2)
{
	V[0] = v0;
	V[1] = v1;
	V[2] = v2;
	this->n0 = n0;

	v10 = v1 - v0;
	v20 = v2 - v0;
	iv10 = v10 * (1.0f / Dot(v10, v10));
	iv20 = v20 * (1.0f / Dot(v20, v20));
	n10 = n1 - n0;
	n20 = n2 - n0;
	this->id = id;
	normal = Cross(v10, v20); normal.normalize();
	area = 0.5f * (Cross(V[1] - V[0], V[2] - V[0])).length();
}

Triangle::Triangle(const Triangle & _t)
{
	V[0] = _t.V[0];	V[1] = _t.V[1];V[2] = _t.V[2];
	n0 = _t.n0;
	v10 = _t.v10;
	v20 = _t.v20;
	iv10 = _t.iv10;
	iv20 = _t.iv20;
	n10 = _t.n10;
	n20 = _t.n20;
	id = _t.id;
	normal = _t.id;
	area = _t.area;
}

void Triangle::get_coord(const Ray & ray, double dist, double & coord_u, double & coord_v) const
{
	const vec3 inter_local = ray.orig + ray.dir * dist - V[0];
	const vec3 u = v10, v = v20;
	double uv = Dot(u, v), vv = Dot(v, v), wu = Dot(inter_local, u), uu = Dot(u, u), wv = Dot(inter_local, v);
	double dom = uv * uv - uu * vv;
	coord_u = (uv * wv - vv * wu) / dom;
	coord_v = (uv * wu - uu * wv) / dom;
}

bool Triangle::intersect(Ray & ray) {
	vec3 p1 = V[0], p2 = V[1], p3 = V[2];
	float t;
	float A = p1.x - p2.x;
	float B = p1.y - p2.y;
	float C = p1.z - p2.z;

	float D = p1.x - p3.x;
	float E = p1.y - p3.y;
	float F = p1.z - p3.z;

	float G = ray.dir.x;
	float H = ray.dir.y;
	float I = ray.dir.z;

	float J = p1.x - ray.orig.x;
	float K = p1.y - ray.orig.y;
	float L = p1.z - ray.orig.z;

	float EIHF = E*I - H*F;
	float GFDI = G*F - D*I;
	float DHEG = D*H - E*G;

	float denom = (A*EIHF + B*GFDI + C*DHEG);

	float beta = (J*EIHF + K*GFDI + L*DHEG) / denom;

	if (beta <= 0.0 || beta >= 1) {
		return false;
	}

	float AKJB = A*K - J*B;
	float JCAL = J*C - A*L;
	float BLKC = B*L - K*C;

	float gamma = (I*AKJB + H*JCAL + G*BLKC) / denom;
	if (gamma <= 0.0 || beta + gamma >= 1.0) return false;
	//cout << beta << " " << gamma << endl;
	t = -(F*AKJB + E*JCAL + D*BLKC) / denom;
	if (t < 0 || t > ray.dist) {
		return false;
	}
	ray.dist = t;
	ray.u = beta;
	ray.v = gamma;

	return true;
}

TriangleMesh::TriangleMesh(int n_vertex, int n_triangle, const vector<float>& _vertices, const vector<int>& _indices)
{
	float* normals = new float[n_vertex * 3];
	int* indices = new int[n_triangle * 3];
	float* vertices = new float[n_vertex * 3];

	for (int i = 0; i < n_vertex * 3; ++i) {
		vertices[i] = _vertices[i];
	}
	for (int i = 0; i < n_triangle * 3; ++i) {
		indices[i] = _indices[i];
	}
	calculate_normals(indices, vertices, n_triangle, normals);
	for (int i = 0; i < n_triangle; ++i) {
		int idx1 = indices[3 * i];
		int idx2 = indices[3 * i + 1];
		int idx3 = indices[3 * i + 2];

		vec3 p1(vertices[3 * idx1], vertices[3 * idx1 + 1], vertices[3 * idx1 + 2]);
		vec3 p2(vertices[3 * idx2], vertices[3 * idx2 + 1], vertices[3 * idx2 + 2]);
		vec3 p3(vertices[3 * idx3], vertices[3 * idx3 + 1], vertices[3 * idx3 + 2]);

		vec3 n1(normals[3 * idx1], normals[3 * idx1 + 1], normals[3 * idx1 + 2]);
		vec3 n2(normals[3 * idx2], normals[3 * idx2 + 1], normals[3 * idx2 + 2]);
		vec3 n3(normals[3 * idx3], normals[3 * idx3 + 1], normals[3 * idx3 + 2]);

		Triangle tri(p1, p2, p3, n1, n2, n3);
		triangles.push_back(tri);
	}
	accel = new GridAccel(triangles, 250);
	delete[] normals;
	delete[] indices;
	delete[] vertices;

}

bool TriangleMesh::intersect(Ray & ray, int & face_idx, bool use_grid_accl)
{
	if (use_grid_accl) {
		return accel->intersect(ray, face_idx);
	}
	else {
		face_idx = -1;
		float min_dist = FloatMax;

		for (size_t i = 0; i < triangles.size(); ++i) {

			if (triangles[i].intersect(ray)) {
				if (min_dist > ray.dist) {
					min_dist = ray.dist;
					face_idx = i;
				}
			}
		}

		return face_idx != -1;
	}
}


TriangleMesh::~TriangleMesh()
{
	delete accel;
}

bool GridNode::intersect(Ray & ray, int & face_idx)
{
	face_idx = -1;
	float min_dist = FloatMax;
	for (size_t i = 0; i < triangles.size(); ++i) {

		if (triangles[i].intersect(ray)) {
			if (min_dist > ray.dist) {
				min_dist = ray.dist;
				face_idx = triangle_indices[i];
			}
		}
	}
	return face_idx != -1;
}

double CrossProduct(const vec2& a, const vec2& b)
{
	return a.x*b.y - a.y*b.x;
}
bool point_inside_triangle(vec2 p, vec2 A, vec2 B, vec2 C) {
	vec2 PA = A - p;
	vec2 PB = B - p;
	vec2 PC = C - p;
	float t1 = CrossProduct(PA, PB);
	float t2 = CrossProduct(PB, PC);
	float t3 = CrossProduct(PC, PA);
	if (t1 * t2 >= 0 && t2 * t3 >= 0) {
		return true;
	}
	else return false;
}
#define SATISFI(x) ( (x) >=0 && (x) < 10 )
#define MyAssert(exp, ...) if(!(exp)) {printf("assert:%s\n",#exp);system("pause");}
GridAccel::GridAccel()
{
}
GridAccel::GridAccel(const vector<Triangle>& triangles, int n)
{
	
	nodes = new GridNode*[n];
	for (int i = 0; i < n; ++i) {
		nodes[i] = new GridNode[n];
	}
	int count = 0;
	for (auto tri : triangles) {
		vec3 p1 = tri.V[0], p2 = tri.V[1], p3 = tri.V[2];
		int min_x = std::min(p1.x, std::min(p2.x, p3.x)) - 1;
		int max_x = std::max(p1.x, std::max(p2.x, p3.x)) + 1;
		int min_y = std::min(p1.y, std::min(p2.y, p3.y)) - 1;
		int max_y = std::max(p1.y, std::max(p2.y, p3.y)) + 1;
		if (min_x < 0) min_x = 0;
		if (max_x >= 250) max_x = 249;
		if (min_y < 0) min_y = 0;
		if (max_y >= 250) max_y = 249;
		
		for (int x = min_x; x <= max_x; ++x) {
			for (int y = min_y; y <= max_y; ++y) {
		
				if(point_inside_triangle(vec2(x,y), p1.xy(), p2.xy(), p3.xy())) {
					nodes[x][y].triangles.push_back(tri);
					nodes[x][y].triangle_indices.push_back(count);
				}
			}
		}	
		count++;
	}
	
}


bool GridAccel::intersect(Ray & ray, int & face_idx)
{

	vec2i Coords;
	width = 1.0f;
	Coords.x = ray.orig.x / width;
	Coords.y = ray.orig.y / width;

	GridNode& node = nodes[Coords.x][Coords.y];
	return node.intersect(ray, face_idx);
}

GridAccel::~GridAccel()
{
	int n = 250;
	for (int i = 0; i < n; ++i) {
		delete[] nodes[i];
	}
	delete[] nodes;
}
