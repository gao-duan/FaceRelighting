#include "math_vec.h"

float Dot(const vec3 & v1, const vec3 & v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

vec3 Cross(const vec3 & v1, const vec3 & v2)
{
	const float _x = v1.y * v2.z - v1.z * v2.y;
	const float _y = v1.z * v2.x - v1.x * v2.z;
	const float _z = v1.x * v2.y - v1.y * v2.x;

	return vec3(_x, _y, _z);
}
mat4 Identity()
{
	mat4 res;
	res.setIdentity();
	return res;
}
mat4 Translate(float x, float y, float z)
{
	mat4 res;
	res.setIdentity();
	res.m[0][3] = x;
	res.m[1][3] = y;
	res.m[2][3] = z;
	return res;
}
mat4 Translate(const vec3 & v)
{
	return Translate(v.x, v.y, v.z);
}
mat4 Scale(float x, float y, float z)
{
	mat4 res;
	res.setIdentity();
	res.m[0][0] = x;
	res.m[1][1] = y;
	res.m[2][2] = z;
	return res;
}
mat4 Scale(float s)
{
	return Scale(s, s, s);
}
mat4 Perspective(float tanHalfFOV, float aspect, float z_near, float z_far)
{
	mat4 res;
	const float zRange = z_near - z_far;

	res.m[0][0] = 1.0f / (tanHalfFOV * aspect); res.m[0][1] = 0.0f;           res.m[0][2] = 0.0f;            res.m[0][3] = 0.0;
	res.m[1][0] = 0.0f;                   res.m[1][1] = 1.0f / tanHalfFOV; res.m[1][2] = 0.0f;            res.m[1][3] = 0.0;
	res.m[2][0] = 0.0f;                   res.m[2][1] = 0.0f;            res.m[2][2] = (-z_near - z_far) / zRange; res.m[2][3] = 2.0f* z_far * z_near / zRange;
	res.m[3][0] = 0.0f;                   res.m[3][1] = 0.0f;            res.m[3][2] = 1.0f;            res.m[3][3] = 0.0;
	return res;
}
mat4 get_perspective_mat(float camera_distance, float camera_fov_x, float yxaspect)
{
	mat4 mat = Identity();
	mat.data[0] = camera_distance / camera_fov_x;
	mat.data[5] = camera_distance / (camera_fov_x * yxaspect);
	mat.data[14] = -1;
	mat.data[15] = camera_distance;
	return mat;
}
mat4 get_rotation_mat(float yaw, float pitch, float roll)
{
	mat4 mat = Identity();
	float alpha = yaw, beta = pitch, gamma = roll;
	mat.data[0] = cos(alpha)*cos(beta);
	mat.data[1] = cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma);
	mat.data[2] = cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma);
	mat.data[4] = sin(alpha)*cos(beta);
	mat.data[5] = sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma);
	mat.data[6] = sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma);
	mat.data[8] = -sin(beta);
	mat.data[9] = cos(beta)*sin(gamma);
	mat.data[10] = cos(beta)*cos(gamma);
	return mat;
}
mat4 Orthographic(float l, float r, float b, float t, float n, float f) {
	mat4 res;
	res.m[0][0] = 2.0f / (r - l); res.m[0][1] = 0.0f;         res.m[0][2] = 0.0f;         res.m[0][3] = -(r + l) / (r - l);
	res.m[1][0] = 0.0f;         res.m[1][1] = 2.0f / (t - b); res.m[1][2] = 0.0f;         res.m[1][3] = -(t + b) / (t - b);
	res.m[2][0] = 0.0f;         res.m[2][1] = 0.0f;         res.m[2][2] = 2.0f / (f - n); res.m[2][3] = -(f + n) / (f - n);
	res.m[3][0] = 0.0f;         res.m[3][1] = 0.0f;         res.m[3][2] = 0.0f;         res.m[3][3] = 1.0;
	return res;
}
mat4 View(const vec3 & camera_pos, const vec3 & target, const vec3 & up)
{
	mat4 res;
	vec3 N = target;
	N.normalize();
	vec3 U = up;
	U = U.Cross(N);
	U.normalize();
	vec3 V = N.Cross(U);

	res.m[0][0] = U.x;   res.m[0][1] = U.y;   res.m[0][2] = U.z;  res.m[0][3] = 0.0f;
	res.m[1][0] = V.x;   res.m[1][1] = V.y;   res.m[1][2] = V.z;  res.m[1][3] = 0.0f;
	res.m[2][0] = N.x;   res.m[2][1] = N.y;   res.m[2][2] = N.z;  res.m[2][3] = 0.0f;
	res.m[3][0] = 0.0f;  res.m[3][1] = 0.0f;  res.m[3][2] = 0.0f;  res.m[3][3] = 1.0f;

	return res * Translate(-camera_pos);
}

mat4 Rotate(float angle_x, float angle_y, float angle_z)
{
	mat4 rx, ry, rz;

	const float x = toRadian(angle_x);
	const float y = toRadian(angle_y);
	const float z = toRadian(angle_z);

	rx.m[0][0] = 1.0f; rx.m[0][1] = 0.0f; rx.m[0][2] = 0.0f; rx.m[0][3] = 0.0f;
	rx.m[1][0] = 0.0f; rx.m[1][1] = cosf(x); rx.m[1][2] = -sinf(x); rx.m[1][3] = 0.0f;
	rx.m[2][0] = 0.0f; rx.m[2][1] = sinf(x); rx.m[2][2] = cosf(x); rx.m[2][3] = 0.0f;
	rx.m[3][0] = 0.0f; rx.m[3][1] = 0.0f; rx.m[3][2] = 0.0f; rx.m[3][3] = 1.0f;

	ry.m[0][0] = cosf(y); ry.m[0][1] = 0.0f; ry.m[0][2] = -sinf(y); ry.m[0][3] = 0.0f;
	ry.m[1][0] = 0.0f; ry.m[1][1] = 1.0f; ry.m[1][2] = 0.0f; ry.m[1][3] = 0.0f;
	ry.m[2][0] = sinf(y); ry.m[2][1] = 0.0f; ry.m[2][2] = cosf(y); ry.m[2][3] = 0.0f;
	ry.m[3][0] = 0.0f; ry.m[3][1] = 0.0f; ry.m[3][2] = 0.0f; ry.m[3][3] = 1.0f;

	rz.m[0][0] = cosf(z); rz.m[0][1] = -sinf(z); rz.m[0][2] = 0.0f; rz.m[0][3] = 0.0f;
	rz.m[1][0] = sinf(z); rz.m[1][1] = cosf(z); rz.m[1][2] = 0.0f; rz.m[1][3] = 0.0f;
	rz.m[2][0] = 0.0f; rz.m[2][1] = 0.0f; rz.m[2][2] = 1.0f; rz.m[2][3] = 0.0f;
	rz.m[3][0] = 0.0f; rz.m[3][1] = 0.0f; rz.m[3][2] = 0.0f; rz.m[3][3] = 1.0f;

	return  rz * ry * rx;
}
float toRadian(float f)
{
	return f / 180.0f * PI;
}

float toDegree(float f)
{
	return f / PI * 180.0f;
}

float getPerimeter(const vec3& p1, const vec3&p2, const vec3& p3) {
	return Cross(p2 - p1, p3 - p1).length();
}
float getArea(const vec3& p1, const vec3&p2, const vec3& p3) {
	float c1 = (p1 - p2).length();
	float c2 = (p1 - p3).length();
	float c3 = (p3 - p2).length();
	return c1 + c2 + c3;
}
void calculate_normals(const int* indices, const float* vertices, const int n_triangle, float* normals, bool inverse)
{
	vector<pair<vec3, float> > face_normals;
	map<int, set<int> > vertex_faces;
	for (int i = 0; i < n_triangle; ++i) {
		int idx1 = indices[3 * i];
		int idx2 = indices[3 * i + 1];
		int idx3 = indices[3 * i + 2];
		if (inverse) {
			std::swap(idx2, idx3);
		}
	
		vec3 p1(vertices[3 * idx1], vertices[3 * idx1 + 1], vertices[3 * idx1 + 2]);
		vec3 p2(vertices[3 * idx2], vertices[3 * idx2 + 1], vertices[3 * idx2 + 2]);
		vec3 p3(vertices[3 * idx3], vertices[3 * idx3 + 1], vertices[3 * idx3 + 2]);

	
		vec3 n = Cross(p2 - p1, p3 - p1);
		float area = 0.5 * n.length();
		n.normalize();
		face_normals.push_back(make_pair(n, area));
		vertex_faces[idx1].insert(i);
		vertex_faces[idx2].insert(i);
		vertex_faces[idx3].insert(i);
	}

	for (auto i = vertex_faces.begin(); i != vertex_faces.end(); ++i) {
		int vertex_index = i->first;
		set<int> neighbor_face_indices = i->second;

		vec3 n(0, 0, 0);
		for (auto idx : neighbor_face_indices) {
			vec3 n_tmp = face_normals[idx].first;
			
			float area = face_normals[idx].second;
			n = n + n_tmp * area;

		}
		
		n.normalize();
		normals[3 * vertex_index] = n.x;
		normals[3 * vertex_index + 1] = n.y;
		normals[3 * vertex_index + 2] = n.z;
	}
}

vec3 normalize_normal(const vec3 & n)
{
	vec3 res(n);
	if (abs(n.x) < 1e-10 && abs(n.y) < 1e-10 && abs(n.z) < 1e-10) {
		res = vec3(0, 0, 0);
	}
	else {
		res.normalize();
	}
	return res;
}

vec3 convert_vec3b_to_vec3(const Vec3b & n)
{
	float x = n[0] / 255.0;
	float y = n[1] / 255.0;
	float z = n[2] / 255.0;

	return vec3(x, y, z);
}

Vec3b convert_vec3_to_vec3b(const vec3 & n)
{
	return Vec3b(n.x * 255, n.y * 255, n.z * 255);
}

vec3 Clamp(vec3 v, float a, float b)
{
	float x = Clamp(v.x, a, b);
	float y = Clamp(v.y, a, b);
	float z = Clamp(v.z, a, b);
	return vec3(x, y, z);
}

Vec3b Clamp(Vec3b v)
{
	int x = Clamp(int(v[0]), 0, 255);
	int y = Clamp(int(v[1]), 0, 255);
	int z = Clamp(int(v[2]), 0, 255);
	return Vec3b(x, y, z);
}

float gaussian(float x, float mu, float sigma)
{
	return 1.0 / (sigma * sqrt(2.0 * PI)) * exp(-(x - mu) * (x-mu) / 2.0 / sigma / sigma);
}

float sqr(float x, float alpha)
{
	float sigma = -1.0 / log(alpha);
	return  exp(-(x - 1) * (x - 1) / sigma);
}

vec3 sphere2cartesian(float theta, float phi)
{
	return vec3(sin(theta)*cos(phi),sin(theta) * sin(phi), cos(theta));
}
