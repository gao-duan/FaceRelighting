#pragma once
#include "common.h"
enum class AXIS { X, Y, Z };

template<typename T>
struct Vec2 {
	T x, y;
	Vec2(T x = 0, T y = 0) :x(x), y(y) {}

	template<typename S>
	Vec2(const Vec2<S>& v) : x(v.x), y(v.y) {}

	T& operator[](int i) {
		switch (i)
		{
		case 0:return x; break;
		case 1:return y; break;
		default:
			break;
		}
	}
	const T& operator[](int i) const {
		switch (i)
		{
		case 0:return x; break;
		case 1:return y; break;
		default:
			break;
		}
	}
	
	friend ostream& operator<<(ostream& out, const Vec2<T>& v) {
		out << "[ " << v.x << ", " << v.y << " ]";
		return out;
	}
	float length() const {
		float res = x * x + y * y;
		return sqrt(res);
	}

	bool operator==(const Vec2<T>& rhs) const {
		return x == rhs.x && y == rhs.y;
	}
	bool operator <(const Vec2<T>& rhs) const
	{
		if (x != rhs.x) return x < rhs.x;
		else  return y < rhs.y;
	}
};
typedef Vec2<int> vec2i;
typedef Vec2<float> vec2;
template<typename T>
struct Vec4;
template<typename T>
struct Vec3 {
	union { T x; T R; };
	union { T y; T G; };
	union { T z; T B; };
	
	Vec3(T x = 0, T y = 0, T z = 0) :x(x), y(y), z(z) {}
	Vec3(const Vec4<T>& v) :x(v.x), y(v.y), z(v.z) {}
	
	Vec3(const Vec2<T>& v) :x(v.x), y(v.y), z(1) {}
	Vec3(const Vec3<T>& v) :x(v.x), y(v.y), z(v.z) {}

	template<typename S>
	Vec3(const Vec3<S>& v) : x(v.x), y(v.y), z(v.z) {}

	Vec2<T> xy() const {
		return Vec2<T>(x, y);
	}

	T& operator[](int i) {
		switch (i)
		{
		case 0:return x; break;
		case 1:return y; break;
		case 2:return z; break;
		default:
			break;
		}
	}
	const T& operator[](int i) const {
		switch (i)
		{
		case 0:return x; break;
		case 1:return y; break;
		case 2:return z; break;
		default:
			break;
		}
	}

	Vec3<T> operator-() const
	{
		Vec3<T> Ret(-x, -y, -z);
		return Ret;
	}
	template<typename T>
	friend ostream& operator<<(ostream& out, const Vec3<T>& v);

	float length() const {
		float res = x*x + y * y + z*z;
		return sqrt(res);
	}
	float maxComponent() const {
		if (x > y) {
			if (x > z) return x;
			else return z;
		}
		else {
			if (y > z) return y;
			else return z;
		}
	}
	float minComponent() const {
		if (x < y) {
			if (x < z) return x;
			else return z;
		}
		else {
			if (y < z) return y;
			else return z;
		}
	}
	int maxComponentIndex() const {
		if (x >= y) {
			if (x >= z) return 0;
			else return 2;
		}
		else {
			if (y >= z) return 1;
			else return 2;
		}
	}
	int minComponentIndex() const {
		if (x <= y) {
			if (x <= z) return 0;
			else return 2;
		}
		else {
			if (y <= z) return 1;
			else return 2;
		}
	}
	void normalize() {
		double len = length();
		if (len == 0) {
			x = 0; y = 0; z = 0;
			return;
		}
		x /= len;
		y /= len;
		z /= len;
	}
	Vec3 Cross(const Vec3& v) const {
		const float _x = y * v.z - z * v.y;
		const float _y = z * v.x - x * v.z;
		const float _z = x * v.y - y * v.x;

		return Vec3(_x, _y, _z);
	}

	bool operator==(const Vec3<T>& rhs) const {
		return abs(x - rhs.x) < 1e-3 && abs(y - rhs.y) < 1e-3 && abs(z - rhs.z) < 1e-3;
	}
	bool operator!=(const Vec3<T>& rhs) const {
		return !((*this) == rhs);
	}
	bool operator <(const Vec3<T>& rhs) const
	{
		if (x != rhs.x) return x < rhs.x;
		else if(y != rhs.y) return y < rhs.y;
		else return z < rhs.z;
	}
	float Sum() const {
		return x + y + z;
	}
	float Lumiance() const {
		return (x + y + z) / 3.0f;
	}
};
typedef Vec3<int> vec3i;
typedef Vec3<float> vec3;
template<typename T>
struct Vec4 {
	T x, y, z, w;
	Vec4(T x = 0, T y = 0, T z = 0, T w = 1) :x(x), y(y), z(z), w(w) {}
	Vec4(const Vec3<T>& v, float w = 1.0) :x(v.x), y(v.y), z(v.z), w(w) {}

	Vec4<T> div_w() {
		return Vec4<T>(x / w, y / w, z / w, 1.0f);
	}
	Vec2<T> xy() const {
		return Vec2<T>(x, y);
	}
	
	Vec3<T> xyz() const {
		return Vec3<T>(x, y, z);
	}

	T& operator[](int i) {
		switch (i)
		{
		case 0:return x; break;
		case 1:return y; break;
		case 2:return z; break;
		case 3:return w; break;
		default:
			break;
		}
	}
	const T& operator[](int i) const {
		switch (i)
		{
		case 0:return x; break;
		case 1:return y; break;
		case 2:return z; break;
		case 3:return w; break;
		default:
			break;
		}
	}
	float length() const {
		float res = x*x + y * y + z*z + w * w;
		return sqrt(res);
	}
	template<typename T>
	friend ostream& operator<<(ostream& out, const Vec4<T>& v);

};

template<typename T>
ostream& operator<<(ostream& out, const Vec4<T>& v) {
	out << "[ " << v.x << ", " << v.y << ", " << v.z << ", " << v.w << " ]";
	return out;
}
template<typename T>
ostream& operator<<(ostream& out, const Vec3<T>& v) {
	out << "[ " << v.x << ", " << v.y << ", " << v.z << " ]";
	return out;
}

template<typename T>
inline Vec3<T> operator+(const Vec3<T>& l, const Vec3<T>& r)
{
	Vec3<T> Ret(l.x + r.x,
		l.y + r.y,
		l.z + r.z);

	return Ret;
}
template<typename T>
inline Vec3<T> operator-(const Vec3<T>& l, const Vec3<T>& r)
{
	Vec3<T> Ret(l.x - r.x,
		l.y - r.y,
		l.z - r.z);

	return Ret;
}

template<typename T>
inline Vec3<T> operator*(const Vec3<T>& l, const Vec3<T>& r)
{
	Vec3<T> Ret(l.x * r.x,
		l.y * r.y,
		l.z * r.z);

	return Ret;
}
template<typename T>
inline Vec3<T> operator/(const Vec3<T>& l, const Vec3<T>& r)
{
	Vec3<T> Ret(l.x / (1e-5+r.x),
		l.y / (1e-5 + r.y),
		l.z / (1e-5 + r.z));

	return Ret;
}
template<typename T>
inline Vec3<T> operator*(const Vec3<T>& l, float f)
{
	Vec3<T> Ret(l.x * f,
		l.y * f,
		l.z * f);

	return Ret;
}
template<typename T>
inline Vec3<T> operator*(float f, const Vec3<T>& l)
{
	Vec3<T> Ret(l.x * f,
		l.y * f,
		l.z * f);

	return Ret;
}
template<typename T>
inline Vec2<T> operator+(const Vec2<T>& l, const Vec2<T>& r)
{
	Vec2<T> Ret(l.x + r.x,
		l.y + r.y);
	return Ret;
}
template<typename T>
inline Vec2<T> operator-(const Vec2<T>& l, const Vec2<T>& r)
{
	Vec2<T> Ret(l.x - r.x,
		l.y - r.y);
	return Ret;
}
template<typename T>
inline Vec2<T> operator*(const Vec2<T>& l, float f)
{
	Vec2<T> Ret(l.x * f,
		l.y * f);
	return Ret;
}
template<typename T>
inline Vec2<T> operator*(float f, const Vec2<T>& l)
{
	Vec2<T> Ret(l.x * f,
		l.y * f);
	return Ret;
}
template<typename T>
inline  Vec4<T> operator/(const Vec4<T>& l, float f)
{
	Vec4<T> Ret(l.x / f,
		l.y / f,
		l.z / f,
		l.w / f);

	return Ret;
}
template<typename T>
inline  Vec3<T> operator/(const Vec3<T>& l, float f)
{
	Vec3<T> Ret(l.x / f,
		l.y / f,
		l.z / f);

	return Ret;
}
template<typename T>
inline  Vec2<T> operator/(const Vec2<T>& l, float f)
{
	Vec2<T> Ret(l.x / f,
		l.y / f);
	return Ret;
}


typedef Vec4<float> vec4;
typedef Vec4<int> vec4i;

struct mat4 {
	union {
		float m[4][4];
		float data[16];
	};
	mat4() {
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				m[i][j] = 0;
			}
		}
	}
	mat4(float n[][4]) {
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				m[i][j] = n[i][j];
			}
		}
	}
	mat4 transpose() const
	{
		mat4 n;
		for (unsigned int i = 0; i < 4; i++) {
			for (unsigned int j = 0; j < 4; j++) {
				n.m[i][j] = m[j][i];
			}
		}
		return n;
	}
	mat4 inverse() const
	{
		Eigen::Matrix3f eigen_m;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				eigen_m(i, j) = m[i][j];
			}
		}
		float det = eigen_m.determinant();
		cout << "det: " << det << endl;
		eigen_m = eigen_m.inverse();
		mat4 res;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				res.m[i][j] = eigen_m(i, j);
			}
		}
		return res;
	}
	inline mat4 operator*(const mat4& Right) const
	{
		mat4 Ret;

		for (unsigned int i = 0; i < 4; i++) {
			for (unsigned int j = 0; j < 4; j++) {
				Ret.m[i][j] = m[i][0] * Right.m[0][j] +
					m[i][1] * Right.m[1][j] +
					m[i][2] * Right.m[2][j] +
					m[i][3] * Right.m[3][j];
			}
		}

		return Ret;
	}

	vec4 operator*(const vec4& v) const
	{
		vec4 r;

		r.x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w;
		r.y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w;
		r.z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w;
		r.w = m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w;

		return r;
	}
	mat4& operator*=(const mat4& n)
	{
		for (unsigned int i = 0; i < 4; i++) {
			for (unsigned int j = 0; j < 4; j++) {
				m[i][j] = m[i][0] * n.m[0][j] +
					m[i][1] * n.m[1][j] +
					m[i][2] * n.m[2][j] +
					m[i][3] * n.m[3][j];
			}
		}

		return *this;
	}

	operator const float*() const
	{
		return &(m[0][0]);
	}
	void setIdentity() {
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				if (i == j) m[i][j] = 1;
				else m[i][j] = 0;
			}
		}
	}


	friend ostream& operator<<(ostream& out, const mat4& m) {
		out<< "[ ";
		for (int i = 0; i < 4; ++i) {
			out<<"\t";
			for (int j = 0; j < 4; ++j) {
				out<< m.m[i][j] << " ";
			}
			out << endl;
		}
		out<<" ]"<< endl;
		return out;
	}
};
float Dot(const vec3& v1, const vec3& v2);
vec3 Cross(const vec3& v1, const vec3& v2);
mat4 Identity();
mat4 Translate(float x, float y, float z);
mat4 Translate(const vec3& v);
mat4 Scale(float x, float y, float z);
mat4 Scale(float s);
mat4 Perspective(float fov, float aspect, float z_near, float z_far);
mat4 Orthographic(float l, float r, float b, float t, float n, float f);
mat4 get_rotation_mat(float yaw, float pitch, float roll);
mat4 get_perspective_mat(float camera_distance, float camera_fov_x, float yxaspect);
mat4 View(const vec3& camera_pos, const vec3& target, const vec3& up);
mat4 Rotate(float angle_x, float angle_y, float angle_z);
float toRadian(float f);
float toDegree(float f);
void calculate_normals(const int* indices, const float* vertices, const int n_triangle, float* normals, bool inverse = false);
vec3 normalize_normal(const vec3& n);
vec3 convert_vec3b_to_vec3(const Vec3b& n);
Vec3b convert_vec3_to_vec3b(const vec3& n);

vec3 Clamp(vec3 v, float a = 0.0, float b = 10000);
Vec3b Clamp(Vec3b v);

float gaussian(float x, float mu, float sigma);
float sqr(float x, float alpha);

vec3 sphere2cartesian(float theta, float phi);