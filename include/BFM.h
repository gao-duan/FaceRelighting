#pragma once
#include "common.h"
#include "math_vec.h"
#include <functional>
#include "image.h"

using namespace std::placeholders;
class Landmarks3D;
class PoseOptimizer;

class BFM {
public:
	// Shape, Texture and Expression
	vector<float> shapeEV, shapeMU, shapePC, texEV, texMU, texPC, expEV, expMU, expPC;
	enum class Attribute {AGE, GENDER, HEIGHT, WEIGHT};
	
	// Attributes: age, gender, height and weight
	vector<float> age_shape, age_tex;
	vector<float> gender_shape, gender_tex;
	vector<float> height_shape, height_tex;
	vector<float> weight_shape, weight_tex;

	vector<int> tl;
	vector<int> segments;
	
	MatrixXf m_tex_pc;
	vector<float> _texPC;
	vector<float> alpha, beta, gamma;

	vector<float> vertices, colors, normals;
	vector<int> landmark_indices;
	vector<vec2> uvs;

	// bbox
	vec3 pMin, pMax, center;
	
	int n_vertex, n_triangle, n_parameter;
	int n_segments;

	bool load(const string& name, const string& exp_name, const string attribute_name = "");
	bool load_attribute(const string& name);

	bool set_landmark_indices(const vector<int>& lm_idx);

	bool initial_mean();
	bool initial_random();

	void update();
	void update_vertices();

	void update_normals();

	void calculate_bbox();

	vec3 get_vertex(int idx,int n=-1) const;
	vec3 get_vertex(int face_idx, float u, float v, int n = -1) const;
	vec3 get_color(int idx, int n = -1) const;
	vec3 get_color(int face_idx, float u, float v, int n = -1) const;
	int get_seg(int idx) const;
	int get_seg(int face_idx, float u, float v) const;
	vec3 get_normal(int idx) const;
	vec3 get_normal(int face_idx, float u, float v) const;
	bool write_to_ply(const string&name, bool inverse=false);
	map<int ,set<int> > get_neighbors();

	void update_by_pose_optimize(std::function<vec3(const vec3&)> f);
	void deformation(const Landmarks3D& lm3d, float weight = 1.0f);

	void applyAttributes(Attribute attrib, float factor);
	void update_color_by_uvmap(const Texture& uvmap);
};
vec2 pos2uv(const vec3& pos);

struct LaplacianCoord {
	vec3 pos;
	vec3 laplacian;
	vector<int> neighbor_indices;
	vector<float> neightbor_weights;
};