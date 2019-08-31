#include "specular_removal_grad.h"
#include "landmark.h"

static int coord2idx(int x, int y, int width) {
	return y * width + x;
}

static vec2i idx2coord(int idx, int width) {
	return vec2i(idx % width, idx / width);
}

Texture SpecularRemoval::solve(float weight)
{
	int width = old_img.width, height = old_img.height;


	int n = width * height;
	int k = valid_points.size();

	vector<int> valid_points_indices;
	for (auto pos : valid_points) {
		valid_points_indices.push_back(coord2idx(pos.x, pos.y, width));
	}

	SparseMatrix<double> A(n * 2 + k, n);
	MatrixXd b(n * 2 + k, 3);

	MatrixXd vpos(n, 3);

	for (int i = 0; i < n; ++i) {
		vec2i pos = idx2coord(i, width);
		vec3 color = old_img.getPixel(pos);
		vpos(i, 0) = color.R;
		vpos(i, 1) = color.G;
		vpos(i, 2) = color.B;

	}
	vector<int> I;
	vector<int> J;
	vector<double> V;

	for (int i = 0; i < n; ++i) {
		vec2i p = idx2coord(i, width);
	
		int idx = 2 * i;
		if (p.x - 1 >= 0) {
			I.push_back(idx);
			J.push_back(coord2idx(p.x-1,p.y,width));
			V.push_back(-1);
		}
		else {
			I.push_back(idx);
			J.push_back(coord2idx(p.x, p.y, width));
			V.push_back(-1);
		}
		if (p.x + 1 <width) {
			I.push_back(idx);
			J.push_back(coord2idx(p.x + 1, p.y, width));
			V.push_back(1);
		}
		else {
			I.push_back(idx);
			J.push_back(coord2idx(p.x, p.y, width));
			V.push_back(1);
		}

		if (p.y - 1 >= 0) {
			I.push_back(idx + 1);
			J.push_back(coord2idx(p.x, p.y - 1, width));
			V.push_back(-1);
		}
		else {
			I.push_back(idx + 1);
			J.push_back(coord2idx(p.x, p.y, width));
			V.push_back(-1);
		}
		if (p.y + 1 < height) {
			I.push_back(idx + 1);
			J.push_back(coord2idx(p.x, p.y + 1, width));
			V.push_back(1);
		}
		else {
			I.push_back(idx + 1);
			J.push_back(coord2idx(p.x, p.y, width));
			V.push_back(1);
		}
	}
	for (int i = 0; i < k; ++i) {
		I.push_back(2 * n + i);
		J.push_back(valid_points_indices[i]);
		vec2i p = valid_points[i];

		if (eye_mouth_matte.getPixel(p).R > 0.1) {
			V.push_back(0.05);
		}
		else V.push_back(weight);
	}

	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	tripletList.reserve(I.size());

	for (int idx = 0; idx < I.size(); ++idx) {
		tripletList.push_back(T(I[idx], J[idx], V[idx]));
	}
	A.setFromTriplets(tripletList.begin(), tripletList.end());

	b = A * vpos;

	for (int i = 0; i < k; ++i) {
		vec3 color = new_img.getPixel(valid_points[i]);	
		vec2i p = valid_points[i];

		vec3 c = old_img.getPixel(p);
		float lumiance = (c.R + c.G + c.B) / 3.0;
		if (eye_mouth_matte.getPixel(p).R > 0.1) {
			b(n * 2 + i, 0) = 0.05 * color.R;
			b(n * 2 + i, 1) = 0.05 * color.G;
			b(n * 2 + i, 2) = 0.05 * color.B;
		}
		else {
			b(n * 2 + i, 0) = weight * color.R;
			b(n * 2 + i, 1) = weight * color.G;
			b(n * 2 + i, 2) = weight * color.B;
		}
	}

	VectorXd x(n, 1);
	VectorXd guess[3];
	for (int j = 0; j < 3; j++) {
		guess[j] = VectorXd(n, 1);
		for (int i = 0; i < n; ++i) {
			vec2i pos = idx2coord(i, width);
			vec3 color = new_img.getPixel(pos);
			guess[j](i, 0) = color[j];
		}
	}

	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> solver;
	Eigen::SparseMatrix<double> _A = A.transpose() * A;

	solver.compute(_A);

	if (solver.info() != Eigen::Success)
	{
		std::cerr << "Waring: Eigen decomposition failed" << std::endl;
	}
	solver.setMaxIterations(100);
	
	Texture res(width, height);
	for (int i = 0; i < 3; ++i) {
		Eigen::MatrixXd _b = A.transpose() * b.col(i);
		x = solver.solveWithGuess(_b, guess[i]);
	
		for (int j = 0; j < n; ++j) {
			vec2i pos = idx2coord(j, width);
			vec3 color = res.getPixel(pos);
			color[i] = x(j, 0);
			res.setPixel(pos.x, pos.y, color);
		}
	}
	
	return res;
}

Texture specular_removal(const Texture & old_img, const Texture & new_img, const Texture & matte, const std::vector<cv::Point>& points, float weight)
{
	vector<vec2i> valid_points = get_valid_points_from_matte(matte);
	Texture mouth_mask = generate_eye_mouth_matte_from_landmarks(points);

	SpecularRemoval opt(old_img, new_img, mouth_mask, valid_points);
	Texture res = opt.solve(weight);
	return res;
}
