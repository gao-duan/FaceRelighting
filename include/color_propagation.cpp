#include "color_propagation.h"

static int coord2idx(int x, int y, int width) {
	return y * width + x;
}

static vec2i idx2coord(int idx, int width) {
	return vec2i(idx % width, idx / width);
}

Texture ColorPropagaton::solve(float weight)
{
	int width = old_img.width, height = old_img.height;

	int n = width * height;
	int k = valid_points.size();

	vector<int> valid_points_indices;
	for (auto pos : valid_points) {
		valid_points_indices.push_back(coord2idx(pos.x, pos.y, width));
	}

	SparseMatrix<float> A(n + k, n);
	MatrixXf b(n + k, 3);

	MatrixXf vpos(n, 3);
	for (int i = 0; i < n; ++i) {
		vec2i pos = idx2coord(i, width);
		vec3 color = old_img.getPixel(pos);
		vpos(i, 0) = color.R;
		vpos(i, 1) = color.G;
		vpos(i, 2) = color.B;
	}

	vector<int> I;
	vector<int> J;
	vector<float> V;
	map<int, set<int> > all_neighbors;

	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < height; ++j) {
			int idx = coord2idx(i, j, width);

			if (matte_img.getPixel(i, j).R < 0.01) {
				for (int x = -3; x <= 3; ++x) {
					for (int y = -3; y <= 3; ++y) {
						if (i + x < width && i + x >= 0 && y + j < height && y + j > 0)
							all_neighbors[idx].insert(coord2idx(i + x, j + y, width));
					}
				}
			}
			else {
				for (int x = -1; x <= 1; ++x) {
					for (int y = -1; y <= 1; ++y) {
						if (i + x < width && i + x >= 0 && y + j < height && y + j > 0)
							all_neighbors[idx].insert(coord2idx(i + x, j + y, width));
					}
				}
			}
		}
	}

	for (int i = 0; i < n; ++i) {
		set<int>& neighbor_indices = all_neighbors[i];
		int z = neighbor_indices.size();

		for (int j = 0; j < z + 1; ++j) I.push_back(i);
		for (auto _idx : neighbor_indices) J.push_back(_idx);
		J.push_back(i);
		for (int j = 0; j < z; ++j) V.push_back(-1);
		V.push_back(z);
	}



	for (int i = 0; i < k; ++i) {
		I.push_back(n + i);
		J.push_back(valid_points_indices[i]);
		vec2i p = valid_points[i];
		
		V.push_back(weight);
	}

	typedef Eigen::Triplet<float> T;
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

		b(n + i, 0) = weight * color.R;
		b(n + i, 1) = weight * color.G;
		b(n + i, 2) = weight * color.B;
	}

	VectorXf x(n, 1);
	VectorXf guess[3];
	for (int j = 0; j < 3; j++) {
		guess[j] = VectorXf(n, 1);
		for (int i = 0; i < n; ++i) {
			vec2i pos = idx2coord(i, width);
			vec3 color = old_img.getPixel(pos);
			guess[j](i, 0) = color[j];
		}
	}

	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> solver;

	Eigen::SparseMatrix<float> _A = A.transpose() * A;
	solver.compute(_A);

	if (solver.info() != Eigen::Success)
	{
		std::cerr << "Waring: Eigen decomposition failed" << std::endl;
	}
	solver.setMaxIterations(500);
	solver.setTolerance(1e-7);

	Texture res(width, height);
	for (int i = 0; i < 3; ++i) {
		Eigen::MatrixXf _b = A.transpose() * b.col(i);

		x = solver.solveWithGuess(_b, guess[i]);

		for (int j = 0; j < n; ++j) {
			vec2i pos = idx2coord(j, width);
			vec3 color = res.getPixel(pos);
			color[i] = x(j, 0);
			res.setPixel(pos.x, pos.y, color);
		}
	}

	for (auto p : valid_points) {
		res.setPixel(p.x, p.y, new_img.getPixel(p));
	}
	
	return res;
}

/*




*/