#pragma once

#include "image.h"

class ColorPropagaton {
	Texture old_img;
	Texture new_img;
	vector<vec2i> valid_points;
	
public:
	ColorPropagaton(const Texture& _old_img, const Texture& _new_img, const Texture& _matte, const vector<vec2i>& _valid_points) :
		old_img(_old_img), new_img(_new_img), matte_img(_matte), valid_points(_valid_points) {
	
	
		map<int, vector<int> > points;
		for (auto p : valid_points) {
			points[p.y].push_back(p.x);
		}
		valid_points.clear();
		for (auto p_row = points.begin(); p_row != points.end();++p_row) {
			int y = p_row->first;
			vector<int> x = p_row->second;
			std::sort(x.begin(), x.end());
			for (int i = 3; i < int(x.size()) - 3; ++i) {
				valid_points.push_back(vec2i(x[i], y));
			}
		}
		
	}
	Texture solve(float weight = 1.0);
	
	Texture matte_img;
};