#pragma once
#include "common.h"
#include "math_vec.h"

struct Image {
	Mat img;
	int width, height;
	int type;
	Image();

	Image(const std::string& name);

	Image(int rows, int cols, int _type = CV_8UC3);

	Image resize(float sx, float sy);

	Vec3b getPixel(int x, int y) const;

	Vec3b getPixel(const vec2i& p) const;

	void setPixel(int x, int y, const Vec3b& c);

	void setPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b);

	void show() const;

	void save(const std::string& name) const;

};

struct Texture {
private:
	vec3** data;
	void initial(int width, int height);
	void clear(vec3**,int w, int h);
public:
	int width, height;
	Texture();

	Texture(const Texture& _tex);

	Texture& operator=(const Texture& _tex);

	Texture(int width, int height);

	~Texture();

	Texture(const std::string& name);

	Texture(const Mat& img);

	vec3 getPixel(int x, int y) const;

	vec3 getPixel(const vec2i& p) const;
	vec3& getPixel(int x, int y);

	vec3 getPixelInterp(float u, float v) const;
	vec3 getPixelInterp(const vec2& p) const;

	void setPixel(int x, int y, const vec3& c);
	void setPixel(int x, int y, float r, float g, float b);
	Mat convert2opencv(int type = CV_8UC3)const;

	void show(int scale = 255.0) const;

	void save(const std::string& name) const;

	void load(const std::string& name);
	void saveHDR(const std::string& name) const;

	void loadHDR(const std::string& name);

	void average_blur();
	Texture resize(float scale);
	Texture resize2(float scale);

	Texture guideFilter(Texture ref);
};

Texture Mask(const Texture& src, const Texture& mask);
//Texture textureComplete(const Texture& src);
Texture UnionTexture(const Texture& background, const Texture& forground, const Texture& mask);