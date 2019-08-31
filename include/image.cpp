#include "image.h"
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

Texture Mask(const Texture & src, const Texture & mask)
{
	Texture res(src);

	for (int y = 0; y < res.height; ++y) {
		for (int x = 0; x < res.width; ++x) {
			vec3 valid = mask.getPixel(x, y);
			if (valid.x != 0 && valid.y != 0 && valid.z != 0) {
				res.setPixel(x, y, src.getPixel(x, y));
			}
			else {
				res.setPixel(x, y, vec3(0,0,0));
			}
		}

	}

	return res;
}

Texture UnionTexture(const Texture & background, const Texture & forground, const Texture & mask)
{
	int width = background.width, height = background.height;
	Texture res(width, height);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			if (mask.getPixel(x, y).x > 0) {
				res.setPixel(x, y, forground.getPixel(x, y));
			}
			else {
				res.setPixel(x, y, background.getPixel(x, y));
			}
		}
	}
	return res;
}

void Texture::initial(int width, int height)
{
	data = new vec3*[height];
	for (int i = 0; i < height; ++i)
		data[i] = new vec3[width];
}

Texture::Texture()
{
	width = 0;
	height = 0;
	data = nullptr;
}

Texture::Texture(const Texture & _tex)
	:width(_tex.width), height(_tex.height) {
	initial(width, height);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			data[y][x] = _tex.data[y][x];
		}
	}
}

Texture & Texture::operator=(const Texture & _tex)
{
	vec3** p_orig = data;
	int w_orig = width;
	int h_orig = height;

	width = _tex.width;
	height = _tex.height;
	initial(width, height);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			data[y][x] = _tex.data[y][x];
		}
	}

	clear(p_orig, w_orig, h_orig);
	return *this;
}

Texture::Texture(int width, int height)
	:width(width), height(height) {
	initial(width, height);
}

void Texture::clear(vec3** p,int w, int h) {
	if (p != nullptr) {
		for (int i = 0; i < h; ++i) {
			if (p[i] != nullptr) {
				delete[] p[i];
				p[i] = nullptr;
			}
		}
		delete[] p;
		p = nullptr;
	}
}
Texture::~Texture()
{
	clear(data, width, height);
}

Texture::Texture(const std::string & name)
{
	Mat img = imread(name.c_str(), IMREAD_COLOR);
	width = img.cols;
	height = img.rows;
	initial(width, height);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			Vec3b c = img.at<Vec3b>(y, x);
			data[y][x] = vec3(float(c[2]) / 255, float(c[1]) / 255, float(c[0]) / 255);
		}
	}
}

Texture::Texture(const Mat & img)
{
	width = img.cols;
	height = img.rows;
	initial(width, height);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			Vec3f c = img.at<Vec3f>(y, x);
			data[y][x] = vec3(float(c[2]), float(c[1]), float(c[0]));
		}
	}
}

vec3 Texture::getPixel(int x, int y) const
{
	if (x < 0) x = 0;
	if (x >= width) x = width - 1;
	if (y < 0) y = 0;
	if (y >= height) y = height - 1;
	return data[y][x];
}

vec3 Texture::getPixel(const vec2i & p) const
{
	return getPixel(p.x, p.y);
}

vec3 & Texture::getPixel(int x, int y)
{
	return data[y][x];
}

vec3 Texture::getPixelInterp(float u, float v) const
{
	float x = u * width;
	float y = v * height;

	int ux = int(x);
	int uy = int(y);

	//cout << ux << " " << uy << endl;
	vec3 c11 = getPixel(ux, uy);
	vec3 c12 = getPixel(ux, uy + 1);
	vec3 c21 = getPixel(ux + 1, uy);
	vec3 c22 = getPixel(ux + 1, uy + 1);

	float x1 = ux, x2 = ux + 1;
	float y1 = uy, y2 = uy + 1;

	float b1 = (x2 - x) * c11[2] + (x - x1) * c21[2];
	float b2 = (x2 - x) * c12[2] + (x - x1) * c22[2];
	float b = (y2 - y)* b1 + (y - y1) * b2;

	float g1 = (x2 - x) * c11[1] + (x - x1) * c21[1];
	float g2 = (x2 - x) * c12[1] + (x - x1) * c22[1];
	float g = (y2 - y)* g1 + (y - y1) * g2;

	float r1 = (x2 - x) * c11[0] + (x - x1) * c21[0];
	float r2 = (x2 - x) * c12[0] + (x - x1) * c22[0];
	float r = (y2 - y)* r1 + (y - y1) * r2;

	return c22;
}

vec3 Texture::getPixelInterp(const vec2 & p) const
{
	return getPixelInterp(p.x, p.y);
}

void Texture::setPixel(int x, int y, const vec3 & c)
{
	data[y][x] = c;
}

void Texture::setPixel(int x, int y, float r, float g, float b)
{
	data[y][x] = vec3(r, g, b);
}

Mat Texture::convert2opencv(int type) const
{
	Mat img(height, width, type);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			Vec3f c(data[y][x].z, data[y][x].y, data[y][x].x);
			img.at<Vec3f>(y, x) = c;
		}
	}
	return img;
}

void Texture::show(int scale) const
{
	Mat img(height, width, CV_8UC3);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			vec3 _tmp = data[y][x];
			_tmp = Clamp(_tmp, 0, 1);
			Vec3b c(_tmp.z * scale, _tmp.y * scale, _tmp.x * scale);
			img.at<Vec3b>(y, x) = c;
		}
	}
	namedWindow("window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("window", img);                   // Show our image inside it.
	cv::setWindowProperty("window", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	waitKey(0);                                          // Wait for a keystroke in the window
}

void Texture::save(const std::string & name) const
{
	Mat img(height, width, CV_8UC3);

	float scale = 255;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			vec3 _tmp = data[y][x];
			_tmp = Clamp(_tmp, 0, 1);
			Vec3b c(_tmp.z * scale, _tmp.y * scale, _tmp.x * scale);
			img.at<Vec3b>(y, x) = c;
		}
	}

	imwrite(name, img);
}

void Texture::load(const std::string & name)
{
	Mat img = imread(name.c_str(), IMREAD_COLOR);
	width = img.cols;
	height = img.rows;
	initial(width, height);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			Vec3b c = img.at<Vec3b>(y, x);
			data[y][x] = vec3(float(c[2]) / 255, float(c[1]) / 255, float(c[0]) / 255);
		}
	}
}

void Texture::saveHDR(const std::string & name) const
{
	EXRHeader header;
	InitEXRHeader(&header);

	EXRImage image;
	InitEXRImage(&image);
	image.num_channels = 3;

	std::vector<float> images[3];
	images[0].resize(width * height);
	images[1].resize(width * height);
	images[2].resize(width * height);

	
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			images[0][y * width + x] = data[y][x].R;
			images[1][y * width + x] = data[y][x].G;
			images[2][y * width + x] = data[y][x].B;
		}
	}

	float* image_ptr[3];
	image_ptr[0] = &(images[2].at(0)); // B
	image_ptr[1] = &(images[1].at(0)); // G
	image_ptr[2] = &(images[0].at(0)); // R

	image.images = (unsigned char**)image_ptr;
	image.width = width;
	image.height = height;

	header.num_channels = 3;
	header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
	// Must be BGR(A) order, since most of EXR viewers expect this channel order.
	strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
	strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
	strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

	header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
	header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
	for (int i = 0; i < header.num_channels; i++) {
		header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
		header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
	}

	const char* err;
	int ret = SaveEXRImageToFile(&image, &header, name.c_str(), &err);

	free(header.channels);
	free(header.pixel_types);
	free(header.requested_pixel_types);

}

void Texture::loadHDR(const std::string & name)
{
	float* out; // width * height * RGBA
	const char* err;
	int ret = LoadEXR(&out, &width, &height, name.c_str(), &err);
	initial(width, height);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = y * width + x;
			float R = out[4 * idx];
			float G = out[4 * idx + 1];
			float B = out[4 * idx + 2];
			
			data[y][x] = vec3(R, G, B);
		}
	}
	free(out);
}

void Texture::average_blur()
{
	Texture _tmp(*this);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {

			int count = 0;
			vec3 avg(0, 0, 0);
			for (int i = -2; i <= 2; ++i) {
				for (int j = -2; j <= 2; ++j) {
					if (y + i < 0 || y + i > height - 1 || x + j < 0 || x + j > width - 1) continue;
					count++;
					avg = avg + data[y + i][x + j];
				}
			}
			avg = avg / count;
			_tmp.data[y][x] = avg;
		}
	}
	*this = _tmp;
}

Texture Texture::resize(float scale)
{
	Texture res(width * scale, height * scale);

	// down sampling
	if (scale < 1.0) {
		for (int i = 0; i < res.width; ++i) {
			for (int j = 0; j < res.height; ++j) {
				// filter
				vec3 color(0, 0, 0);
				int x1 = float(i) / scale;
				int y1 = float(j) / scale;
				int x2 = float(i + 1) / scale;
				int y2 = float(j + 1) / scale;
				int count = 0;
				for (int x = x1; x < x2; ++x) {
					for (int y = y1; y < y2; ++y) {
						color = color + data[y][x];
						count++;
					}
				}
				color = color / count;
				res.data[j][i] = color;
			}
		}
	}
	// up sampling
	else {
		for (int i = 0; i < res.width; ++i) {
			for (int j = 0; j < res.height; ++j) {
				vec3 color(0, 0, 0);
				float x = i / scale;
				float y = j / scale;
				// bilinear filter

				int x1 = int(x);
				int y1 = int(y);
				int x2 = Clamp(x1 + 1, 0, width - 1);
				int y2 = Clamp(y1 + 1, 0, height - 1);

				vec3 c11 = data[y1][x1];
				vec3 c12 = data[y2][x1];
				vec3 c21 = data[y1][x2];
				vec3 c22 = data[y2][x2];
				//cout << x << " " << y << " " << i << " " << j << endl;

				vec3 t1, t2;
				if (x2 == x1) {
					t1 = c11;
					t2 = c12;
				}
				else {
					t1 = (x2 - x) / (x2 - x1) * c11 + (x - x1) / (x2 - x1) * c21;
					t2 = (x2 - x) / (x2 - x1) * c12 + (x - x1) / (x2 - x1) * c22;
				}
				if (y1 == y2) {
					color = t1;
				}
				else 
					color = (y2 - y) / (y2 - y1) * t1 + (y - y1) / (y2 - y1) * t2;
				res.data[j][i] = color;
			}
		}
	}
	return res;
}

Texture Texture::resize2(float scale)
{
	Mat src = convert2opencv(CV_32FC3);
	Mat dst;
	cv::resize(src, dst, cv::Size(0,0), scale, scale, CV_INTER_LANCZOS4);
	return Texture(dst);
}

Texture Texture::guideFilter(Texture ref)
{
	return Texture();
}

Image::Image()
{
}

Image::Image(const std::string & name)
{
	img = imread(name.c_str(), IMREAD_COLOR);
	if (img.type() == 16) {
		type = CV_8UC3;
	}
	else if (img.type() == 24) {
		type = CV_8UC4;
	}
	width = img.cols;
	height = img.rows;
}

Image::Image(int rows, int cols, int _type)
{
	img = Mat(rows, cols, _type);
	type = _type;
	width = img.cols;
	height = img.rows;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; ++x) {
			img.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
		}
	}
}

Image Image::resize(float sx, float sy)
{
	Image res(width * sx, height * sy);
	cv::resize(img, res.img, cv::Size(0, 0), sx, sy);
	return res;
}

Vec3b Image::getPixel(int x, int y) const
{
	if (type == CV_8UC3) {
		return img.at<Vec3b>(y, x);
	}
	else if (type == CV_8UC4) {
		Vec4b tmp = img.at<Vec4b>(y, x);
		return Vec3b(tmp[0], tmp[1], tmp[2]);
	}
}

Vec3b Image::getPixel(const vec2i & p) const
{
	return getPixel(p.x, p.y);
}

void Image::setPixel(int x, int y, const Vec3b & c)
{
	setPixel(x, y, c[2], c[1], c[0]);
}

void Image::setPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
	if (type == CV_8UC3)
		img.at<Vec3b>(y, x) = Vec3b(b, g, r);
	else if (type == CV_8UC4) {
		img.at<Vec4b>(y, x) = Vec4b(b, g, r, 1.0f);
	}
}

void Image::show() const
{
	namedWindow("window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("window", img);                   // Show our image inside it.
	cv::setWindowProperty("window", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	waitKey(0);                                          // Wait for a keystroke in the window
}

void Image::save(const std::string & name) const
{
	Mat tmp = Mat(img.rows, img.cols, CV_8UC3);
	for (int y = 0; y < img.rows; ++y) {
		for (int x = 0; x < img.cols; ++x) {
			tmp.at<Vec3b>(y, x) = getPixel(x, y);
		}
	}
	imwrite(name, tmp);
}
