#pragma once
#include "common.h"
#include "image.h"

struct Parameter {
	float value;
	float lr;
	Parameter() :value(0), lr(0.15) {}
	void update(float v) {
		value += v * lr;
	}
};