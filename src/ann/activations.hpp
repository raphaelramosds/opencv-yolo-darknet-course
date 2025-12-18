#pragma once

#include <cmath>

static inline float stepfunc_activate(float x, float offset)
{
    return (x >= offset) ? 1.0f : 0.0f;
}

static inline float logistic_activate(float x)
{
	return 1.0f / (1.0f + std::exp(-x));
}