#ifndef MULTINOMIALSAMPLER_H_
#define MULTINOMIALSAMPLER_H_

#include <random>

#include "randgen.h"

class MultinomialSampler
{
	static const unsigned int kDefMaxVal = 4000000000u;

public:
	MultinomialSampler() : uint_dist(0, kDefMaxVal) {}
	MultinomialSampler(int *weights, int len);
	~MultinomialSampler();

	void Init(int *weights, int len);
	void Init(unsigned short *weights, int len);

	int Sample(std::default_random_engine &generator);
	int Sample(RandGen &rand_gen);

private:
	unsigned int *intervals_ = 0;
	int num_vals = 0;
	std::uniform_int_distribution<unsigned int> uint_dist;
};

#endif
