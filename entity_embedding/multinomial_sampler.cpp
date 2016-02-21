#include "multinomial_sampler.h"

MultinomialSampler::MultinomialSampler(int *weights, int len) : uint_dist(0, kDefMaxVal)
{
	Init(weights, len);
}

MultinomialSampler::~MultinomialSampler()
{
	if (intervals_ != 0)
		delete[] intervals_;
}

void MultinomialSampler::Init(int *weights, int len)
{
	num_vals = len;

	int sum_weights = 0;
	for (int i = 0; i < len; ++i)
		sum_weights += weights[i];

	intervals_ = new unsigned int[num_vals];
	int cur_weight_sum = 0;
	for (int i = 0; i < len; ++i)
	{
		cur_weight_sum += weights[i];
		intervals_[i] = (unsigned int)((double)cur_weight_sum / sum_weights * kDefMaxVal);
	}
}

void MultinomialSampler::Init(unsigned short *weights, int len)
{
	num_vals = len;

	int sum_weights = 0;
	for (int i = 0; i < len; ++i)
		sum_weights += weights[i];

	intervals_ = new unsigned int[num_vals];
	int cur_weight_sum = 0;
	for (int i = 0; i < len; ++i)
	{
		cur_weight_sum += weights[i];
		intervals_[i] = (unsigned int)((double)cur_weight_sum / sum_weights * kDefMaxVal);
	}
}

int MultinomialSampler::Sample(std::default_random_engine &generator)
{
	if (intervals_ == 0)
		return -1;

	unsigned int val = uint_dist(generator);
	int l = 0, r = num_vals - 1, m;
	while (l <= r)
	{
		m = (l + r) >> 1;
		if (intervals_[m] > val)
			r = m - 1;
		else
			l = m + 1;
	}

	if (l >= num_vals)
		l = num_vals - 1;
	return l;
}

int MultinomialSampler::Sample(RandGen &rand_gen)
{
	if (intervals_ == 0)
		return -1;

	//unsigned int val = uint_dist(generator);
	unsigned int val = rand_gen.NextRandom() % kDefMaxVal;
	int l = 0, r = num_vals - 1, m;
	while (l <= r)
	{
		m = (l + r) >> 1;
		if (intervals_[m] > val)
			r = m - 1;
		else
			l = m + 1;
	}

	if (l >= num_vals)
		l = num_vals - 1;
	return l;
}
