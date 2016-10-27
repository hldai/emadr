#include "negsamplingbase.h"

#include <cassert>

float **NegSamplingBase::GetInitedVecs0(int num_objs, int vec_dim)
{
	float **vecs = new float*[num_objs];
	for (int i = 0; i < num_objs; ++i)
	{
		vecs[i] = new float[vec_dim];
		for (int j = 0; j < vec_dim; ++j)
			vecs[i][j] = ((float)rand() / RAND_MAX - 0.5f) / vec_dim;
	}
	return vecs;
}

void NegSamplingBase::InitVec0Def(float *vecs, int vec_dim)
{
	for (int i = 0; i < vec_dim; ++i)
		vecs[i] = ((float)rand() / RAND_MAX - 0.5f) / vec_dim;
}

float **NegSamplingBase::GetInitedVecs1(int num_objs, int vec_dim)
{
	float **vecs = new float*[num_objs];
	for (int i = 0; i < num_objs; ++i)
	{
		vecs[i] = new float[vec_dim];
		std::fill(vecs[i], vecs[i] + vec_dim, 0.0f);
	}
	return vecs;
}

float *NegSamplingBase::GetDefNegativeSamplingWeights(int *obj_cnts, int num_objs)
{
	float *weights = new float[num_objs];
	for (int i = 0; i < num_objs; ++i)
		weights[i] = powf(obj_cnts[i], 0.75f);
	return weights;
}

void NegSamplingBase::loadFreqFile(const char *freq_file, int &num_objs,
	std::discrete_distribution<int> &negative_sample_dist)
{
	FILE *fp = fopen(freq_file, "rb");
	assert(fp != 0);

	fread(&num_objs, 4, 1, fp);
	int *cnts = new int[num_objs];
	fread(cnts, 4, num_objs, fp);
	fclose(fp);

	initNegativeSamplingDist(num_objs, cnts, negative_sample_dist);
}

void NegSamplingBase::initNegativeSamplingDist(int num_objs, int *obj_cnts, 
	std::discrete_distribution<int> &negative_sample_dist)
{
	float *weights = GetDefNegativeSamplingWeights(obj_cnts, num_objs);
	negative_sample_dist = std::discrete_distribution<int>(weights, weights + num_objs);
	delete[] weights;
}
