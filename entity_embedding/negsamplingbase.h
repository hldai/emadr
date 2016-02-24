#ifndef NEGSAMPLINGBASE_H_
#define NEGSAMPLINGBASE_H_

#include "exp_table.h"

#include <random>

class NegSamplingBase
{
public:
	static float **GetInitedVecs0(int num_objs, int vec_dim);
	static void InitVec0Def(float *vecs, int vec_dim);
	static float **GetInitedVecs1(int num_objs, int vec_dim);

	static float *GetDefNegativeSamplingWeights(int *obj_cnts, int num_objs);

public:
	NegSamplingBase(ExpTable *exp_table, int num_negative_samples) 
		: exp_table_(exp_table), num_negative_samples_(num_negative_samples) {}

protected:
	void loadFreqFile(const char *freq_file, int &num_objs,
		std::discrete_distribution<int> &negative_sample_dist);
	void initNegativeSamplingDist(int num_objs, int *obj_cnts,
		std::discrete_distribution<int> &negative_sample_dist);

protected:
	ExpTable *exp_table_;
	int num_negative_samples_ = 0;
};

#endif
