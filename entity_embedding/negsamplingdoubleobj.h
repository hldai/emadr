#ifndef NEGSAMPLINGDOUBLEOBJ_H_
#define NEGSAMPLINGDOUBLEOBJ_H_

#include <random>

#include "exp_table.h"
#include "negsamplingbase.h"

class NegSamplingDoubleObj : public NegSamplingBase
{
public:
	NegSamplingDoubleObj(ExpTable *exp_table, int num_negative_samples,
		const char *freq_file0, const char *freq_file1);
	~NegSamplingDoubleObj();

	void TrainEdge(int dim0, int dim1, float *vec_in, int obj_out0, float **vecs_out0, int obj_out1,
		float **vecs_out1, float alpha, float *tmp_neu1e, std::default_random_engine &generator,
		bool update_in = true, bool update_out = true);

private:
	int num_objs0_ = 0;
	int num_objs1_ = 0;

	int num_negative_samples_ = 0;
	std::discrete_distribution<int> negative_sample_dist0_;
	std::discrete_distribution<int> negative_sample_dist1_;
};

#endif
