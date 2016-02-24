#include "negsamplingdoubleobj.h"

NegSamplingDoubleObj::NegSamplingDoubleObj(ExpTable *exp_table, int num_negative_samples,
	const char *freq_file0, const char *freq_file1) : NegSamplingBase(exp_table, num_negative_samples),
	num_negative_samples_(num_negative_samples)
{
	loadFreqFile(freq_file0, num_objs0_, negative_sample_dist0_);
	loadFreqFile(freq_file1, num_objs1_, negative_sample_dist1_);
}

NegSamplingDoubleObj::~NegSamplingDoubleObj()
{
}

void NegSamplingDoubleObj::TrainEdge(int dim0, int dim1, float *vec_in, int obj_out0, float **vecs_out0, int obj_out1,
	float **vecs_out1, float alpha, float *tmp_neu1e, std::default_random_engine &generator,
	bool update_in, bool update_out)
{
	int dim = dim0 + dim1;
	for (int i = 0; i < dim; ++i)
		tmp_neu1e[i] = 0.0f;

	const float lambda = alpha * 0.01f;
	int target0 = obj_out0, target1 = obj_out1;
	int label = 1;
	for (int i = 0; i < num_negative_samples_ + 1; ++i)
	{
		if (i != 0)
		{
			target0 = negative_sample_dist0_(generator);
			if (target0 == obj_out0) continue;

			target1 = negative_sample_dist1_(generator);
			if (target1 == obj_out1) continue;

			label = 0;
		}

		float dot_product = 0;
		if (target0 > -1)
			for (int i = 0; i < dim0; ++i)
				dot_product += vec_in[i] * vecs_out0[target0][i];
		if (target1 > -1)
			for (int i = 0; i < dim1; ++i)
				dot_product += vec_in[i + dim0] * vecs_out1[target1][i];

		float g = (label - exp_table_->getSigmaValue(dot_product)) * alpha;

		if (target0 > -1)
			for (int j = 0; j < dim0; ++j)
				tmp_neu1e[j] += g * vecs_out0[target0][j];
		if (target1 > -1)
			for (int j = 0; j < dim1; ++j)
				tmp_neu1e[j + dim0] += g * vecs_out1[target1][j];

		if (update_out)
		{
			if (target0 > -1)
				for (int j = 0; j < dim0; ++j)
					vecs_out0[target0][j] += g * vec_in[j] - lambda * vecs_out0[target0][j];
			if (target1 > -1)
				for (int j = 0; j < dim1; ++j)
					vecs_out1[target1][j] += g * vec_in[j + dim0] - lambda * vecs_out1[target1][j];
		}
	}

	if (update_in)
		for (int j = 0; j < dim; ++j)
			vec_in[j] += tmp_neu1e[j] - lambda * vec_in[j];
}

