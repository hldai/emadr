#include "negative_sampling_trainer.h"

#include <cmath>

#include "math_utils.h"

float **NegativeSamplingTrainer::GetInitedVecs0(int num_objs, int vec_dim)
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

void NegativeSamplingTrainer::InitVec0Def(float *vecs, int vec_dim)
{
	for (int i = 0; i < vec_dim; ++i)
		vecs[i] = ((float)rand() / RAND_MAX - 0.5f) / vec_dim;
}

float **NegativeSamplingTrainer::GetInitedVecs1(int num_objs, int vec_dim)
{
	float **vecs = new float*[num_objs];
	for (int i = 0; i < num_objs; ++i)
	{
		vecs[i] = new float[vec_dim];
		std::fill(vecs[i], vecs[i] + vec_dim, 0.0f);
	}
	return vecs;
}

double *NegativeSamplingTrainer::GetDefNegativeSamplingWeights(int *obj_cnts, int num_objs)
{
	double *weights = new double[num_objs];
	for (int i = 0; i < num_objs; ++i)
		weights[i] = pow(obj_cnts[i], 0.75);
	return weights;
}

NegativeSamplingTrainer::NegativeSamplingTrainer(ExpTable *exp_table, int vec_dim, int num_objs1,
	int num_negative_samples, std::discrete_distribution<int> *negative_sample_dist) : exp_table_(exp_table), vec_dim_(vec_dim),
	num_objs1_(num_objs1), num_negative_samples_(num_negative_samples),
	negative_sample_dist_(negative_sample_dist)
{
}

NegativeSamplingTrainer::~NegativeSamplingTrainer()
{
}

void NegativeSamplingTrainer::TrainPrediction(float *vec0, int obj1, float **vecs1, float alpha, float *tmp_neu1e,
	std::default_random_engine &generator, bool update0, bool update1)
{
	for (int i = 0; i < vec_dim_; ++i)
		tmp_neu1e[i] = 0.0f;

	int target = obj1;
	int label = 1;
	for (int i = 0; i < num_negative_samples_ + 1; ++i)
	{
		if (i != 0)
		{
			target = (*negative_sample_dist_)(generator);
			if (target == obj1) continue;

			label = 0;
		}

		float dot_product = MathUtils::DotProduct(vec0, vecs1[target], vec_dim_);
		float g = (label - exp_table_->getSigmaValue(dot_product)) * alpha;

		for (int j = 0; j < vec_dim_; ++j)
			tmp_neu1e[j] += g * vecs1[target][j];
		if (update1)
		{
			for (int j = 0; j < vec_dim_; ++j)
				vecs1[target][j] += g * vec0[j];
		}
	}

	if (update0)
	{
		for (int j = 0; j < vec_dim_; ++j)
			vec0[j] += tmp_neu1e[j];
	}
}

void NegativeSamplingTrainer::CheckObject(float *cur_vec, float **vecs1)
{
	//const int max_num_edges = 100;
	//int vv[max_num_edges], weights[max_num_edges];
	//int ecnt = 0;
	//for (int i = 0; i < num_edges_ && edges_[i].va <= entity_index; ++i)
	//{
	//	if (edges_[i].va == entity_index)
	//	{
	//		vv[ecnt] = edges_[i].vb;
	//		weights[ecnt++] = weights_[i];
	//	}
	//	if (edges_[i].vb == entity_index)
	//	{
	//		vv[ecnt] = edges_[i].va;
	//		weights[ecnt++] = weights_[i];
	//	}
	//}
	//float target_val = 0;
	//for (int i = 0; i < ecnt; ++i)
	//{
	//	float dp = MathUtils::DotProduct(entity_vec, syn1_[vv[i]], vec_dim_);
	//	printf("%d\t%d\t%f\n", vv[i], weights[i], dp);
	//	target_val += weights[i] * MathUtils::DotProduct(entity_vec, syn1_[vv[i]], vec_dim_);
	//}

	const int k = 10;
	int top_indices[k];
	float vals[k];
	std::fill(vals, vals + k, -1e4f);
	float target_sum = 0;
	for (int i = 0; i < num_objs1_; ++i)
	{
		float dp = MathUtils::DotProduct(cur_vec, vecs1[i], vec_dim_);
		target_sum += exp(dp);

		int pos = k - 1;
		while (pos > -1 && vals[pos] < dp) --pos;
		for (int j = k - 2; j > pos; --j)
		{
			vals[j + 1] = vals[j];
			top_indices[j + 1] = top_indices[j];
		}
		if (pos != k - 1)
		{
			vals[pos + 1] = dp;
			top_indices[pos + 1] = i;
		}
	}

	//printf("%f %f %f\n", target_val, target_sum, target_val / log(target_sum));

	for (int i = 0; i < k; ++i)
	{
		printf("%d\t%f\n", top_indices[i], vals[i]);
	}
}

void NegativeSamplingTrainer::CloseVectors(float **vecs, int num_vecs, int vec_dim, int idx)
{
	const int k = 10;
	int top_indices[k];
	float vals[k];
	std::fill(vals, vals + k, -1e4f);

	float *vec = vecs[idx];
	for (int i = 0; i < num_vecs; ++i)
	{
		if (i == idx)
			continue;

		float sim = MathUtils::Cosine(vec, vecs[i], vec_dim);

		int pos = k - 1;
		while (pos > -1 && vals[pos] < sim) --pos;
		for (int j = k - 2; j > pos; --j)
		{
			vals[j + 1] = vals[j];
			top_indices[j + 1] = top_indices[j];
		}
		if (pos != k - 1)
		{
			vals[pos + 1] = sim;
			top_indices[pos + 1] = i;
		}
	}

	//printf("%f %f %f\n", target_val, target_sum, target_val / log(target_sum));

	for (int i = 0; i < k; ++i)
	{
		printf("%d\t%f\n", top_indices[i], vals[i]);
	}
}
