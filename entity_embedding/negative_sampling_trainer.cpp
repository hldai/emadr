#include "negative_sampling_trainer.h"

#include <cassert>
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

float *NegativeSamplingTrainer::GetInitedCMParams(int vec_dim)
{
	float *cm_params = new float[vec_dim];
	for (int i = 0; i < vec_dim; ++i)
		cm_params[i] = (float)rand() / RAND_MAX;
	return cm_params;
}

void NegativeSamplingTrainer::InitMatrix(float *matrix, int dim0, int dim1)
{
	std::default_random_engine generator(1217);
	float rand_val = 4.0f * sqrtf(6.0f / (dim0 + dim1));
	std::uniform_real_distribution<float> distribution(-rand_val, rand_val);
	for (int i = 0; i < dim0 * dim1; ++i)
		matrix[i] = distribution(generator);
}

float *NegativeSamplingTrainer::GetDefNegativeSamplingWeights(int *obj_cnts, int num_objs)
{
	float *weights = new float[num_objs];
	for (int i = 0; i < num_objs; ++i)
		weights[i] = powf(obj_cnts[i], 0.75f);
	return weights;
}

NegativeSamplingTrainer::NegativeSamplingTrainer(ExpTable *exp_table, int num_objs1,
	int num_negative_samples, std::discrete_distribution<int> *negative_sample_dist) : exp_table_(exp_table),
	num_objs1_(num_objs1), num_negative_samples_(num_negative_samples),
	negative_sample_dist_(negative_sample_dist)
{
}

NegativeSamplingTrainer::~NegativeSamplingTrainer()
{
}

void NegativeSamplingTrainer::TrainEdge(int vec_dim, float *vec0, int obj1, float **vecs1, float alpha, float *tmp_neu1e,
	std::default_random_engine &generator, bool update0, bool update1)
{
	for (int i = 0; i < vec_dim; ++i)
		tmp_neu1e[i] = 0.0f;

	const float lambda = alpha * 0.01f;
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

		float dot_product = MathUtils::DotProduct(vec0, vecs1[target], vec_dim);
		float g = (label - exp_table_->getSigmaValue(dot_product)) * alpha;

		for (int j = 0; j < vec_dim; ++j)
			tmp_neu1e[j] += g * vecs1[target][j];
		if (update1)
			for (int j = 0; j < vec_dim; ++j)
				vecs1[target][j] += g * vec0[j] - lambda * vecs1[target][j];
	}

	if (update0)
		for (int j = 0; j < vec_dim; ++j)
			vec0[j] += tmp_neu1e[j] - lambda * vec0[j];
}

//void NegativeSamplingTrainer::TrainEdgeCM(int vec_dim, float *vec0, int obj1, float **vecs1, float *cm_params, bool complement,
//	float alpha, float *tmp_neu1e, float *tmp_cme, std::default_random_engine &generator, bool update0, 
//	bool update1, bool update_cm_params)
//{
//	int full_vec_dim = vec_dim * 2;
//	if (update0)
//		for (int i = 0; i < full_vec_dim; ++i)
//			tmp_neu1e[i] = 0.0f;
//	if (update_cm_params)
//		for (int i = 0; i < vec_dim; ++i)
//			tmp_cme[i] = 0.0f;
//
//	const float lambda = alpha * 0.01f;
//	int target = obj1;
//	int label = 1;
//	for (int i = 0; i < num_negative_samples_ + 1; ++i)
//	{
//		if (i != 0)
//		{
//			target = (*negative_sample_dist_)(generator);
//			if (target == obj1) continue;
//
//			label = 0;
//		}
//
//		float fval = calcCMEnergy(vec_dim, vec0, vecs1[target], cm_params);
//		//if (isnan(fval))
//		//{
//		//	for (int i = 0; i < vec_dim; ++i)
//		//		printf("%f ", vec0[i]);
//		//	printf("\n");
//		//}
//		assert(!isnan(fval));
//		float g = (label - exp_table_->getSigmaValue(fval)) * alpha;
//
//		if (update0)
//			for (int j = 0; j < full_vec_dim; ++j)
//				tmp_neu1e[j] += g * vecs1[target][j / 2] * cm_params[j];
//		if (update1)
//		{
//			for (int j = 0; j < vec_dim; ++j)
//			{
//				vecs1[target][j] += g * (cm_params[2 * j] * vec0[2 * j] + cm_params[2 * j + 1] * vec0[2 * j + 1])
//					- lambda * vecs1[target][j];
//			}
//		}
//		if (update_cm_params)
//		{
//			for (int j = 0; j < full_vec_dim; ++j)
//				tmp_cme[j] += g * vecs1[target][j / 2] * vec0[j];
//		}
//	}
//
//	if (update0)
//		for (int i = 0; i < full_vec_dim; ++i)
//			vec0[i] += tmp_neu1e[i] - lambda * vec0[i];
//	if (update_cm_params)
//		for (int i = 0; i < full_vec_dim; ++i)
//			cm_params[i] += tmp_cme[i] - lambda * cm_params[i];
//}

void NegativeSamplingTrainer::TrainEdgeMatrix(int dim0, int dim1, float *vec0, int obj1, float **vecs1, float *matrix, float alpha,
	float *tmp_neu1e, std::default_random_engine &generator, bool update0, bool update1, bool update_matrix)
{
	if (update0)
		for (int i = 0; i < dim0; ++i)
			tmp_neu1e[i] = 0.0f;

	const float nf = 0.001f;
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

		float *cur_vec1 = vecs1[target];

		float fval = MathUtils::XMY(vec0, dim0, cur_vec1, dim1, matrix);
		//printf("%f ", fval);
		assert(!isnan(fval));
		float g = (label - exp_table_->getSigmaValue(fval)) * alpha;

		if (update0)
			for (int j = 0; j < dim0; ++j)
				for (int k = 0; k < dim1; ++k)
					tmp_neu1e[j] += g * cur_vec1[k] * matrix[j * dim0 + k];
		if (update1)
			for (int k = 0; k < dim1; ++k)
				for (int j = 0; j < dim0; ++j)
					cur_vec1[k] += g * vec0[j] * matrix[j * dim0 + k] - nf * alpha * cur_vec1[k];
		if (update_matrix)
			for (int j = 0; j < dim0; ++j)
				for (int k = 0; k < dim1; ++k)
					matrix[j * dim0 + k] += g * vec0[j] * cur_vec1[k] - nf * alpha * matrix[j * dim0 + k];
		//printf("f %f %f %f\n", matrix[0 * vec0_dim_ + 0], g * vec0[0] * cur_vec1[0], vec0[0] * cur_vec1[0]);
	}

	if (update0)
		for (int j = 0; j < dim0; ++j)
			vec0[j] += tmp_neu1e[j] - nf * alpha * vec0[j];
}

void NegativeSamplingTrainer::CheckObject(int vec_dim, float *cur_vec, float **vecs1)
{
	const int k = 10;
	int top_indices[k];
	float vals[k];
	std::fill(vals, vals + k, -1e4f);
	float target_sum = 0;
	for (int i = 0; i < num_objs1_; ++i)
	{
		float dp = MathUtils::DotProduct(cur_vec, vecs1[i], vec_dim);
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

void NegativeSamplingTrainer::trainEdgeCM(int vec_dim, float *vec0, int obj1, float **vecs1, float *cm_params,
	float alpha, float *tmp_neu1e, float *tmp_cme, std::default_random_engine &generator,
	bool update0 = true, bool update1 = true, bool update_cm_params = true)
{
	if (update0)
		for (int i = 0; i < (vec_dim << 1); ++i)
			tmp_neu1e[i] = 0.0f;
	if (update_cm_params)
		for (int i = 0; i < vec_dim; ++i)
			tmp_cme[i] = 0.0f;

	const float lambda = alpha * 0.01f;
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

		float fval = calcCMEnergy(vec_dim, vec0, vecs1[target], cm_params);
		//if (isnan(fval))
		//{
		//	for (int i = 0; i < vec_dim; ++i)
		//		printf("%f ", vec0[i]);
		//	printf("\n");
		//}
		assert(!isnan(fval));
		float g = (label - exp_table_->getSigmaValue(fval)) * alpha;

		if (update0)
		{
			for (int j = 0; j < vec_dim; ++j)
			{
				tmp_neu1e[j << 1] += g * vecs1[target][j] * cm_params[j];
				tmp_neu1e[(j << 1) + 1] += g * vecs1[target][j] * (1 - cm_params[j]);
			}
		}
		if (update1)
		{
			for (int j = 0; j < vec_dim; ++j)
			{
				vecs1[target][j] += g * (cm_params[j] * vec0[j << 1] + (1 - cm_params[j]) * vec0[(j << 1) + 1])
					- lambda * vecs1[target][j];
			}
		}
		if (update_cm_params)
		{
			for (int j = 0; j < vec_dim; ++j)
				tmp_cme[j] += g * vecs1[target][j] * (vec0[j << 1] - vec0[(j << 1) + 1]);
		}
	}

	if (update0)
		for (int i = 0; i < (vec_dim << 1); ++i)
			vec0[i] += tmp_neu1e[i] - lambda * vec0[i];
	if (update_cm_params)
		for (int i = 0; i < vec_dim; ++i)
			cm_params[i] += tmp_cme[i] - lambda * (cm_params[i] - 0.5f);
}

void NegativeSamplingTrainer::trainEdgeCMComplement(int vec_dim, float *vec0, int obj1, float **vecs1, float *cm_params,
	float alpha, float *tmp_neu1e, float *tmp_cme, std::default_random_engine &generator,
	bool update0 = true, bool update1 = true, bool update_cm_params = true)
{
	if (update0)
		for (int i = 0; i < (vec_dim << 1); ++i)
			tmp_neu1e[i] = 0.0f;
	if (update_cm_params)
		for (int i = 0; i < vec_dim; ++i)
			tmp_cme[i] = 0.0f;

	const float lambda = alpha * 0.01f;
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

		float fval = calcCMEnergyComplement(vec_dim, vec0, vecs1[target], cm_params);
		//if (isnan(fval))
		//{
		//	for (int i = 0; i < vec_dim; ++i)
		//		printf("%f ", vec0[i]);
		//	printf("\n");
		//}
		assert(!isnan(fval));
		float g = (label - exp_table_->getSigmaValue(fval)) * alpha;

		if (update0)
		{
			for (int j = 0; j < vec_dim; ++j)
			{
				tmp_neu1e[j << 1] += g * vecs1[target][j] * (1 - cm_params[j]);
				tmp_neu1e[(j << 1) + 1] += g * vecs1[target][j] * cm_params[j];
			}
		}
		if (update1)
		{
			for (int j = 0; j < vec_dim; ++j)
			{
				vecs1[target][j] += g * ((1 - cm_params[j]) * vec0[j << 1] + cm_params[j] * vec0[(j << 1) + 1])
					- lambda * vecs1[target][j];
			}
		}
		if (update_cm_params)
		{
			for (int j = 0; j < vec_dim; ++j)
				tmp_cme[j] += g * vecs1[target][j] * (vec0[(j << 1) + 1] - vec0[j << 1]);
		}
	}

	if (update0)
		for (int i = 0; i < (vec_dim << 1); ++i)
			vec0[i] += tmp_neu1e[i] - lambda * vec0[i];
	if (update_cm_params)
		for (int i = 0; i < vec_dim; ++i)
			cm_params[i] += tmp_cme[i] - lambda * (cm_params[i] - 0.5f);
}
