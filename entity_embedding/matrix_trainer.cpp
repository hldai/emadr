#include "matrix_trainer.h"

#include <cmath>
#include <cassert>
#include <random>

#include "math_utils.h"

void MatrixTrainer::InitMatrix(float *matrix, int dim0, int dim1)
{
	std::default_random_engine generator(1217);
	float rand_val = 4 * sqrt(6.0 / (dim0 + dim1));
	std::uniform_real_distribution<float> distribution(-rand_val, rand_val);
	for (int i = 0; i < dim0 * dim1; ++i)
		matrix[i] = distribution(generator);
}

void MatrixTrainer::PreCalcVecs1(float *matrix, int dim0, int dim1, float **vecs1, 
	int num_vecs, float **dst_vecs1)
{
	for (int i = 0; i < num_vecs; ++i)
		MathUtils::MY(matrix, vecs1[i], dim0, dim1, dst_vecs1[i]);
}

MatrixTrainer::MatrixTrainer(ExpTable *exp_table, int vec0_dim, int vec1_dim, int num_objs1,
	int num_negative_samples, std::discrete_distribution<int> *negative_sample_dist) : exp_table_(exp_table), 
	vec0_dim_(vec0_dim), vec1_dim_(vec1_dim), num_objs1_(num_objs1), num_negative_samples_(num_negative_samples),
	negative_sample_dist_(negative_sample_dist)
{
}

void MatrixTrainer::TrainPrediction(float *vec0, int obj1, float **vecs1, float *matrix, float alpha,
	float *tmp_neu1e, std::default_random_engine &generator, bool update0, bool update1, bool update_matrix)
{
	for (int i = 0; i < vec0_dim_; ++i)
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

		float fval = MathUtils::XMY(vec0, vec0_dim_, cur_vec1, vec1_dim_, matrix);
		//printf("%f ", fval);
		assert(!isnan(fval));
		float g = (label - exp_table_->getSigmaValue(fval)) * alpha;
		//printf("%f\n", g);
		//printf("%f\n", fval);
		//float tmp = exp(fval);
		//float g = (label - tmp / (1 + tmp)) * alpha;
		//printf("g %f\n", g);

		//printf("%f %f %f\n", g * cur_vec1[0] * matrix[0 * vec0_dim_ + 0], cur_vec1[0], matrix[0 * vec0_dim_ + 0]);
		if (update0)
			for (int j = 0; j < vec0_dim_; ++j)
				for (int k = 0; k < vec1_dim_; ++k)
					tmp_neu1e[j] += g * cur_vec1[k] * matrix[j * vec0_dim_ + k];

		if (update1)
			for (int k = 0; k < vec1_dim_; ++k)
				for (int j = 0; j < vec0_dim_; ++j)
					cur_vec1[k] += g * vec0[j] * matrix[j * vec0_dim_ + k] - nf * alpha * cur_vec1[k];

		if (update_matrix)
			for (int j = 0; j < vec0_dim_; ++j)
				for (int k = 0; k < vec1_dim_; ++k)
					matrix[j * vec0_dim_ + k] += g * vec0[j] * cur_vec1[k] - nf * alpha * matrix[j * vec0_dim_ + k];
		//printf("f %f %f %f\n", matrix[0 * vec0_dim_ + 0], g * vec0[0] * cur_vec1[0], vec0[0] * cur_vec1[0]);
	}

	if (update0)
	{
		for (int j = 0; j < vec0_dim_; ++j)
			vec0[j] += tmp_neu1e[j] - nf * alpha * vec0[j];
	}
}

void MatrixTrainer::TrainMatrix(float *vec0, int obj1, float **vecs1, float *matrix, float alpha,
	std::default_random_engine &generator)
{
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

		float fval = MathUtils::XMY(vec0, vec0_dim_, cur_vec1, vec1_dim_, matrix);
		//printf("%f ", fval);
		assert(!isnan(fval));
		float g = (label - exp_table_->getSigmaValue(fval)) * alpha;

		for (int j = 0; j < vec0_dim_; ++j)
			for (int k = 0; k < vec1_dim_; ++k)
				matrix[j * vec0_dim_ + k] += g * vec0[j] * cur_vec1[k] - nf * alpha * matrix[j * vec0_dim_ + k];
	}
}

void MatrixTrainer::TrainWithPreCalcVecs1(float *vec0, int obj1, float **prec_vecs1, float alpha, float *tmp_neu1e,
	std::default_random_engine &generator)
{
	for (int i = 0; i < vec0_dim_; ++i)
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

		float *cur_vec1 = prec_vecs1[target];

		float fval = MathUtils::DotProduct(vec0, cur_vec1, vec0_dim_);
		//float fval = MathUtils::XMY(vec0, vec0_dim_, cur_vec1, vec1_dim_, matrix);
		//assert(!isnan(fval));
		float g = (label - exp_table_->getSigmaValue(fval)) * alpha;

		for (int j = 0; j < vec0_dim_; ++j)
			tmp_neu1e[j] += g * cur_vec1[j];
	}

	for (int j = 0; j < vec0_dim_; ++j)
		vec0[j] += tmp_neu1e[j] - nf * alpha * vec0[j];
}

void MatrixTrainer::CheckObject(float *cur_vec, float **vecs1, float *matrix)
{
	const int k = 10;
	int top_indices[k];
	float vals[k];
	std::fill(vals, vals + k, -1e4f);
	float target_sum = 0;
	for (int i = 0; i < num_objs1_; ++i)
	{
		//float dp = MathUtils::DotProduct(cur_vec, vecs1[i], vec_dim_);
		float dp = MathUtils::XMY(cur_vec, vec0_dim_, vecs1[i], vec1_dim_, matrix);
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

void MatrixTrainer::ListScores(int num_objs, int *objs, float *cur_vec0, float **vecs1, float *matrix)
{
	int cnt = 0;
	for (int i = 0; i < num_objs; ++i)
	{
		float *vec1 = vecs1[objs[i]];
		float dp = MathUtils::XMY(cur_vec0, vec0_dim_, vec1, vec1_dim_, matrix);
		//printf("dp: %f\n", dp);
		if (dp > 0)
			++cnt;
	}
	printf("positive: %d %d %f\n", cnt, num_objs, (float)cnt / num_objs);

	//printf("\n");
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, num_objs1_);
	const int num_test_neg_samples = 100;
	cnt = 0;
	for (int i = 0; i < num_test_neg_samples; ++i)
	{
		float *vec1 = vecs1[distribution(generator)];
		float dp = MathUtils::XMY(cur_vec0, vec0_dim_, vec1, vec1_dim_, matrix);
		//printf("dp: %f\n", dp);
		if (dp < 0)
			++cnt;
	}
	printf("negative: %d %d %f\n", cnt, num_test_neg_samples,
		(float)cnt / num_test_neg_samples);
}
