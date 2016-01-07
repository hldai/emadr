#ifndef MATRIX_TRAINER_H_
#define MATRIX_TRAINER_H_

#include <random>

#include "exp_table.h"

// thead safe
class MatrixTrainer
{
public:
	static void InitMatrix(float *matrix, int dim0, int dim1);
	// dst_vecs1[0] = matrix * vecs1[0] ...
	static void PreCalcVecs1(float *matrix, int dim0, int dim1, float **vecs1, 
		int num_vecs, float **dst_vecs1);

public:
	// use objs0 to predict objs1
	// e.g. objs0: documents, objs1: words
	MatrixTrainer(ExpTable *exp_table, int vec0_dim, int vec1_dim, int num_objs1, int num_negative_samples, 
		std::discrete_distribution<int> *negative_sample_dist);

	// obj0 -> obj1
	void TrainPrediction(float *vec0, int obj1, float **vecs1, float *matrix, float alpha, float *tmp_neu1e,
		std::default_random_engine &generator, bool update0 = true, bool update1 = true, bool update_matrix = true);

	void TrainMatrix(float *vec0, int obj1, float **vecs1, float *matrix, float alpha,
		std::default_random_engine &generator);

	void TrainWithPreCalcVecs1(float *vec0, int obj1, float **prec_vecs1, float alpha, float *tmp_neu1e,
		std::default_random_engine &generator);

	void CheckObject(float *cur_vec, float **vecs1, float *matrix);

	void ListScores(int num_objs, int *objs, float *cur_vec0, float **vecs1, float *matrix);

	int vec0_dim() { return vec0_dim_; }
	int vec1_dim() { return vec1_dim_; }

private:
	ExpTable *exp_table_;

	// use objs0 to predict objs1, e.g. objs0: documents, objs1: words
	int num_objs1_ = 0;

	int vec0_dim_ = 0;
	int vec1_dim_ = 0;

	int num_negative_samples_ = 0;
	std::discrete_distribution<int> *negative_sample_dist_;
};

#endif
