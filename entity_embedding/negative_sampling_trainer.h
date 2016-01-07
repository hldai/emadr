#ifndef NEGATIVE_SAMPLING_TRAINER_H_
#define NEGATIVE_SAMPLING_TRAINER_H_

#include <random>

#include "exp_table.h"

class NegativeSamplingTrainer
{
public:
	static float **GetInitedVecs0(int num_objs, int vec_dim);
	static void InitVec0Def(float *vecs, int vec_dim);
	static float **GetInitedVecs1(int num_objs, int vec_dim);

	static void CloseVectors(float **vecs, int num_vecs, int vec_dim, int idx);

	static double *GetDefNegativeSamplingWeights(int *obj_cnts, int num_objs);

public:
	// use objs0 to predict objs1
	// e.g. objs0: documents, objs1: words
	NegativeSamplingTrainer(ExpTable *exp_table, int vec_dim, int num_objs1, 
		int num_negative_samples, std::discrete_distribution<int> *negative_sample_dist);
	//NegativeSamplingTrainer(ExpTable *exp_table, int vec_dim, int num_objs, int num_negative_samples,
	//	std::discrete_distribution<int> *obj_sample_dist);
	~NegativeSamplingTrainer();

	// obj0 -> obj1
	void TrainPrediction(float *vec0, int obj1, float **vecs1, float alpha, float *tmp_neu1e,
		std::default_random_engine &generator, bool update0 = true, bool update1 = true);

	void CheckObject(float *cur_vec, float **vecs1);


private:
	ExpTable *exp_table_;

	// use objs0 to predict objs1, e.g. objs0: documents, objs1: words
	int num_objs1_ = 0;

	int vec_dim_ = 0;

	int num_negative_samples_ = 0;
	std::discrete_distribution<int> *negative_sample_dist_;
};

#endif
