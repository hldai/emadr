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

	static float *GetInitedCMParams(int vec_dim);

	// for energy with a matrix
	// probably not used
	static void InitMatrix(float *matrix, int dim0, int dim1);

	static void CloseVectors(float **vecs, int num_vecs, int vec_dim, int idx);

	static float *GetDefNegativeSamplingWeights(int *obj_cnts, int num_objs);

public:
	// use objs0 to predict objs1
	// e.g. objs0: documents, objs1: words
	NegativeSamplingTrainer(ExpTable *exp_table, int num_objs1, 
		int num_negative_samples, std::discrete_distribution<int> *negative_sample_dist);
	//NegativeSamplingTrainer(ExpTable *exp_table, int vec_dim, int num_objs, int num_negative_samples,
	//	std::discrete_distribution<int> *obj_sample_dist);
	~NegativeSamplingTrainer();

	// obj0 -> obj1
	void TrainEdge(int vec_dim, float *vec0, int obj1, float **vecs1, float alpha, float *tmp_neu1e,
		std::default_random_engine &generator, bool update0 = true, bool update1 = true);

	// controled mix
	// dimention of vec0: vec_dim * 2
	void TrainEdgeCM(int vec_dim, float *vec0, int obj1, float **vecs1, float *cm_params, bool complement,
		float alpha, float *tmp_neu1e, float *tmp_cme, std::default_random_engine &generator,
		bool update0 = true, bool update1 = true, bool update_cm_params = true)
	{
		if (complement)
		{
			trainEdgeCMComplement(vec_dim, vec0, obj1, vecs1, cm_params, alpha, tmp_neu1e,
				tmp_cme, generator, update0, update1, update_cm_params);
		}
		else
		{
			trainEdgeCM(vec_dim, vec0, obj1, vecs1, cm_params, alpha, tmp_neu1e,
				tmp_cme, generator, update0, update1, update_cm_params);
		}

		for (int i = 0; i < vec_dim; ++i)
		{
			if (cm_params[i] < 0)
				cm_params[i] = 0.01f;
			else if (cm_params[i] > 1)
				cm_params[i] = 0.99f;
		}
	}

	void TrainEdgeMatrix(int dim0, int dim1, float *vec0, int obj1, float **vecs1, float *matrix, float alpha, float *tmp_neu1e,
		std::default_random_engine &generator, bool update0 = true, bool update1 = true, bool update_matrix = true);

	void CheckObject(int vec_dim, float *cur_vec, float **vecs1);

private:
	void trainEdgeCM(int vec_dim, float *vec0, int obj1, float **vecs1, float *cm_params,
		float alpha, float *tmp_neu1e, float *tmp_cme, std::default_random_engine &generator,
		bool update0, bool update1, bool update_cm_params);
	void trainEdgeCMComplement(int vec_dim, float *vec0, int obj1, float **vecs1, float *cm_params,
		float alpha, float *tmp_neu1e, float *tmp_cme, std::default_random_engine &generator,
		bool update0, bool update1, bool update_cm_params);

	float calcCMEnergy(int vec_dim, float *vec0, float *vec1, float *cm_params)
	{
		float result = 0;
		for (int i = 0; i < vec_dim; ++i)
		{
			int tmp = i << 1;
			result += vec1[i] * (vec0[tmp] * cm_params[i] + vec0[tmp + 1] * (1 - cm_params[i]));
		}
		return result;
	}

	float calcCMEnergyComplement(int vec_dim, float *vec0, float *vec1, float *cm_params)
	{
		float result = 0;
		for (int i = 0; i < vec_dim; ++i)
		{
			int tmp = i << 1;
			result += vec1[i] * (vec0[tmp] * (1 - cm_params[i]) + vec0[tmp + 1] * cm_params[i]);
		}
		return result;
	}


private:
	ExpTable *exp_table_;

	// use objs0 to predict objs1, e.g. objs0: documents, objs1: words
	int num_objs1_ = 0;

	int num_negative_samples_ = 0;
	std::discrete_distribution<int> *negative_sample_dist_;
};

#endif
