// entity_vec_trainer.h
//
//  Created on: Dec 18, 2015
//      Author: dhl

#ifndef ENTITY_VEC_TRAINER_H_
#define ENTITY_VEC_TRAINER_H_

#include <random>

class EntityVecTrainer {
	static const int kExpTableSize = 1000;
	static const float kMaxExp;

	static const int kDefAlphaUpdateFreq = 50000;
	static const float kDefAlpha;

public:
	struct Edge {
		int va;
		int vb;
	};

public:
	EntityVecTrainer(const char *edge_list_file_name);
	~EntityVecTrainer();

	void ThreadedTrain(int vec_dim, int num_rounds, int num_threads, int num_negative_samples,
		const char *dst_entity_vec_file_name, const char *dst_output_vecs_file_name);

	void Train(int seed);

private:
	void trainWithEdge(int va, int vb, float alpha, float *tmp_neu1e,
		std::default_random_engine &generator);

	void loadEdgesFromFile(const char *edge_list_file_name);
	void initExpTable();
	void initNet();

	void writeVectorsToFile(float **vecs, int vec_len, int num_vecs, const char *dst_file_name);

	float getSigmaValue(float x)
	{
		return exp_table_[(int)((x + kMaxExp) * (kExpTableSize / kMaxExp / 2))];
	}

	int valueInArray(int val, int *arr, int arr_len)
	{
		if (arr == 0)
			return -1;

		int l = 0, r = arr_len - 1;
		while (r >= l)
		{
			int m = (l + r) >> 1;
			if (arr[m] > val)
				r = m - 1;
			else if (arr[m] < val)
				l = m + 1;
			else
				return m;
		}
		return -1;
	}

	void closeEntities(int entity_index);

	int num_entities_ = 0;
	int num_edges_ = 0;
	int sum_weights_ = 0;
	int sum_entity_cnts_ = 0;
	Edge *edges_ = 0;
	int *weights_ = 0;
	int *entity_cnts_ = 0;

	double *entity_sample_weights_ = 0;

	int **adj_list_;
	int *num_adj_vertices_;

	int num_negative_samples_ = 0;

	int vec_dim_ = 0;
	float **syn0_ = 0, **syn1_ = 0;

	std::discrete_distribution<int> edge_sample_dist_;
	std::discrete_distribution<int> entity_sample_dist_;


	float *exp_table_ = 0;

	int num_training_rounds_ = 0;
	float starting_alpha_;

	int *sampled_cnts_ = 0;
};

#endif /* ENTITY_VEC_TRAINER_H_ */
