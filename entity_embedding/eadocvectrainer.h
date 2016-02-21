#ifndef EADOCVECTRAINER_H_
#define EADOCVECTRAINER_H_

#include <random>

#include "net_edge_sampler.h"
#include "negative_sampling_trainer.h"

class EADocVecTrainer
{
public:
	void Init(const char *ee_net_file_name, const char *doc_entity_net_file_name,
		const char *doc_words_file_name);

	void AllJointThreaded(int vec_dim, int num_rounds, int num_threads, int num_negative_samples, float starting_alpha,
		float min_alpha, const char *dst_dedw_vec_file_name, const char *dst_word_vecs_file_name,
		const char *dst_entity_vecs_file_name);

	void AllJoint(int seed, int num_rounds, long long num_samples_per_round, std::discrete_distribution<int> &net_sample_dist,
		NegativeSamplingTrainer &entity_ns_trainer, NegativeSamplingTrainer &word_ns_trainer);

private:
	void saveConcatnatedVectors(float **vecs0, float **vecs1, int num_vecs, int vec_dim,
		const char *dst_file_name);

private:
	float starting_alpha_;
	float min_alpha_;

	NetEdgeSampler *dw_edge_sampler_ = 0;
	NetEdgeSampler *de_edge_sampler_ = 0;
	NetEdgeSampler *ee_edge_sampler_ = 0;

	int num_words_ = 0;
	int num_docs_ = 0;
	int num_entities_ = 0;

	float **word_vecs_ = 0;
	float **ee_vecs0_ = 0;
	float **ee_vecs1_ = 0;
	float **dw_vecs_ = 0;
	float **de_vecs_ = 0;

	int entity_vec_dim_ = 0;
	int word_vec_dim_ = 0;
};

#endif
