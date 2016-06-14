#ifndef EADOCVECTRAINER_H_
#define EADOCVECTRAINER_H_

#include <random>

#include "pairsampler.h"
#include "negtrain.h"
#include "negsamplingdoubleobj.h"

class EADocVecTrainer
{
public:
	EADocVecTrainer(int num_rounds, int num_threads, int num_negative_samples, float starting_alpha,
		float min_alpha = 0.0001f);

	void AllJointThreaded(const char *ee_file, const char *doc_entity_file,
		const char *doc_words_file_name, const char *entity_cnts_file, const char *word_cnts_file,
		int vec_dim, bool shared, const char *dst_dedw_vec_file_name, const char *dst_word_vecs_file_name, 
		const char *dst_entity_vecs_file_name);

	void TrainWEFixed(const char *doc_words_file, const char *doc_entities_file, const char *word_cnts_file,
		const char *entity_cnts_file, const char *word_vecs_file_name, const char *entity_vecs_file_name, 
		int vec_dim, const char *dst_doc_vecs_file);

	void TrainDocWord(const char *doc_words_file_name, const char *word_cnts_file, int vec_dim,
		const char *dst_doc_vecs_file_name, const char *dst_word_vecs_file_name = 0);

	void TrainDocWordFixedWordVecs(const char *doc_words_file_name, const char *word_cnts_file, 
		const char *word_vecs_file_name, int vec_dim, const char *dst_doc_vecs_file_name);

private:
	void initDocWordList(const char *doc_words_file_name)
	{
		dw_sampler_ = new PairSampler(doc_words_file_name);
		num_words_ = dw_sampler_->num_vertex_right();
		num_docs_ = dw_sampler_->num_vertex_left();
		printf("%d docs, %d words.\n", num_docs_, num_words_);
	}

	void initDocEntityList(const char *de_file)
	{
		de_sampler_ = new PairSampler(de_file);
		num_docs_ = de_sampler_->num_vertex_left();
		num_entities_ = de_sampler_->num_vertex_right();
		printf("%d docs, %d entities.\n", num_docs_, num_entities_);
	}

	void initEntityEntityList(const char *ee_file)
	{
		ee_sampler_ = new PairSampler(ee_file);
		num_entities_ = ee_sampler_->num_vertex_left();
		printf("%d entities.\n", num_entities_);
	}

	void saveConcatnatedVectors(float **vecs0, float **vecs1, int num_vecs, int vec_dim,
		const char *dst_file_name);

	void allJoint(int seed, long long num_samples_per_round, std::discrete_distribution<int> &list_sample_dist,
		NegTrain &entity_ns_trainer, NegTrain &word_ns_trainer);

	void trainDocWordMT(const char *word_cnts_file, bool update_word_vecs, const char *dst_doc_vecs_file_name);
	void trainDocWordList(int seed, long long num_samples_per_round, bool update_word_vecs, 
		NegTrain &word_ns_trainer);

	void trainDWEMT(const char *word_cnts_file, const char *entity_cnts_file, bool update_word_vecs, 
		bool update_entity_vecs, const char *dst_doc_vecs_file_name);
	void trainDWETh(int seed, long long num_samples_per_round, bool update_word_vecs, bool update_entity_vecs, std::discrete_distribution<int> &list_sample_dist,
		NegTrain &word_ns_trainer, NegTrain &entity_ns_trainer);

private:
	int num_rounds_ = 10;
	int num_threads_ = 1;
	int num_negative_samples_ = 10;

	float starting_alpha_;
	float min_alpha_;

	PairSampler *dw_sampler_ = 0;
	PairSampler *de_sampler_ = 0;
	PairSampler *ee_sampler_ = 0;

	int num_words_ = 0;
	int num_docs_ = 0;
	int num_entities_ = 0;

	float **word_vecs_ = 0;
	float **ee_vecs0_ = 0;
	float **ee_vecs1_ = 0;
	float **dw_vecs_ = 0;
	float **de_vecs_ = 0;

	float **doc_vecs_ = 0;

	int entity_vec_dim_ = 0;
	int word_vec_dim_ = 0;
};

#endif
