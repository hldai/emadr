#ifndef PARAGRAPH_VECTOR_H_
#define PARAGRAPH_VECTOR_H_

#include <random>

#include "negative_sampling_trainer.h"

class ParagraphVector
{
	struct Edge
	{
		int va;
		int vb;
	};

	static const int kNumNegSamples = 10;

public:
	ParagraphVector(const char *doc_word_indices_file_name, const char *dict_file_name,
		float starting_alpha = 0.02);
	~ParagraphVector();

	void Train(int vec_dim, int num_threads, const char *dst_vec_file_name);
	void Train(int vec_dim, float **vecs0, float **vecs1, int num_samples,
		int num_rounds, NegativeSamplingTrainer &ns_trainer, int random_seed);

private:
	int num_words_;

	int num_docs_;
	int **doc_word_indices_ = 0;
	int **doc_word_cnts_ = 0;
	int *num_doc_words_ = 0;

	int num_edges_ = 0;
	Edge *edges_ = 0;

	int num_negative_samples_ = kNumNegSamples;
	float starting_alpha_;

	std::discrete_distribution<int> word_sample_dist_;
	std::discrete_distribution<int> edge_sample_dist_;
};

#endif
