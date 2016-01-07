#include "paragraph_vector.h"

#include <cstdio>
#include <cassert>
#include <thread>

#include "mem_utils.h"
#include "io_utils.h"

ParagraphVector::ParagraphVector(const char *doc_word_indices_file_name, const char *dict_file_name,
	float starting_alpha) : starting_alpha_(starting_alpha)
{
	FILE *fp = fopen(dict_file_name, "r");
	assert(fp != 0);
	fscanf(fp, "%d", &num_words_);
	printf("vocabulary size: %d\n", num_words_);
	fclose(fp);

	fp = fopen(doc_word_indices_file_name, "r");
	assert(fp != 0);

	fscanf(fp, "%d", &num_docs_);
	printf("%d documents.\n", num_docs_);

	int *word_cnts = new int[num_words_];
	std::fill(word_cnts, word_cnts + num_words_, 0);

	num_edges_ = 0;
	doc_word_indices_ = new int*[num_docs_];
	doc_word_cnts_ = new int*[num_docs_];
	num_doc_words_ = new int[num_docs_];
	for (int i = 0; i < num_docs_; ++i)
	{
		fscanf(fp, "%d", &num_doc_words_[i]);
		num_edges_ += num_doc_words_[i];

		doc_word_indices_[i] = new int[num_doc_words_[i]];
		doc_word_cnts_[i] = new int[num_doc_words_[i]];
		for (int j = 0; j < num_doc_words_[i]; ++j)
		{
			fscanf(fp, "%d %d", &doc_word_indices_[i][j], &doc_word_cnts_[i][j]);
			word_cnts[doc_word_indices_[i][j]] += doc_word_cnts_[i][j];
		}
	}

	fclose(fp);

	printf("%d edges\n", num_edges_);

	word_sample_dist_ = std::discrete_distribution<int>(word_cnts, 
		word_cnts + num_words_);
	delete[] word_cnts;


	int *edge_weights = new int[num_edges_];
	edges_ = new Edge[num_edges_];
	int edge_idx = 0;
	for (int i = 0; i < num_docs_; ++i)
	{
		for (int j = 0; j < num_doc_words_[i]; ++j)
		{
			edges_[edge_idx].va = i;
			edges_[edge_idx].vb = doc_word_indices_[i][j];
			edge_weights[edge_idx] = doc_word_cnts_[i][j];
			++edge_idx;
		}
	}

	edge_sample_dist_ = std::discrete_distribution<int>(edge_weights,
		edge_weights + num_edges_);
	delete[] edge_weights;
}

ParagraphVector::~ParagraphVector()
{
	MemUtils::Release(doc_word_indices_, num_docs_);
	MemUtils::Release(doc_word_cnts_, num_docs_);
	delete[] num_doc_words_;
	delete[] edges_;
}

void ParagraphVector::Train(int vec_dim, int num_threads,
	const char *dst_vec_file_name)
{
	float **vecs0 = NegativeSamplingTrainer::GetInitedVecs0(num_docs_, vec_dim);
	float **vecs1 = NegativeSamplingTrainer::GetInitedVecs1(num_words_, vec_dim);

	int seeds[] = { 3177, 17131, 313, 299297 };
	const int num_samples = num_edges_;
	const int num_rounds = 2;
	ExpTable exp_table;
	NegativeSamplingTrainer ns_trainer(&exp_table, vec_dim, num_words_,
		num_negative_samples_, &word_sample_dist_);
	std::thread *threads = new std::thread[num_threads];
	for (int i = 0; i < num_threads; ++i)
	{
		int cur_seed = seeds[i];
		threads[i] = std::thread([=, &ns_trainer]
		{
			Train(vec_dim, vecs0, vecs1, num_samples, num_rounds, ns_trainer,
				cur_seed);
		});
	}
	for (int i = 0; i < num_threads; ++i)
		threads[i].join();
	delete[] threads;

	NegativeSamplingTrainer::CloseVectors(vecs0, num_docs_, vec_dim, 2);

	IOUtils::SaveVectors(vecs0, vec_dim, num_docs_, dst_vec_file_name);

	MemUtils::Release(vecs0, num_docs_);
	MemUtils::Release(vecs1, num_words_);
}

void ParagraphVector::Train(int vec_dim, float **vecs0, float **vecs1, 
	int num_samples, int num_rounds, NegativeSamplingTrainer &ns_trainer,
	int random_seed)
{
	printf("thread seed: %d\n", random_seed);
	std::default_random_engine generator(random_seed);
	float alpha = starting_alpha_;
	float *tmp_neu1e = new float[vec_dim];
	for (int i = 0; i < num_rounds; ++i)
	{
		printf("round %d, alpha %f\n", i, alpha);
		for (int j = 0; j < num_samples; ++j)
		{
			int edge_idx = edge_sample_dist_(generator);
			int va = edges_[edge_idx].va, vb = edges_[edge_idx].vb;
			ns_trainer.TrainPrediction(vecs0[va], vb, vecs1, alpha, tmp_neu1e,
				generator, true, true);
		}
	}
	delete[] tmp_neu1e;
}
