#include "paragraph_vector.h"

#include <cstdio>
#include <cassert>
#include <thread>

#include "mem_utils.h"
#include "io_utils.h"
#include "adj_list_net.h"
#include "math_utils.h"

ParagraphVector::ParagraphVector(const char *doc_word_net_file_name, float starting_alpha) : starting_alpha_(starting_alpha)
{
	AdjListNet doc_word_net;
	doc_word_net.LoadBinFile(doc_word_net_file_name);
	printf("%d\n", doc_word_net.num_adj_vertices[0]);
	num_words_ = doc_word_net.num_vertices_right;
	doc_word_net.ToEdgeNet(doc_word_net_);
	num_docs_ = doc_word_net_.num_vertices_left;

	int *word_cnts = doc_word_net.CountRightVertices();
	double *word_sample_weights = NegativeSamplingTrainer::GetDefNegativeSamplingWeights(word_cnts, num_words_);
	word_sample_dist_ = std::discrete_distribution<int>(word_sample_weights, word_sample_weights + num_words_);
	delete[] word_cnts;
	delete[] word_sample_weights;

	edge_sample_dist_ = std::discrete_distribution<int>(doc_word_net_.weights,
		doc_word_net_.weights + doc_word_net_.num_edges);
}

ParagraphVector::~ParagraphVector()
{
	//MemUtils::Release(doc_word_indices_, num_docs_);
	//MemUtils::Release(doc_word_cnts_, num_docs_);
	//delete[] num_doc_words_;
	//delete[] edges_;
}

void ParagraphVector::Train(int vec_dim, int num_threads,
	const char *dst_vec_file_name)
{
	float **vecs0 = NegativeSamplingTrainer::GetInitedVecs0(num_docs_, vec_dim);
	float **vecs1 = NegativeSamplingTrainer::GetInitedVecs1(num_words_, vec_dim);

	int seeds[] = { 3177, 17131, 313, 299297 };
	int sum_dw_edge_weights = MathUtils::Sum(doc_word_net_.weights, doc_word_net_.num_edges);
	const int num_samples = sum_dw_edge_weights;
	printf("%d samples per round\n", num_samples);
	const int num_rounds = 3;
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
	float min_alpha = starting_alpha_ * 0.005;
	int total_num_samples = num_rounds * num_samples;
	//int total_num_samples = num_samples;

	float *tmp_neu1e = new float[vec_dim];
	for (int i = 0; i < num_rounds; ++i)
	{
		//alpha = starting_alpha_;
		printf("round %d, alpha %f\n", i, alpha);
		for (int j = 0; j < num_samples; ++j)
		{
			int cur_num_samples = (i * num_samples) + j;
			//int cur_num_samples = j;
			if (cur_num_samples % 10000 == 10000 - 1)
			{
				alpha = starting_alpha_ + (min_alpha - starting_alpha_) * cur_num_samples / total_num_samples;
				//printf("alpha: %f\n", alpha);
				//alpha *= 0.97;
				//if (alpha < starting_alpha_ * 0.1)
				//	alpha = starting_alpha_ * 0.1;
			}

			int edge_idx = edge_sample_dist_(generator);
			int va = doc_word_net_.edges[edge_idx].va, vb = doc_word_net_.edges[edge_idx].vb;

			ns_trainer.TrainPrediction(vecs0[va], vb, vecs1, alpha, tmp_neu1e,
				generator, true, true);
		}
	}
	delete[] tmp_neu1e;
}
