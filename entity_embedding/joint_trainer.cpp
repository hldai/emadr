#include "joint_trainer.h"

#include <cstdio>
#include <cassert>
#include <algorithm>
#include <random>
#include <thread>

#include "negative_sampling_trainer.h"
#include "io_utils.h"
#include "mem_utils.h"
#include "adj_list_net.h"
#include "math_utils.h"

JointTrainer::JointTrainer(const char *ee_net_file_name, const char *doc_entity_net_file_name,
	const char *doc_words_file_name)
{
	if (ee_net_file_name != 0)
	{
		entity_net_.LoadTextFile(ee_net_file_name);
		num_entities_ = entity_net_.num_vertices_left;
		//decObjIndices(ee_edges_, num_ee_edges_);
	}

	AdjListNet doc_entity_net;
	doc_entity_net.LoadBinFile(doc_entity_net_file_name);
	doc_entity_net.ToEdgeNet(entity_doc_net_);
	num_docs_ = doc_entity_net.num_vertices_left;

	assert(num_entities_ == 0 || doc_entity_net.num_vertices_right == num_entities_);

	num_entities_ = doc_entity_net.num_vertices_right;

	//int *entity_cnts = getEntitySampleWeights(doc_entity_net);
	int *entity_cnts = doc_entity_net.CountRightVertices();
	double *entity_sample_weights = NegativeSamplingTrainer::GetDefNegativeSamplingWeights(entity_cnts, num_entities_);
	entity_sample_dist_ = std::discrete_distribution<int>(entity_sample_weights, entity_sample_weights + num_entities_);
	delete[] entity_cnts;
	delete[] entity_sample_weights;

	AdjListNet doc_word_net;
	doc_word_net.LoadBinFile(doc_words_file_name);
	printf("%d\n", doc_word_net.num_adj_vertices[0]);
	num_words_ = doc_word_net.num_vertices_right;
	doc_word_net.ToEdgeNet(doc_word_net_);

	int *word_cnts = doc_word_net.CountRightVertices();
	double *word_sample_weights = NegativeSamplingTrainer::GetDefNegativeSamplingWeights(word_cnts, num_words_);
	word_sample_dist_ = std::discrete_distribution<int>(word_sample_weights, word_sample_weights + num_words_);
	delete[] word_cnts;
	delete[] word_sample_weights;
}

JointTrainer::~JointTrainer()
{
	if (ee_vecs0_ != 0)
		MemUtils::Release(ee_vecs0_, entity_net_.num_vertices_left);
	if (ee_vecs1_ != 0)
		MemUtils::Release(ee_vecs1_, entity_net_.num_vertices_left);

	if (doc_vecs_ != 0)
		MemUtils::Release(doc_vecs_, num_docs_);
}

void JointTrainer::JointTrainingThreaded(int entity_vec_dim, int word_vec_dim, int doc_vec_dim, 
	int num_rounds, int num_threads, int num_negative_samples, const char * dst_doc_vec_file_name)
{
	entity_vec_dim_ = entity_vec_dim;
	word_vec_dim_ = word_vec_dim;
	doc_vec_dim_ = doc_vec_dim;

	std::discrete_distribution<int> ee_edge_sample_dist(entity_net_.weights,
		entity_net_.weights + entity_net_.num_edges);
	std::discrete_distribution<int> de_edge_sample_dist(entity_doc_net_.weights,
		entity_doc_net_.weights + entity_doc_net_.num_edges);
	std::discrete_distribution<int> dw_edge_sample_dist(doc_word_net_.weights,
		doc_word_net_.weights + doc_word_net_.num_edges);

	printf("initing model....\n");
	ee_vecs0_ = NegativeSamplingTrainer::GetInitedVecs0(num_entities_, entity_vec_dim_);
	ee_vecs1_ = NegativeSamplingTrainer::GetInitedVecs1(num_entities_, entity_vec_dim_);
	doc_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_docs_, doc_vec_dim_);
	word_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_words_, word_vec_dim_);
	ExpTable exp_table;
	NegativeSamplingTrainer entity_ns_trainer(&exp_table, entity_vec_dim_, num_entities_, 
		num_negative_samples, &entity_sample_dist_);
	NegativeSamplingTrainer word_ns_trainer(&exp_table, word_vec_dim_, num_words_,
		num_negative_samples, &word_sample_dist_);
	printf("inited.\n");

	int sum_ee_edge_weights = MathUtils::Sum(entity_net_.weights, entity_net_.num_edges);
	int sum_de_edge_weights = MathUtils::Sum(entity_doc_net_.weights, entity_doc_net_.num_edges);
	int sum_dw_edge_weights = MathUtils::Sum(doc_word_net_.weights, doc_word_net_.num_edges);
	int sum_weights = sum_ee_edge_weights + sum_de_edge_weights + sum_dw_edge_weights;
	int num_samples_per_round = sum_weights;
	//int num_samples_per_round = sum_dw_edge_weights;
	//int num_samples_per_round = sum_ee_edge_weights + sum_de_edge_weights;

	float weight_portions[] = { (float)sum_ee_edge_weights / sum_weights,
		(float)sum_de_edge_weights / sum_weights, (float)sum_dw_edge_weights / sum_weights };
	printf("net distribution: %f %f %f\n", weight_portions[0], weight_portions[1], 
		weight_portions[2]);
	std::discrete_distribution<int> net_sample_dist(weight_portions, weight_portions + 3);
	//std::discrete_distribution<int> net_sample_dist{ 0, 0, 1 };

	int seeds[] = { 317, 7, 31, 297 };
	std::thread *threads = new std::thread[num_threads];
	for (int i = 0; i < num_threads; ++i)
	{
		int cur_seed = seeds[i];
		threads[i] = std::thread([=, &ee_edge_sample_dist, &de_edge_sample_dist, &dw_edge_sample_dist,
			&net_sample_dist, &entity_ns_trainer, &word_ns_trainer] 
		{ 
			JointTraining(cur_seed, num_rounds, num_samples_per_round, ee_edge_sample_dist, 
				de_edge_sample_dist, dw_edge_sample_dist, net_sample_dist, entity_ns_trainer, word_ns_trainer);
		});
	}
	for (int i = 0; i < num_threads; ++i)
		threads[i].join();
	printf("\n");

	IOUtils::SaveVectors(doc_vecs_, doc_vec_dim_, num_docs_, dst_doc_vec_file_name);
}


void JointTrainer::JointTraining(int seed, int num_rounds, int num_samples_per_round, std::discrete_distribution<int> &ee_edge_sample_dist,
	std::discrete_distribution<int> &de_edge_sample_dist, std::discrete_distribution<int> &dw_edge_sample_dist, std::discrete_distribution<int> &net_sample_dist,
	NegativeSamplingTrainer &entity_ns_trainer, NegativeSamplingTrainer &word_ns_trainer)
{
	printf("seed %d samples_per_round %d. training...\n", seed, num_samples_per_round);
	std::default_random_engine generator(seed);

	const float starting_alpha_ = 0.05f;
	const float min_alpha = starting_alpha_ * 0.005;
	int total_num_samples = num_rounds * num_samples_per_round;

	float *tmp_entity_neu1e = new float[entity_vec_dim_];
	float *tmp_word_neu1e = new float[word_vec_dim_];

	float alpha = starting_alpha_;  // TODO
	for (int i = 0; i < num_rounds; ++i)
	{
		printf("round %d, alpha %f\n", i, alpha);
		//alpha *= 0.96f;
		//if (alpha < starting_alpha_ * 0.1f)
		//	alpha = starting_alpha_ * 0.1f;
		//if (seed < 10)
		//	printf("\r%d %f %d \n", i, alpha, num_samples_per_round);
		for (int j = 0; j < num_samples_per_round; ++j)
		{
			int cur_num_samples = (i * num_samples_per_round) + j;
			if (cur_num_samples % 10000 == 10000 - 1)
				alpha = starting_alpha_ + (min_alpha - starting_alpha_) * cur_num_samples / total_num_samples;

			int net_idx = net_sample_dist(generator);
			if (net_idx == 0)
			{
				int edge_idx = ee_edge_sample_dist(generator);
				Edge &edge = entity_net_.edges[edge_idx];
				entity_ns_trainer.TrainPrediction(ee_vecs0_[edge.va], edge.vb, ee_vecs1_,
					alpha, tmp_entity_neu1e, generator);
				entity_ns_trainer.TrainPrediction(ee_vecs0_[edge.vb], edge.va, ee_vecs1_,
					alpha, tmp_entity_neu1e, generator);
			}
			else if (net_idx == 1)
			{
				int edge_idx = de_edge_sample_dist(generator);
				Edge &edge = entity_doc_net_.edges[edge_idx];
				entity_ns_trainer.TrainPrediction(doc_vecs_[edge.va], edge.vb, ee_vecs0_,
					alpha, tmp_entity_neu1e, generator);
			}
			else
			{
				int edge_idx = dw_edge_sample_dist(generator);
				Edge &edge = doc_word_net_.edges[edge_idx];
				word_ns_trainer.TrainPrediction(doc_vecs_[edge.va] + doc_vec_dim_ - word_vec_dim_, edge.vb, word_vecs_, 
					alpha, tmp_word_neu1e, generator);
			}
		}
	}

	delete[] tmp_entity_neu1e;
	delete[] tmp_word_neu1e;
}

void JointTrainer::TrainEntityNetThreaded(int vec_dim, int num_rounds, int num_threads, int num_negative_samples,
	const char *dst_input_vec_file_name, const char *dst_output_vecs_file_name)
{
	entity_vec_dim_ = vec_dim;

	std::discrete_distribution<int> ee_edge_sample_dist = std::discrete_distribution<int>(entity_net_.weights,
		entity_net_.weights + entity_net_.num_edges);

	printf("initing model....\n");
	ee_vecs0_ = NegativeSamplingTrainer::GetInitedVecs0(num_entities_, entity_vec_dim_);
	ee_vecs1_ = NegativeSamplingTrainer::GetInitedVecs1(num_entities_, entity_vec_dim_);
	ExpTable exp_table;
	NegativeSamplingTrainer ns_trainer(&exp_table, entity_vec_dim_, num_entities_, num_negative_samples, &entity_sample_dist_);
	printf("inited.\n");

	int sum_ee_edge_weights = MathUtils::Sum(entity_net_.weights, entity_net_.num_edges);

	int seeds[] = { 317, 7, 31, 297 };
	std::thread *threads = new std::thread[num_threads];
	for (int i = 0; i < num_threads; ++i)
	{
		int cur_seed = seeds[i];
		threads[i] = std::thread([=, &ee_edge_sample_dist, &ns_trainer] { TrainEntityNet(cur_seed, num_rounds, sum_ee_edge_weights,
			ee_edge_sample_dist, ns_trainer); });
	}
	for (int i = 0; i < num_threads; ++i)
		threads[i].join();
	printf("\n");

	IOUtils::SaveVectors(ee_vecs0_, entity_vec_dim_, num_entities_, dst_input_vec_file_name);
	if (dst_output_vecs_file_name != 0)
		IOUtils::SaveVectors(ee_vecs1_, entity_vec_dim_, num_entities_, dst_output_vecs_file_name);
}

void JointTrainer::TrainEntityNet(int seed, int num_training_rounds, int num_samples_per_round, 
	std::discrete_distribution<int> &ee_edge_sample_dist, NegativeSamplingTrainer &ns_trainer)
{
	printf("%d %lld training...\n", seed, (unsigned long long)ee_vecs0_);
	std::default_random_engine generator(seed);
	
	const float starting_alpha_ = 0.035f;

	float *tmp_neu1e = new float[entity_vec_dim_];
	float alpha = starting_alpha_;  // TODO
	for (int i = 0; i < num_training_rounds; ++i)
	{
		alpha *= 0.95f;
		if (alpha < starting_alpha_ * 0.01f)
			alpha = starting_alpha_ * 0.01f;
		if (seed < 10)
			printf("\r%d %f %d \n", i, alpha, num_samples_per_round);
		for (int j = 0; j < num_samples_per_round; ++j)
		{
			int edge_idx = ee_edge_sample_dist(generator);
			ns_trainer.TrainPrediction(ee_vecs0_[entity_net_.edges[edge_idx].va], entity_net_.edges[edge_idx].vb, ee_vecs1_,
				alpha, tmp_neu1e, generator);
			ns_trainer.TrainPrediction(ee_vecs0_[entity_net_.edges[edge_idx].vb], entity_net_.edges[edge_idx].va, ee_vecs1_,
				alpha, tmp_neu1e, generator);
		}
		if (seed < 10)
		{
			printf("\n");
			ns_trainer.CheckObject(ee_vecs0_[1], ee_vecs1_);
			//closeEntities(1);
		}
	}

	delete[] tmp_neu1e;
}

void JointTrainer::TrainDocEntityNetThreaded(const char *entity_vec_file_name, int num_rounds,
	int num_negative_samples, int num_threads, const char *dst_doc_vec_file_name)
{
	IOUtils::LoadVectors(entity_vec_file_name, num_entities_, entity_vec_dim_, ee_vecs0_);
	std::discrete_distribution<int> doc_entity_edge_sample_dist(entity_doc_net_.weights, 
		entity_doc_net_.weights + entity_doc_net_.num_edges);

	printf("initing model....\n");
	doc_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_docs_, entity_vec_dim_);
	ExpTable exp_table;
	NegativeSamplingTrainer ns_trainer(&exp_table, entity_vec_dim_, num_entities_, num_negative_samples, &entity_sample_dist_);
	printf("inited.\n");

	int sum_ed_edge_weights = MathUtils::Sum(entity_doc_net_.weights, entity_doc_net_.num_edges);

	int seeds[] = { 317, 7, 31, 297 };
	std::thread *threads = new std::thread[num_threads];
	for (int i = 0; i < num_threads; ++i)
	{
		int cur_seed = seeds[i];
		threads[i] = std::thread([=, &doc_entity_edge_sample_dist, &ns_trainer] { TrainDocEntityNet(cur_seed, num_rounds, 
			sum_ed_edge_weights, doc_entity_edge_sample_dist, ns_trainer); });
	}
	for (int i = 0; i < num_threads; ++i)
		threads[i].join();
	printf("\n");

	IOUtils::SaveVectors(doc_vecs_, entity_vec_dim_, num_docs_, dst_doc_vec_file_name);
}

void JointTrainer::TrainDocEntityNet(int seed, int num_rounds, int num_samples_per_round, std::discrete_distribution<int> &edge_sample_dist,
	NegativeSamplingTrainer &ns_trainer)
{
	printf("seed %d samples_per_round %d. training...\n", seed, num_samples_per_round);
	std::default_random_engine generator(seed);

	const float starting_alpha_ = 0.025f;

	float *tmp_neu1e = new float[entity_vec_dim_];
	float alpha = starting_alpha_;  // TODO
	for (int i = 0; i < num_rounds; ++i)
	{
		printf("round %d\n", i);
		alpha *= 0.96f;
		if (alpha < starting_alpha_ * 0.01f)
			alpha = starting_alpha_ * 0.01f;
		//if (seed < 10)
		//	printf("\r%d %f %d \n", i, alpha, num_samples_per_round);
		for (int j = 0; j < num_samples_per_round; ++j)
		{
			int edge_idx = edge_sample_dist(generator);
			ns_trainer.TrainPrediction(doc_vecs_[entity_doc_net_.edges[edge_idx].va], entity_doc_net_.edges[edge_idx].vb, ee_vecs0_,
				alpha, tmp_neu1e, generator, true, false);
		}
	}

	delete[] tmp_neu1e;
}
