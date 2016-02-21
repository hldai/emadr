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

	int *tmp_weights = new int[num_docs_];
	std::fill(tmp_weights, tmp_weights + num_docs_, 1);
	doc_sample_dist_ = std::discrete_distribution<int>(tmp_weights, tmp_weights + num_docs_);
	delete[] tmp_weights;

	assert(num_entities_ == 0 || doc_entity_net.num_vertices_right == num_entities_);

	num_entities_ = doc_entity_net.num_vertices_right;

	//int *entity_cnts = getEntitySampleWeights(doc_entity_net);
	int *entity_cnts = doc_entity_net.CountRightVertices();
	float *entity_sample_weights = NegativeSamplingTrainer::GetDefNegativeSamplingWeights(entity_cnts, num_entities_);
	entity_sample_dist_ = std::discrete_distribution<int>(entity_sample_weights, entity_sample_weights + num_entities_);
	delete[] entity_cnts;
	delete[] entity_sample_weights;

	dw_edge_sampler_ = new NetEdgeSampler(doc_words_file_name);
	num_words_ = dw_edge_sampler_->num_vertex_right();

	//AdjListNet doc_word_net;
	//doc_word_net.LoadBinFile(doc_words_file_name);
	//printf("%d\n", doc_word_net.num_adj_vertices[0]);
	//num_words_ = doc_word_net.num_vertices_right;
	//doc_word_net.ToEdgeNet(doc_word_net_);

	//int *word_cnts = doc_word_net.CountRightVertices();
	//float *word_sample_weights = NegativeSamplingTrainer::GetDefNegativeSamplingWeights(word_cnts, num_words_);
	//word_sample_dist_ = std::discrete_distribution<int>(word_sample_weights, word_sample_weights + num_words_);
	//delete[] word_cnts;
	//delete[] word_sample_weights;
}

JointTrainer::~JointTrainer()
{
	if (ee_vecs0_ != 0)
		MemUtils::Release(ee_vecs0_, entity_net_.num_vertices_left);
	if (ee_vecs1_ != 0)
		MemUtils::Release(ee_vecs1_, entity_net_.num_vertices_left);

	if (doc_vecs_ != 0)
		MemUtils::Release(doc_vecs_, num_docs_);
	if (word_vecs_ != 0)
		MemUtils::Release(word_vecs_, num_words_);

	if (dw_edge_sampler_ != 0)
		delete dw_edge_sampler_;
}

void JointTrainer::TrainCMThreaded(int vec_dim, int num_rounds, int num_threads, int num_negative_samples, float starting_alpha,
	float min_alpha, const char *dst_doc_vecs_file_name)
{
	starting_alpha_ = starting_alpha;
	min_alpha_ = min_alpha;
	entity_vec_dim_ = word_vec_dim_ = vec_dim;
	doc_vec_dim_ = 2 * vec_dim;

	std::discrete_distribution<int> ee_edge_sample_dist(entity_net_.weights,
		entity_net_.weights + entity_net_.num_edges);
	std::discrete_distribution<int> de_edge_sample_dist(entity_doc_net_.weights,
		entity_doc_net_.weights + entity_doc_net_.num_edges);
	std::discrete_distribution<int> dw_edge_sample_dist(doc_word_net_.weights,
		doc_word_net_.weights + doc_word_net_.num_edges);

	word_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_words_, word_vec_dim_);
	ee_vecs0_ = NegativeSamplingTrainer::GetInitedVecs0(num_entities_, entity_vec_dim_);
	ee_vecs1_ = NegativeSamplingTrainer::GetInitedVecs1(num_entities_, entity_vec_dim_);
	doc_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_docs_, doc_vec_dim_);

	ExpTable exp_table;
	NegativeSamplingTrainer entity_ns_trainer(&exp_table, num_entities_,
		num_negative_samples, &entity_sample_dist_);
	NegativeSamplingTrainer word_ns_trainer(&exp_table, num_words_,
		num_negative_samples, &word_sample_dist_);
	printf("inited.\n");

	int sum_ee_edge_weights = MathUtils::Sum(entity_net_.weights, entity_net_.num_edges);
	//sum_ee_edge_weights = 0;
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

	float *cm_params_e = NegativeSamplingTrainer::GetInitedCMParams(entity_vec_dim_);

	for (int i = 0; i < word_vec_dim_; ++i)
	{
		cm_params_e[i] = 0.1f;
	}
	float tmp_val = 0;
	for (int i = 0; i < vec_dim; ++i)
	{
		printf("%f ", cm_params_e[i]);
		tmp_val += cm_params_e[i] > 0.5 ? 1 - cm_params_e[i] : cm_params_e[i];
	}
	printf("\n%f\n", tmp_val);

	std::default_random_engine cm_params_gen(32942);

	int seeds[] = { 317, 7, 31, 297, 1238, 23487, 238593, 92384, 129380, 23848 };
	std::default_random_engine *generators = new std::default_random_engine[num_threads];
	for (int i = 0; i < num_threads; ++i)
		generators[i].seed(seeds[i]);

	const int num_turns = 1;
	std::thread *threads = new std::thread[num_threads];
	int cur_num_rounds = 4;
	for (int k = 0; k < num_turns; ++k)
	{
		if (k != 0)
		{
		//	printf("train cm params.\n");
		//	TrainCMParams(cm_params_e, doc_vecs_, ee_vecs0_, entity_doc_net_.edges, de_edge_sample_dist,
		//		entity_ns_trainer, cm_params_gen);
		//	TrainCMParams(cm_params_w, doc_vecs_, word_vecs_, doc_word_net_.edges, dw_edge_sample_dist,
		//		word_ns_trainer, cm_params_gen);
			float tmp_val = 0;
			for (int i = 0; i < vec_dim; ++i)
			{
				printf("%f ", cm_params_e[i]);
				tmp_val += cm_params_e[i] > 0.5 ? 1 - cm_params_e[i] : cm_params_e[i];
			}
			printf("\n%f\n", tmp_val);
		}

		if (k == num_turns - 1)
			cur_num_rounds = num_rounds;

		for (int i = 0; i < num_threads; ++i)
		{
			std::default_random_engine *cur_gen = generators + i;
			threads[i] = std::thread([&, cur_num_rounds, num_samples_per_round,
				cm_params_e, cur_gen]
			{
				TrainCM(cur_num_rounds, num_samples_per_round, cm_params_e,
					ee_edge_sample_dist, de_edge_sample_dist, dw_edge_sample_dist, net_sample_dist,
					entity_ns_trainer, word_ns_trainer, cur_gen, k == 1);
			});
		}
		for (int i = 0; i < num_threads; ++i)
			threads[i].join();
		printf("\n");
	}
	printf("done.\n");

	IOUtils::SaveVectors(doc_vecs_, doc_vec_dim_, num_docs_, dst_doc_vecs_file_name);
	delete[] cm_params_e;
	delete[] generators;
}

void JointTrainer::TrainCM(int num_rounds, int num_samples_per_round, float *cm_params_e,
	std::discrete_distribution<int> &ee_edge_sample_dist, std::discrete_distribution<int> &de_edge_sample_dist,
	std::discrete_distribution<int> &dw_edge_sample_dist, std::discrete_distribution<int> &net_sample_dist,
	NegativeSamplingTrainer &entity_ns_trainer, NegativeSamplingTrainer &word_ns_trainer, std::default_random_engine *rand_engine,
	bool train_cm_params)
{
	//printf("seed %d samples_per_round %d. training...\n", seed, num_samples_per_round);
	//const float min_alpha = starting_alpha_ * 0.001;
	std::default_random_engine &generator = *rand_engine;
	int total_num_samples = num_rounds * num_samples_per_round;

	float *tmp_neu1e = new float[doc_vec_dim_];
	float *tmp_cme = new float[entity_vec_dim_];

	float alpha = starting_alpha_;
	for (int i = 0; i < num_rounds; ++i)
	{
		printf("\rround %d, alpha %f", i, alpha);
		fflush(stdout);
		//alpha *= 0.96f;
		//if (alpha < starting_alpha_ * 0.1f)
		//	alpha = starting_alpha_ * 0.1f;
		for (int j = 0; j < num_samples_per_round; ++j)
		{
			int cur_num_samples = (i * num_samples_per_round) + j;
			if (cur_num_samples % 10000 == 10000 - 1)
			{
				alpha = starting_alpha_ + (min_alpha_ - starting_alpha_) * cur_num_samples / total_num_samples;
				if (alpha < min_alpha_)
					alpha = min_alpha_;
			}

			int net_idx = net_sample_dist(generator);
			if (net_idx == 0)
			{
				int edge_idx = ee_edge_sample_dist(generator);
				Edge &edge = entity_net_.edges[edge_idx];
				entity_ns_trainer.TrainEdge(entity_vec_dim_, ee_vecs0_[edge.va], edge.vb, ee_vecs1_,
					alpha, tmp_neu1e, generator);
				entity_ns_trainer.TrainEdge(entity_vec_dim_, ee_vecs0_[edge.vb], edge.va, ee_vecs1_,
					alpha, tmp_neu1e, generator);
			}
			else if (net_idx == 1)
			{
				int edge_idx = de_edge_sample_dist(generator);
				Edge &edge = entity_doc_net_.edges[edge_idx];
				entity_ns_trainer.TrainEdgeCM(entity_vec_dim_, doc_vecs_[edge.va], edge.vb, ee_vecs0_,
					cm_params_e, false, alpha, tmp_neu1e, tmp_cme, generator, true, true, train_cm_params);
			}
			else
			{
				int edge_idx = dw_edge_sample_dist(generator);
				Edge &edge = doc_word_net_.edges[edge_idx];
				word_ns_trainer.TrainEdgeCM(word_vec_dim_, doc_vecs_[edge.va], edge.vb, word_vecs_,
					cm_params_e, true, alpha, tmp_neu1e, tmp_cme, generator, true, true, train_cm_params);
			}
		}
	}

	delete[] tmp_neu1e;
	delete[] tmp_cme;
}

void JointTrainer::TrainCMParams(float *cm_params, bool complement, float **vecs0, float **vecs1, Edge *edges,
	std::discrete_distribution<int> &edge_sample_dist, NegativeSamplingTrainer &ns_trainer,
	std::default_random_engine &generator)
{
	const int num_samples = 100000;
	const float starting_alpha = 0.03f;
	float *tmp_cme = new float[doc_vec_dim_];

	float alpha = starting_alpha;
	for (int i = 0; i < num_samples; ++i)
	{
		if (i % 100 == 100 - 1)
		{
			alpha = starting_alpha + (min_alpha_ - starting_alpha) * i / num_samples;
			if (alpha < min_alpha_)
				alpha = min_alpha_;
		}

		int edge_idx = edge_sample_dist(generator);
		Edge &edge = edges[edge_idx];
		ns_trainer.TrainEdgeCM(word_vec_dim_, vecs0[edge.va], edge.vb, vecs1,
			cm_params, complement, alpha, 0, tmp_cme, generator, false, false, true);
	}
	delete[] tmp_cme;
}

void JointTrainer::JointTrainingOMLThreaded(int vec_dim, int num_rounds, int num_threads, int num_negative_samples, 
	float starting_alpha, float ws_rate, float min_alpha, const char *dst_dedw_vec_file_name, 
	const char *dst_mixed_vecs_file_name, const char *dst_word_vecs_file_name, const char *dst_entity_vecs_file_name)
{
	starting_alpha_ = starting_alpha;
	min_alpha_ = min_alpha;
	doc_vec_dim_ = entity_vec_dim_ = word_vec_dim_ = vec_dim;

	std::discrete_distribution<int> ee_edge_sample_dist(entity_net_.weights,
		entity_net_.weights + entity_net_.num_edges);
	std::discrete_distribution<int> de_edge_sample_dist(entity_doc_net_.weights,
		entity_doc_net_.weights + entity_doc_net_.num_edges);
	//std::discrete_distribution<int> dw_edge_sample_dist(doc_word_net_.weights,
	//	doc_word_net_.weights + doc_word_net_.num_edges);
	std::uniform_int_distribution<int> dwe_sample_dist(0, num_docs_ - 1);
	std::bernoulli_distribution we_sample_dist(ws_rate);

	printf("initing model....\n");
	word_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_words_, doc_vec_dim_);
	dw_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_docs_, doc_vec_dim_);

	ee_vecs0_ = NegativeSamplingTrainer::GetInitedVecs0(num_entities_, doc_vec_dim_);
	ee_vecs1_ = NegativeSamplingTrainer::GetInitedVecs1(num_entities_, doc_vec_dim_);
	de_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_docs_, doc_vec_dim_);

	doc_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_docs_, doc_vec_dim_);

	ExpTable exp_table;
	NegativeSamplingTrainer entity_ns_trainer(&exp_table, num_entities_,
		num_negative_samples, &entity_sample_dist_);
	//NegativeSamplingTrainer word_ns_trainer(&exp_table, num_words_,
	//	num_negative_samples, &word_sample_dist_);
	NegativeSamplingTrainer word_ns_trainer(&exp_table, num_words_,
		num_negative_samples, dw_edge_sampler_->neg_sampling_dist());
	// for both dw and de
	NegativeSamplingTrainer doc_ns_trainer(&exp_table, num_docs_,
		num_negative_samples, &doc_sample_dist_);
	printf("inited.\n");

	int sum_ee_edge_weights = MathUtils::Sum(entity_net_.weights, entity_net_.num_edges);
	int sum_de_edge_weights = MathUtils::Sum(entity_doc_net_.weights, entity_doc_net_.num_edges);
	//int sum_dw_edge_weights = MathUtils::Sum(doc_word_net_.weights, doc_word_net_.num_edges);
	int sum_dw_edge_weights = dw_edge_sampler_->sum_weights();
	int sum_dew_edge_weights = num_docs_;
	sum_dew_edge_weights = 0;
	int sum_weights = sum_ee_edge_weights + sum_de_edge_weights + sum_dw_edge_weights + sum_dew_edge_weights;
	int num_samples_per_round = sum_weights;
	//int num_samples_per_round = sum_dw_edge_weights;
	//int num_samples_per_round = sum_ee_edge_weights + sum_de_edge_weights;

	float weight_portions[] = { (float)sum_ee_edge_weights / sum_weights,
		(float)sum_de_edge_weights / sum_weights, (float)sum_dw_edge_weights / sum_weights, 
		(float)sum_dew_edge_weights / sum_weights };
	printf("net distribution: %f %f %f %f\n", weight_portions[0], weight_portions[1],
		weight_portions[2], weight_portions[3]);
	std::discrete_distribution<int> net_sample_dist(weight_portions, weight_portions + 4);
	//std::discrete_distribution<int> net_sample_dist{ 0, 0, 1 };

	int seeds[] = { 317, 7, 31, 297, 1238, 23487, 238593, 92384, 129380, 23848 };
	std::thread *threads = new std::thread[num_threads];
	for (int i = 0; i < num_threads; ++i)
	{
		int cur_seed = seeds[i];
		threads[i] = std::thread([&, cur_seed, num_rounds, num_samples_per_round]
		{
			JointTrainingOML(cur_seed, num_rounds, num_samples_per_round, ee_edge_sample_dist,
				de_edge_sample_dist, dwe_sample_dist, net_sample_dist, 
				we_sample_dist, entity_ns_trainer, word_ns_trainer, doc_ns_trainer);
		});
	}
	for (int i = 0; i < num_threads; ++i)
		threads[i].join();
	printf("\n");

	saveConcatnatedVectors(de_vecs_, dw_vecs_, num_docs_, doc_vec_dim_, dst_dedw_vec_file_name);

	if (dst_mixed_vecs_file_name != 0)
		IOUtils::SaveVectors(doc_vecs_, doc_vec_dim_, num_docs_, dst_mixed_vecs_file_name);
	if (dst_word_vecs_file_name != 0)
		IOUtils::SaveVectors(word_vecs_, doc_vec_dim_, num_words_, dst_word_vecs_file_name);
	if (dst_entity_vecs_file_name != 0)
		IOUtils::SaveVectors(ee_vecs0_, doc_vec_dim_, num_entities_, dst_entity_vecs_file_name);
}

//void JointTrainer::JointTrainingOML(int seed, int num_rounds, int num_samples_per_round,
//	std::discrete_distribution<int>& ee_edge_sample_dist, std::discrete_distribution<int>& de_edge_sample_dist,
//	std::discrete_distribution<int>& dw_edge_sample_dist, std::uniform_int_distribution<int> &dwe_sample_dist,
//	std::discrete_distribution<int>& net_sample_dist, std::bernoulli_distribution &we_sample_dist,
//	NegativeSamplingTrainer &entity_ns_trainer, NegativeSamplingTrainer &word_ns_trainer,
//	NegativeSamplingTrainer &doc_ns_trainer)
void JointTrainer::JointTrainingOML(int seed, int num_rounds, int num_samples_per_round, 
	std::discrete_distribution<int>& ee_edge_sample_dist, std::discrete_distribution<int>& de_edge_sample_dist, 
	std::uniform_int_distribution<int> &dwe_sample_dist, 
	std::discrete_distribution<int>& net_sample_dist, std::bernoulli_distribution &we_sample_dist, 
	NegativeSamplingTrainer &entity_ns_trainer, NegativeSamplingTrainer &word_ns_trainer,
	NegativeSamplingTrainer &doc_ns_trainer)
{
	//printf("seed %d samples_per_round %d. training...\n", seed, num_samples_per_round);
	std::default_random_engine generator(seed);

	//const float min_alpha = starting_alpha_ * 0.001;
	int total_num_samples = num_rounds * num_samples_per_round;

	float *tmp_neu1e = new float[doc_vec_dim_];

	float alpha = starting_alpha_;  // TODO
	for (int i = 0; i < num_rounds; ++i)
	{
		printf("\rround %d, alpha %f", i, alpha);
		fflush(stdout);
		//alpha *= 0.96f;
		//if (alpha < starting_alpha_ * 0.1f)
		//	alpha = starting_alpha_ * 0.1f;
		//if (seed < 10)
		//	printf("\r%d %f %d \n", i, alpha, num_samples_per_round);
		for (int j = 0; j < num_samples_per_round; ++j)
		{
			int cur_num_samples = (i * num_samples_per_round) + j;
			if (cur_num_samples % 10000 == 10000 - 1)
				alpha = starting_alpha_ + (min_alpha_ - starting_alpha_) * cur_num_samples / total_num_samples;

			int net_idx = net_sample_dist(generator);
			if (net_idx == 0)
			{
				int edge_idx = ee_edge_sample_dist(generator);
				Edge &edge = entity_net_.edges[edge_idx];
				entity_ns_trainer.TrainEdge(doc_vec_dim_, ee_vecs0_[edge.va], edge.vb, ee_vecs1_,
					alpha, tmp_neu1e, generator);
				entity_ns_trainer.TrainEdge(doc_vec_dim_, ee_vecs0_[edge.vb], edge.va, ee_vecs1_,
					alpha, tmp_neu1e, generator);
			}
			else if (net_idx == 1)
			{
				int edge_idx = de_edge_sample_dist(generator);
				Edge &edge = entity_doc_net_.edges[edge_idx];
				entity_ns_trainer.TrainEdge(doc_vec_dim_, de_vecs_[edge.va], edge.vb, ee_vecs0_,
					alpha, tmp_neu1e, generator);
			}
			else if (net_idx == 2)
			{
				int va = 0, vb = 0;
				dw_edge_sampler_->SampleEdge(va, vb, generator);
				//int edge_idx = dw_edge_sample_dist(generator);
				//Edge &edge = doc_word_net_.edges[edge_idx];
				//word_ns_trainer.TrainEdge(doc_vec_dim_, dw_vecs_[edge.va], edge.vb, word_vecs_,
				//	alpha, tmp_neu1e, generator);
				word_ns_trainer.TrainEdge(doc_vec_dim_, dw_vecs_[va], vb, word_vecs_,
					alpha, tmp_neu1e, generator);
			}
			else
			{
				int doc_idx = dwe_sample_dist(generator);
				doc_ns_trainer.TrainEdge(doc_vec_dim_, de_vecs_[doc_idx], doc_idx, dw_vecs_,
					alpha, tmp_neu1e, generator);
				doc_ns_trainer.TrainEdge(doc_vec_dim_, dw_vecs_[doc_idx], doc_idx, de_vecs_,
					alpha, tmp_neu1e, generator);
				//bool is_dw = we_sample_dist(generator);
				//if (is_dw)
				//{
				//	doc_ns_trainer.TrainPrediction(doc_vecs_[doc_idx], doc_idx, dw_vecs_,
				//		alpha, tmp_neu1e, generator);
				//}
				//else
				//{
				//	doc_ns_trainer.TrainPrediction(doc_vecs_[doc_idx], doc_idx, de_vecs_,
				//		alpha, tmp_neu1e, generator);
				//}
			}
		}
	}

	delete[] tmp_neu1e;
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
	NegativeSamplingTrainer entity_ns_trainer(&exp_table, num_entities_, 
		num_negative_samples, &entity_sample_dist_);
	NegativeSamplingTrainer word_ns_trainer(&exp_table, num_words_,
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
	const float min_alpha = starting_alpha_ * 0.001f;
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
				entity_ns_trainer.TrainEdge(entity_vec_dim_, ee_vecs0_[edge.va], edge.vb, ee_vecs1_,
					alpha, tmp_entity_neu1e, generator);
				entity_ns_trainer.TrainEdge(entity_vec_dim_, ee_vecs0_[edge.vb], edge.va, ee_vecs1_,
					alpha, tmp_entity_neu1e, generator);
			}
			else if (net_idx == 1)
			{
				int edge_idx = de_edge_sample_dist(generator);
				Edge &edge = entity_doc_net_.edges[edge_idx];
				entity_ns_trainer.TrainEdge(entity_vec_dim_, doc_vecs_[edge.va], edge.vb, ee_vecs0_,
					alpha, tmp_entity_neu1e, generator);
			}
			else
			{
				int edge_idx = dw_edge_sample_dist(generator);
				Edge &edge = doc_word_net_.edges[edge_idx];
				word_ns_trainer.TrainEdge(word_vec_dim_, doc_vecs_[edge.va] + doc_vec_dim_ - word_vec_dim_,
					edge.vb, word_vecs_, alpha, tmp_word_neu1e, generator);
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
	NegativeSamplingTrainer ns_trainer(&exp_table, num_entities_, num_negative_samples, &entity_sample_dist_);
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
			ns_trainer.TrainEdge(entity_vec_dim_, ee_vecs0_[entity_net_.edges[edge_idx].va], entity_net_.edges[edge_idx].vb, ee_vecs1_,
				alpha, tmp_neu1e, generator);
			ns_trainer.TrainEdge(entity_vec_dim_, ee_vecs0_[entity_net_.edges[edge_idx].vb], entity_net_.edges[edge_idx].va, ee_vecs1_,
				alpha, tmp_neu1e, generator);
		}
		if (seed < 10)
		{
			printf("\n");
			ns_trainer.CheckObject(entity_vec_dim_, ee_vecs0_[1], ee_vecs1_);
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
	NegativeSamplingTrainer ns_trainer(&exp_table, num_entities_, num_negative_samples, &entity_sample_dist_);
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
			ns_trainer.TrainEdge(entity_vec_dim_, doc_vecs_[entity_doc_net_.edges[edge_idx].va],
				entity_doc_net_.edges[edge_idx].vb, ee_vecs0_,
				alpha, tmp_neu1e, generator, true, false);
		}
	}

	delete[] tmp_neu1e;
}

void JointTrainer::saveConcatnatedVectors(float **vecs0, float **vecs1, int num_vecs, int vec_dim, 
	const char *dst_file_name)
{
	FILE *fp = fopen(dst_file_name, "wb");
	assert(fp != 0);

	fwrite(&num_vecs, 4, 1, fp);
	int full_vec_dim = vec_dim << 1;
	fwrite(&full_vec_dim, 4, 1, fp);

	for (int i = 0; i < num_vecs; ++i)
	{
		fwrite(vecs0[i], 4, vec_dim, fp);
		fwrite(vecs1[i], 4, vec_dim, fp);
	}

	fclose(fp);
}
