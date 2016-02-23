#include "eadocvectrainer.h"

#include <cassert>
#include <thread>

#include "negative_sampling_trainer.h"
#include "io_utils.h"

EADocVecTrainer::EADocVecTrainer(int num_rounds, int num_threads, int num_negative_samples, 
	float starting_alpha, float min_alpha) : num_rounds_(num_rounds), num_threads_(num_threads),
	num_negative_samples_(num_negative_samples), starting_alpha_(starting_alpha), min_alpha_(min_alpha)
{
}

void EADocVecTrainer::AllJointThreaded(const char *ee_net_file_name, const char *doc_entity_net_file_name,
	const char *doc_words_file_name, const char *entity_cnts_file, const char *word_cnts_file, 
	int vec_dim, const char *dst_dedw_vec_file_name, const char *dst_word_vecs_file_name,
	const char *dst_entity_vecs_file_name)
{
	initDocEntityNet(doc_entity_net_file_name);
	initDocWordNet(doc_words_file_name);
	initEntityEntityNet(ee_net_file_name);

	entity_vec_dim_ = word_vec_dim_ = vec_dim;

	printf("initing model....\n");
	word_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_words_, word_vec_dim_);
	dw_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_docs_, word_vec_dim_);

	ee_vecs0_ = NegativeSamplingTrainer::GetInitedVecs0(num_entities_, entity_vec_dim_);
	ee_vecs1_ = NegativeSamplingTrainer::GetInitedVecs1(num_entities_, entity_vec_dim_);
	de_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_docs_, entity_vec_dim_);

	ExpTable exp_table;
	NegativeSamplingTrainer entity_ns_trainer(&exp_table, num_negative_samples_,
		entity_cnts_file);
	NegativeSamplingTrainer word_ns_trainer(&exp_table, num_negative_samples_,
		word_cnts_file);
	printf("inited.\n");

	int sum_ee_edge_weights = ee_edge_sampler_->sum_weights();
	int sum_de_edge_weights = de_edge_sampler_->sum_weights();
	int sum_dw_edge_weights = dw_edge_sampler_->sum_weights();
	long long sum_weights = sum_ee_edge_weights + sum_de_edge_weights + sum_dw_edge_weights;
	long long num_samples_per_round = sum_weights / 2;
	//long long num_samples_per_round = sum_dw_edge_weights;
	//int num_samples_per_round = sum_ee_edge_weights + sum_de_edge_weights;

	printf("edge_weights: %d %d %d\n", sum_ee_edge_weights, sum_de_edge_weights, sum_dw_edge_weights);
	printf("%lld samples per round\n", num_samples_per_round);

	float weight_portions[] = { (float)sum_ee_edge_weights / sum_weights,
		(float)sum_de_edge_weights / sum_weights, (float)sum_dw_edge_weights / sum_weights };
	printf("net distribution: %f %f %f\n", weight_portions[0], weight_portions[1],
		weight_portions[2]);
	std::discrete_distribution<int> net_sample_dist(weight_portions, weight_portions + 3);
	//std::discrete_distribution<int> net_sample_dist{ 0, 0, 1 };

	int seeds[] = { 317, 7, 31, 297, 1238, 23487, 238593, 92384, 129380, 23848 };
	std::thread *threads = new std::thread[num_threads_];
	for (int i = 0; i < num_threads_; ++i)
	{
		int cur_seed = seeds[i];
		threads[i] = std::thread([&, cur_seed, num_samples_per_round]
		{
			allJoint(cur_seed, num_samples_per_round, net_sample_dist, entity_ns_trainer, word_ns_trainer);
		});
	}
	for (int i = 0; i < num_threads_; ++i)
		threads[i].join();
	printf("\n");

	//printf("dw0: %d\n", dw_edge_sampler_->CountZeros());
	//printf("de0: %d\n", de_edge_sampler_->CountZeros());
	//printf("ee0: %d\n", ee_edge_sampler_->CountZeros());

	saveConcatnatedVectors(de_vecs_, dw_vecs_, num_docs_, entity_vec_dim_, dst_dedw_vec_file_name);
	IOUtils::SaveVectors(word_vecs_, word_vec_dim_, num_words_, dst_word_vecs_file_name);
	IOUtils::SaveVectors(ee_vecs0_, entity_vec_dim_, num_entities_, dst_entity_vecs_file_name);
}

void EADocVecTrainer::TrainDocWord(const char *doc_words_file_name, const char *word_cnts_file, int vec_dim,
	const char *dst_doc_vecs_file_name, const char *dst_word_vecs_file_name)
{
	initDocWordNet(doc_words_file_name);

	word_vec_dim_ = vec_dim;

	printf("initing model....\n");
	word_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_words_, word_vec_dim_);
	dw_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_docs_, word_vec_dim_);
	printf("inited.\n");

	trainDocWordMT(word_cnts_file, true, dst_doc_vecs_file_name);

	if (dst_word_vecs_file_name != 0)
		IOUtils::SaveVectors(word_vecs_, word_vec_dim_, num_words_, dst_word_vecs_file_name);
}

void EADocVecTrainer::TrainDocWordFixedWordVecs(const char *doc_words_file_name, const char *word_cnts_file,
	const char *word_vecs_file_name, int vec_dim, const char *dst_doc_vecs_file_name)
{
	initDocWordNet(doc_words_file_name);

	word_vec_dim_ = vec_dim;

	printf("initing model....\n");
	int tmp_num = 0, tmp_dim = 0;
	IOUtils::LoadVectors(word_vecs_file_name, tmp_num, tmp_dim, word_vecs_);
	if (tmp_num != num_words_ || tmp_dim != vec_dim)
	{
		printf("num words: %d %d\n", num_words_, tmp_num);
		printf("vec dim: %d %d\n", vec_dim, tmp_dim);
		return;
	}
	//dw_vecs_ = NegativeSamplingTrainer::GetInitedVecs0(num_docs_, word_vec_dim_);
	dw_vecs_ = NegativeSamplingTrainer::GetInitedVecs1(num_docs_, word_vec_dim_);
	printf("inited.\n");

	trainDocWordMT(word_cnts_file, false, dst_doc_vecs_file_name);
}

void EADocVecTrainer::saveConcatnatedVectors(float **vecs0, float **vecs1, int num_vecs, int vec_dim,
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

void EADocVecTrainer::allJoint(int seed, long long num_samples_per_round, std::discrete_distribution<int> &net_sample_dist,
	NegativeSamplingTrainer &entity_ns_trainer, NegativeSamplingTrainer &word_ns_trainer)
{
	//printf("seed %d samples_per_round %d. training...\n", seed, num_samples_per_round);
	std::default_random_engine generator(seed);

	RandGen rand_gen(seed);

	//const float min_alpha = starting_alpha_ * 0.001;
	long long total_num_samples = num_rounds_ * num_samples_per_round;

	float *tmp_neu1e = new float[entity_vec_dim_];

	float alpha = starting_alpha_;
	for (int i = 0; i < num_rounds_; ++i)
	{
		printf("\rround %d, alpha %f", i, alpha);
		fflush(stdout);
		for (int j = 0; j < num_samples_per_round; ++j)
		{
			long long cur_num_samples = (i * num_samples_per_round) + j;
			if (cur_num_samples % 10000 == 10000 - 1)
				alpha = starting_alpha_ + (min_alpha_ - starting_alpha_) * cur_num_samples / total_num_samples;

			int net_idx = net_sample_dist(generator);
			int va = 0, vb = 0;
			if (net_idx == 0)
			{
				ee_edge_sampler_->SampleEdge(va, vb, generator, rand_gen);
				entity_ns_trainer.TrainEdge(entity_vec_dim_, ee_vecs0_[va], vb, ee_vecs1_,
					alpha, tmp_neu1e, generator);
				entity_ns_trainer.TrainEdge(entity_vec_dim_, ee_vecs0_[vb], va, ee_vecs1_,
					alpha, tmp_neu1e, generator);
			}
			else if (net_idx == 1)
			{
				de_edge_sampler_->SampleEdge(va, vb, generator, rand_gen);
				entity_ns_trainer.TrainEdge(entity_vec_dim_, de_vecs_[va], vb, ee_vecs0_,
					alpha, tmp_neu1e, generator);
			}
			else if (net_idx == 2)
			{
				dw_edge_sampler_->SampleEdge(va, vb, generator, rand_gen);
				word_ns_trainer.TrainEdge(word_vec_dim_, dw_vecs_[va], vb, word_vecs_,
					alpha, tmp_neu1e, generator);
			}
		}
	}

	delete[] tmp_neu1e;
}

void EADocVecTrainer::trainDocWordMT(const char *word_cnts_file, bool update_word_vecs, const char *dst_doc_vecs_file_name)
{
	ExpTable exp_table;
	NegativeSamplingTrainer word_ns_trainer(&exp_table, num_negative_samples_, word_cnts_file);

	int sum_dw_edge_weights = dw_edge_sampler_->sum_weights();
	long long num_samples_per_round = sum_dw_edge_weights / 2;

	printf("%lld samples per round\n", num_samples_per_round);

	int seeds[] = { 317, 7, 31, 297, 1238, 23487, 238593, 92384, 129380, 23848 };
	std::thread *threads = new std::thread[num_threads_];
	for (int i = 0; i < num_threads_; ++i)
	{
		int cur_seed = seeds[i];
		threads[i] = std::thread([&, cur_seed, num_samples_per_round]
		{
			trainDocWordNet(cur_seed, num_samples_per_round, update_word_vecs, word_ns_trainer);
		});
	}
	for (int i = 0; i < num_threads_; ++i)
		threads[i].join();
	printf("\n");

	IOUtils::SaveVectors(dw_vecs_, word_vec_dim_, num_docs_, dst_doc_vecs_file_name);
}

void EADocVecTrainer::trainDocWordNet(int seed, long long num_samples_per_round, bool update_word_vecs, 
	NegativeSamplingTrainer &word_ns_trainer)
{
	//printf("seed %d samples_per_round %d. training...\n", seed, num_samples_per_round);
	std::default_random_engine generator(seed);

	RandGen rand_gen(seed);

	//const float min_alpha = starting_alpha_ * 0.001;
	long long total_num_samples = num_rounds_ * num_samples_per_round;

	float *tmp_neu1e = new float[word_vec_dim_];

	float alpha = starting_alpha_;
	int va = 0, vb = 0;
	for (int i = 0; i < num_rounds_; ++i)
	{
		printf("\rround %d, alpha %f", i, alpha);
		fflush(stdout);
		for (int j = 0; j < num_samples_per_round; ++j)
		{
			long long cur_num_samples = (i * num_samples_per_round) + j;
			if (cur_num_samples % 10000 == 10000 - 1)
				alpha = starting_alpha_ + (min_alpha_ - starting_alpha_) * cur_num_samples / total_num_samples;

			dw_edge_sampler_->SampleEdge(va, vb, generator, rand_gen);
			word_ns_trainer.TrainEdge(word_vec_dim_, dw_vecs_[va], vb, word_vecs_,
				alpha, tmp_neu1e, generator, true, update_word_vecs);
		}
	}

	delete[] tmp_neu1e;
}
