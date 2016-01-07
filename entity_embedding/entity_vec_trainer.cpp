// entity_vec_trainer.cpp
//
//  Created on: Dec 18, 2015
//      Author: dhl

#include "entity_vec_trainer.h"

#include <cassert>
#include <cstdio>
#include <random>
#include <iostream>
#include <thread>
#include <cmath>

#include "math_utils.h"

const float EntityVecTrainer::kMaxExp = 6;
const float EntityVecTrainer::kDefAlpha = 0.035f;

EntityVecTrainer::EntityVecTrainer(const char *edge_list_file_name)
{
	loadEdgesFromFile(edge_list_file_name);
	initExpTable();

	starting_alpha_ = kDefAlpha;
}

EntityVecTrainer::~EntityVecTrainer()
{
	if (num_edges_ != 0)
	{
		delete[] edges_;
		delete[] weights_;
		delete[] entity_cnts_;

		for (int i = 0; i < num_entities_; ++i)
			delete[] adj_list_[i];
		delete[] adj_list_;
		delete[] num_adj_vertices_;
	}

	if (syn0_ != 0)
	{
		for (int i = 0; i < num_entities_; ++i)
		{
			delete[] syn0_[i];
			delete[] syn1_[i];
		}
		delete[] syn0_;
		delete[] syn1_;
	}

	delete[] exp_table_;
}

void EntityVecTrainer::ThreadedTrain(int vec_dim, int num_rounds, int num_threads, int num_negative_samples,
	const char *dst_entity_vec_file_name, const char *dst_output_vecs_file_name)
{
	vec_dim_ = vec_dim;
	num_training_rounds_ = num_rounds;
	num_negative_samples_ = num_negative_samples;

	printf("initing net....\n");
	initNet();
	printf("net inited.\n");

	edge_sample_dist_ = std::discrete_distribution<int>(weights_, weights_ + num_edges_);
	entity_sample_dist_ = std::discrete_distribution<int>(entity_sample_weights_, entity_sample_weights_ + num_entities_);

	//sampled_cnts_ = new int[num_entities_];
	//std::fill(sampled_cnts_, sampled_cnts_ + num_entities_, 0);


	int seeds[] = { 317, 7, 31, 297 };
	std::thread *threads = new std::thread[num_threads];
	for (int i = 0; i < num_threads; ++i)
	{
		int cur_seed = seeds[i];
		threads[i] = std::thread([=] { Train(cur_seed); });
	}
	for (int i = 0; i < num_threads; ++i)
		threads[i].join();
	printf("\n");

	//	for (int i = 0; i < 100; ++i) {
	//		printf("%d %d\n", entity_cnts_[i], sampled_cnts_[i]);
	//	}

	//int num_zeros = 0;
	//for (int i = 0; i < num_entities_; ++i) {
	//	if (sampled_cnts_[i] == 0)
	//		++num_zeros;
	//}
	//printf("zeros %d\n", num_zeros);

	writeVectorsToFile(syn0_, vec_dim_, num_entities_, dst_entity_vec_file_name);
	if (dst_output_vecs_file_name != 0)
		writeVectorsToFile(syn1_, vec_dim_, num_entities_, dst_output_vecs_file_name);

	//delete[] sampled_cnts_;
}

void EntityVecTrainer::Train(int seed)
{
	printf("%d training...\n", seed);
	std::default_random_engine generator(seed);

	float *tmp_neu1e = new float[vec_dim_];
	float alpha = starting_alpha_;
	for (int i = 0; i < num_training_rounds_; ++i)
	{
		alpha *= 0.95f;
		if (alpha < starting_alpha_ * 0.01f)
			alpha = starting_alpha_ * 0.01f;
		if (seed < 10)
			printf("\r%d %f %d \n", i, alpha, sum_weights_);
		for (int j = 0; j < sum_weights_; ++j)
		{
			//int cur_training_idx = i * sum_weights_ + j + 1;
			//if (cur_training_idx % kDefAlphaUpdateFreq == 0)
			//{
			//	alpha *= 0.98f;
			//	if (alpha < starting_alpha_ * 0.005f)
			//		alpha = starting_alpha_ * 0.005f;
			//	printf("\r%d %f %d %d", i, alpha, j, sum_weights_);
			//}

			int edge_idx = edge_sample_dist_(generator);
			trainWithEdge(edges_[edge_idx].va, edges_[edge_idx].vb, alpha, tmp_neu1e, generator);
			trainWithEdge(edges_[edge_idx].vb, edges_[edge_idx].va, alpha, tmp_neu1e, generator);
		}
		if (seed < 10)
		{
			printf("\n");
			closeEntities(1);
		}
	}

	delete[] tmp_neu1e;
	//int num_samples = sum_entity_cnts_ * 3;
	//for (int i = 0; i < num_samples; ++i) {
	//	int entity_sample_idx = entity_sample_dist_(generator);
	//	++sampled_cnts_[entity_sample_idx];
	//}
}

void EntityVecTrainer::trainWithEdge(int va, int vb, float alpha, float *tmp_neu1e,
	std::default_random_engine &generator)
{
	for (int i = 0; i < vec_dim_; ++i)
		tmp_neu1e[i] = 0;

	int target = vb;
	int label = 1;
	for (int i = 0; i < num_negative_samples_ + 1; ++i)
	{
		if (i != 0)
		{
			target = entity_sample_dist_(generator);
			if (target == vb) continue;

			//int tmpv = va < vb ? va : vb;
			//if (valueInArray(target, adj_list_[tmpv], num_adj_vertices_[tmpv]))
			//{
			//	--i;
			//	continue;
			//}
			
			label = 0;
		}

		float dot_product = MathUtils::DotProduct(syn0_[va], syn1_[target], vec_dim_);
		float g = 0;
		if (dot_product > kMaxExp)
			g = (label - 1) * alpha;
		else if (dot_product < -kMaxExp)
			g = (label - 0) * alpha;
		else
			g = (label - getSigmaValue(dot_product)) * alpha;

		for (int j = 0; j < vec_dim_; ++j)
			tmp_neu1e[j] += g * syn1_[target][j];
		for (int j = 0; j < vec_dim_; ++j)
			syn1_[target][j] += g * syn0_[va][j];
	}

	for (int j = 0; j < vec_dim_; ++j)
		syn0_[va][j] += tmp_neu1e[j];
}

void EntityVecTrainer::loadEdgesFromFile(const char *edge_list_file_name)
{
	if (edge_list_file_name == 0)
		return;

	printf("Loading edge list file...\n");

	FILE *fp = fopen(edge_list_file_name, "r");
	assert(fp != 0);

	fscanf(fp, "%d %d %d", &num_entities_, &num_edges_, &sum_weights_);
	edges_ = new Edge[num_edges_];
	weights_ = new int[num_edges_];

	entity_cnts_ = new int[num_entities_];
	std::fill(entity_cnts_, entity_cnts_ + num_entities_, 1);
	sum_entity_cnts_ += num_entities_;
	num_adj_vertices_ = new int[num_entities_];
	std::fill(num_adj_vertices_, num_adj_vertices_ + num_entities_, 0);
	for (int i = 0; i < num_edges_; ++i)
	{
		fscanf(fp, "%d %d %d", &edges_[i].va, &edges_[i].vb, &weights_[i]);
		--edges_[i].va;
		--edges_[i].vb;

		if (weights_[i] > 2)
			++num_adj_vertices_[edges_[i].va];

		entity_cnts_[edges_[i].va] += weights_[i];
		entity_cnts_[edges_[i].vb] += weights_[i];
		sum_entity_cnts_ += weights_[i] * 2;
	}

	fclose(fp);

	adj_list_ = new int*[num_entities_];
	for (int i = 0; i < num_entities_; ++i)
	{
		if (num_adj_vertices_[i] == 0)
			adj_list_[i] = 0;
		else
			adj_list_[i] = new int[num_adj_vertices_[i]];
	}

	int pre_v = -1;
	int vcnt = 0;
	for (int i = 0; i < num_edges_; ++i)
	{
		int cur_v = edges_[i].va;
		if (cur_v != pre_v)
			vcnt = 0;
		if (weights_[i] > 3)
			adj_list_[cur_v][vcnt++] = edges_[i].vb;
		pre_v = cur_v;
	}

	entity_sample_weights_ = new double[num_entities_];
	double div = pow(sum_entity_cnts_, 0.75);
	for (int i = 0; i < num_entities_; ++i)
		entity_sample_weights_[i] = pow(entity_cnts_[i], 0.75) / div;

	//for (int i = 0; i < num_adj_vertices_[1]; ++i)
	//	printf("%d ", adj_list_[1][i]);
	//printf("\n");

	printf("Done.\n");
}

void EntityVecTrainer::initExpTable()
{
	exp_table_ = new float[kExpTableSize + 1];
	for (int i = 0; i < kExpTableSize; ++i)
	{
		exp_table_[i] = exp((i / (float)kExpTableSize * 2 - 1) * kMaxExp);
		exp_table_[i] = exp_table_[i] / (exp_table_[i] + 1);
	}
}

void EntityVecTrainer::initNet()
{
	syn0_ = new float*[num_entities_];
	syn1_ = new float*[num_entities_];
	for (int i = 0; i < num_entities_; ++i)
	{
		syn0_[i] = new float[vec_dim_];
		for (int j = 0; j < vec_dim_; ++j)
			syn0_[i][j] = ((float)rand() / RAND_MAX - 0.5f) / vec_dim_;
		
		syn1_[i] = new float[vec_dim_];
		std::fill(syn1_[i], syn1_[i] + vec_dim_, 0.0f);
	}
}

void EntityVecTrainer::writeVectorsToFile(float **vecs, int vec_len, int num_vecs, const char *dst_file_name)
{
	FILE *fp = fopen(dst_file_name, "wb");
	assert(fp != 0);

	fwrite(&num_vecs, 4, 1, fp);
	fwrite(&vec_len, 4, 1, fp);
	
	for (int i = 0; i < num_vecs; ++i)
		fwrite(vecs[i], 4, vec_len, fp);

	fclose(fp);
}

void EntityVecTrainer::closeEntities(int entity_index)
{
	const int max_num_edges = 100;
	int vv[max_num_edges], weights[max_num_edges];
	int ecnt = 0;
	for (int i = 0; i < num_edges_ && edges_[i].va <= entity_index; ++i)
	{
		if (edges_[i].va == entity_index)
		{
			vv[ecnt] = edges_[i].vb;
			weights[ecnt++] = weights_[i];
		}
		if (edges_[i].vb == entity_index)
		{
			vv[ecnt] = edges_[i].va;
			weights[ecnt++] = weights_[i];
		}
	}
	float *entity_vec = syn0_[entity_index];
	float target_val = 0;
	for (int i = 0; i < ecnt; ++i)
	{
		float dp = MathUtils::DotProduct(entity_vec, syn1_[vv[i]], vec_dim_);
		printf("%d\t%d\t%f\n", vv[i], weights[i], dp);
		target_val += weights[i] * MathUtils::DotProduct(entity_vec, syn1_[vv[i]], vec_dim_);
	}

	const int k = 10;
	int top_indices[k];
	float vals[k];
	std::fill(vals, vals + k, -1e4f);
	float target_sum = 0;
	for (int i = 0; i < num_entities_; ++i)
	{
		float dp = MathUtils::DotProduct(entity_vec, syn1_[i], vec_dim_);
		target_sum += exp(dp);

		int pos = k - 1;
		while (pos > -1 && vals[pos] < dp) --pos;
		for (int j = k - 2; j > pos; --j)
		{
			vals[j + 1] = vals[j];
			top_indices[j + 1] = top_indices[j];
		}
		if (pos != k - 1)
		{
			vals[pos + 1] = dp;
			top_indices[pos + 1] = i;
		}
	}

	printf("%f %f %f\n", target_val, target_sum, target_val / log(target_sum));

	for (int i = 0; i < k; ++i)
	{
		printf("%d\t%f\n", top_indices[i], vals[i]);
	}
}
