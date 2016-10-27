#ifndef PAIRSAMPLER_H_
#define PAIRSAMPLER_H_

#include <random>

#include "multinomialsampler.h"

class PairSampler
{
public:
	PairSampler(const char *adj_list_file_name);

	~PairSampler();

	void SamplePair(int &lidx, int &ridx, std::default_random_engine &generator);
	void SamplePair(int &lidx, int &ridx, std::default_random_engine &generator, RandGen &rand_gen);

	int SampleRight(int lidx, RandGen &rand_gen);

	std::discrete_distribution<int> *neg_sampling_dist()
	{
		return &neg_sampling_dist_;
	}

	int sum_weights()
	{
		return sum_weights_;
	}

	int num_vertex_left()
	{
		return num_vertex_left_;
	}

	int num_vertex_right()
	{
		return num_vertex_right_;
	}

	int CountZeros()
	{
		int cnt = 0;
		const int len = 5;
		int tmpcnts[len];
		for (int i = 0; i < len; ++i)
			tmpcnts[i] = 0;
		for (int i = 0; i < num_vertex_left_; ++i)
		{
			for (int j = 0; j < num_adj_vertices_[i]; ++j)
			{
				if (num_adj_vertices_[i] == len)
					tmpcnts[j] += cnts_[i][j];
				if (cnts_[i][j] == 0)
				{
					++cnt;
					//printf("%d %d\n", j, num_adj_vertices_[i]);
				}
			}
		}
		for (int i = 0; i < len; ++i)
			printf(" %d", tmpcnts[i]);
		printf("\n");
		return cnt;
	}

private:
	std::discrete_distribution<int> left_vertex_dist_;

	std::discrete_distribution<int> *right_vertex_dists_ = 0;

	std::discrete_distribution<int> neg_sampling_dist_;

	//MultinomialSampler left_vertex_sampler_;
	MultinomialSampler *right_vertex_samplers_ = 0;

	int num_vertex_left_ = 0;
	int num_vertex_right_ = 0;
	int **adj_list_;
	int *num_adj_vertices_;
	int sum_weights_ = 0;

	int **cnts_ = 0;
};

#endif
