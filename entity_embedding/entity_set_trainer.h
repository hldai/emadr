#ifndef ENTITY_SET_TRAINER_H_
#define ENTITY_SET_TRAINER_H_

#include <cstdio>
#include <random>

#include "matrix_trainer.h"
#include "negative_sampling_trainer.h"
#include "vector_dict.h"

class EntitySetTrainer
{
	static const short kMaxDocNameLen = 32;
	static const int kFreqTableLen = (int)1e8;

	static const int kNumNegSamples = 10;

public:
	EntitySetTrainer(const char *entity_vec_file_name, float starting_alpha = 0.02f);
	~EntitySetTrainer();

	void Train(const char *doc_entity_list_file_name,
		const char *dst_doc_vec_file_name);

	void TrainM(const char *doc_entity_list_file_name, int dst_dim,
		const char *dst_doc_vec_file_name);

private:
	void trainDocVectorsWithNegativeSampling(const char *dst_file_name);
	void trainDocVectorsThreaded(int *doc_indices, int num_docs, float **vecs1, NegativeSamplingTrainer &ns_trainer, float **dst_vecs,
		int num_threads);
	void trainDocVectorsWithNegativeSamplingMem(int *doc_indices, int doc_beg, int doc_end, float **vecs1,
		NegativeSamplingTrainer &ns_trainer, int seed, float **dst_vecs);

	float *pretrainMatrix(MatrixTrainer &matrix_trainer);
	void trainDocVectorsWithNegativeSamplingM(int vec_dim, float *matrix, const char *dst_file_name,
		MatrixTrainer &matrix_trainer);

	void trainDocVectorWithFixedEntityVecs(int num_entities, int *entities, int *entity_cnts,
		float **entity_vecs, float *tmp_neu1e, NegativeSamplingTrainer &ns_trainer, 
		std::default_random_engine &generator, float *dst_vec);

	void trainMatrix(int *doc_indices, int num_docs, float *matrix, 
		MatrixTrainer &matrix_trainer, float **doc_vecs, float **vecs1, 
		std::default_random_engine &generator);

	void trainDocVectorWithMatrix(int num_entities, int *entities, int *entity_cnts, float *tmp_neu1e,
		float *dst_vec, float *matrix, MatrixTrainer &matrix_trainer, std::default_random_engine &generator);

	void sampleDocs(int num_docs, int *dst_indices, std::default_random_engine &generator);

	double *getEntitySampleWeights();
	void testDocVec(int *entities, int num_entities, float *doc_vec);
	void listPositiveEntityScores(int doc_idx, float *doc_vec, float **entity_vecs);

	void readDocEntities(const char *doc_entity_list_file_name);
	int getNumDocs(FILE *doc_entity_list_fp);
	
	void release();

	unsigned long long nextRandom()
	{
		next_random_ = next_random_ * (unsigned long long)25214903917 + 11;
		return next_random_;
	}

	void saveVectors(float **vecs, int num_vecs, int dim, const char *dst_file_name);

	//void runTrain(int *entities, int num_entities, float *tmp_vec,
	//	int vec_len, float *dst_vec, float alpha, std::default_random_engine &generator,
	//	ExpTable &exp_table);

private:
	VectorDict vec_dict_;

	int num_docs_ = 0;

	char **doc_names_ = 0;
	int **doc_entities_ = 0;
	int **doc_entity_cnts_ = 0;
	int *nums_doc_entities_ = 0;
	int *entity_freqs_;
	std::discrete_distribution<int> entity_sample_dist_;

	// table for generating multinomial distribution random numbers
	unsigned long long next_random_ = 1;

	int dst_dim_ = 0;

	int num_negative_samples_ = kNumNegSamples;
	float starting_alpha_;
};

#endif
