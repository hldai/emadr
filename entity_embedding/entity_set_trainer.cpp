#include "entity_set_trainer.h"

#include <iostream>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <thread>

#include "math_utils.h"
#include "negative_sampling_trainer.h"

using namespace std;

EntitySetTrainer::EntitySetTrainer(const char *entity_vec_file_name, float starting_alpha)
	: vec_dict_(entity_vec_file_name, true), starting_alpha_(starting_alpha)
{
}

EntitySetTrainer::~EntitySetTrainer()
{
	release();
}

void EntitySetTrainer::release()
{
	if (doc_names_ != NULL)
	{
		for (int i = 0; i < num_docs_; ++i)
		{
			delete[] doc_names_[i];
			delete[] doc_entities_[i];
		}
		delete[] doc_names_;
		delete[] doc_entities_;
		delete[] nums_doc_entities_;
		delete[] entity_freqs_;
	}
}

void EntitySetTrainer::Train(const char *doc_entity_list_file_name,
	const char *dst_doc_vec_file_name)
{
	readDocEntities(doc_entity_list_file_name);
	dst_dim_ = vec_dict_.vec_len();
	trainDocVectorsWithNegativeSampling(dst_doc_vec_file_name);
}

void EntitySetTrainer::TrainM(const char *doc_entity_list_file_name, int dst_dim,
	const char *dst_doc_vec_file_name)
{
	readDocEntities(doc_entity_list_file_name);

	dst_dim_ = dst_dim;
	int vec1_dim = vec_dict_.vec_len();
	int num_entity_vecs = vec_dict_.num_vectors();

	printf("dim0: %d dim1: %d num_vectors: %d num_negative_samples: %d\n", dst_dim, vec1_dim,
		num_entity_vecs, kNumNegSamples);

	ExpTable exp_table;
	double *entity_sample_weights = getEntitySampleWeights();
	entity_sample_dist_ = std::discrete_distribution<int>(entity_sample_weights,
		entity_sample_weights + vec_dict_.num_vectors());

	std::default_random_engine generator(13787);

	MatrixTrainer matrix_trainer(&exp_table, dst_dim, vec1_dim, num_entity_vecs, kNumNegSamples,
		&entity_sample_dist_);
	NegativeSamplingTrainer ns_trainer(&exp_table, dst_dim, num_entity_vecs, kNumNegSamples,
		&entity_sample_dist_);

	float **doc_vecs = new float*[num_docs_], **prec_vecs1 = new float*[num_entity_vecs];
	for (int i = 0; i < num_docs_; ++i)
	{
		doc_vecs[i] = new float[dst_dim];
		NegativeSamplingTrainer::InitVec0Def(doc_vecs[i], dst_dim);
	}
	for (int i = 0; i < num_entity_vecs; ++i)
		prec_vecs1[i] = new float[dst_dim];

	const int num_pre_iter_samples = 10000;
	int *doc_samples = new int[num_pre_iter_samples];

	const int num_iter = 5;
	float *matrix = new float[dst_dim * vec1_dim];
	MatrixTrainer::InitMatrix(matrix, dst_dim, vec1_dim);
	for (int i = 0; i < num_iter; ++i)
	{
		printf("matrix norm: %f\n", MathUtils::NormSqr(matrix, dst_dim * vec1_dim));

		printf("pre calc vectors1 ...\n");
		MatrixTrainer::PreCalcVecs1(matrix, dst_dim, vec1_dim, vec_dict_.vecs(), 
			num_entity_vecs, prec_vecs1);

		sampleDocs(num_pre_iter_samples, doc_samples, generator);

		printf("training vectors ...\n");
		trainDocVectorsThreaded(doc_samples, num_pre_iter_samples, vec_dict_.vecs(), 
			ns_trainer, doc_vecs, 4);
		//trainDocVectorsWithNegativeSamplingMem(0, 30, prec_vecs1, ns_trainer, 317, doc_vecs);
		//trainDocVectorsThreaded(prec_vecs1, ns_trainer, doc_vecs, 4);
		//trainDocVectorsThreaded(0, num_docs_, vec_dict_.vecs(), ns_trainer, doc_vecs, 4);

		printf("training matrix ...\n");
		trainMatrix(doc_samples, num_pre_iter_samples, matrix, matrix_trainer, doc_vecs,
			vec_dict_.vecs(), generator);
	}

	printf("pre calc vectors1 ...\n");
	MatrixTrainer::PreCalcVecs1(matrix, dst_dim, vec1_dim, vec_dict_.vecs(),
		num_entity_vecs, prec_vecs1);
	printf("training vectors ...\n");
	trainDocVectorsThreaded(0, num_docs_, prec_vecs1, ns_trainer, doc_vecs, 4);
	//trainDocVectorsWithNegativeSamplingM(dst_dim, matrix, dst_doc_vec_file_name, matrix_trainer);

	saveVectors(doc_vecs, num_docs_, dst_dim, dst_doc_vec_file_name);

	delete[] doc_samples;
	delete[] entity_sample_weights;
	delete[] matrix;
	for (int i = 0; i < num_docs_; ++i)
		delete[] doc_vecs[i];
	delete[] doc_vecs;
	for (int i = 0; i < num_entity_vecs; ++i)
		delete[] prec_vecs1[i];
	delete[] prec_vecs1;
}

void EntitySetTrainer::trainDocVectorsWithNegativeSampling(const char *dst_file_name)
{
	FILE *fp = fopen(dst_file_name, "wb");
	assert(fp != NULL);

	int vec_len = vec_dict_.vec_len();
	ExpTable exp_table;
	double *entity_sample_weights = getEntitySampleWeights();
	entity_sample_dist_ = std::discrete_distribution<int>(entity_sample_weights,
		entity_sample_weights + vec_dict_.num_vectors());
	NegativeSamplingTrainer ns_trainer(&exp_table, vec_len, vec_dict_.num_vectors(), kNumNegSamples,
		&entity_sample_dist_);

	std::default_random_engine generator;

	fwrite(&num_docs_, sizeof(int), 1, fp);
	fwrite(&vec_len, sizeof(int), 1, fp);

	float *no_entity_doc_vec = new float[vec_len];
	//makeRandomVec(no_entity_doc_vec, vec_len);
	NegativeSamplingTrainer::InitVec0Def(no_entity_doc_vec, vec_len);

	float *tmp_neu1e = new float[vec_len];
	float *doc_vec = new float[vec_len];

	for (int i = 0; i < num_docs_; ++i)
	{
		if ((i + 1) % 1000 == 0)
			printf("doc %d\n", (i + 1));

		if (nums_doc_entities_[i] == 0)
		{
			fwrite(no_entity_doc_vec, sizeof(float), vec_len, fp);
		}
		else
		{
			//listPositiveEntityScores(doc_entities_[i], nums_doc_entities_[i], doc_vec);
			//testDocVec(doc_entities_[i], nums_doc_entities_[i], doc_vec);
			NegativeSamplingTrainer::InitVec0Def(doc_vec, vec_len);
			trainDocVectorWithFixedEntityVecs(nums_doc_entities_[i], doc_entities_[i], doc_entity_cnts_[i], vec_dict_.vecs(),
				tmp_neu1e, ns_trainer, generator, doc_vec);
			//testDocVec(doc_entities_[i], nums_doc_entities_[i], doc_vec);
			//listPositiveEntityScores(i, doc_vec, vec_dict_.vecs());
			fwrite(doc_vec, sizeof(float), vec_len, fp);
		}

		//if (i == 50)
		//	break;
	}

	fclose(fp);
	delete[] tmp_neu1e;
	delete[] doc_vec;
	delete[] no_entity_doc_vec;
	delete[] entity_sample_weights;
}

void EntitySetTrainer::trainDocVectorsThreaded(int *doc_indices, int num_docs, float **vecs1, NegativeSamplingTrainer &ns_trainer, float **dst_vecs,
	int num_threads)
{
	int seeds[] = { 3177, 17, 313, 299297 };
	std::thread *threads = new std::thread[num_threads];
	for (int i = 0; i < num_threads; ++i)
	{
		int doc_beg = i * num_docs / num_threads, doc_end = (i + 1) * num_docs / num_threads;
		if (i == num_threads - 1)
			doc_end = num_docs;
		int cur_seed = seeds[i];
		//threads[i] = std::thread([=] { Train(cur_seed); });
		threads[i] = std::thread([=, &ns_trainer]
		{
			trainDocVectorsWithNegativeSamplingMem(doc_indices, doc_beg, doc_end, vecs1,
				ns_trainer, cur_seed, dst_vecs); 
		});
	}
	for (int i = 0; i < num_threads; ++i)
		threads[i].join();
	delete[] threads;
}

void EntitySetTrainer::trainDocVectorsWithNegativeSamplingMem(int *doc_indices, int doc_beg, int doc_end, float **vecs1,
	NegativeSamplingTrainer &ns_trainer, int seed, float **dst_vecs)
{
	printf("thread %d\t%d\t%d\n", doc_beg, doc_end, seed);
	std::default_random_engine generator(seed);

	float *tmp_neu1e = new float[dst_dim_];

	for (int i = doc_beg; i < doc_end; ++i)
	{
		if ((i + 1 - doc_beg) % 3000 == 0)
			printf("doc %d\n", (i + 1 - doc_beg));

		int doc_idx = i;
		if (doc_indices != 0)
			doc_idx = doc_indices[i];

		if (nums_doc_entities_[i] != 0)
		{
			//listPositiveEntityScores(doc_entities_[i], nums_doc_entities_[i], doc_vec);
			//testDocVec(doc_entities_[i], nums_doc_entities_[i], doc_vec);
			trainDocVectorWithFixedEntityVecs(nums_doc_entities_[doc_idx], doc_entities_[doc_idx], doc_entity_cnts_[doc_idx], vecs1,
				tmp_neu1e, ns_trainer, generator, dst_vecs[doc_idx]);
			//testDocVec(doc_entities_[i], nums_doc_entities_[i], doc_vec);

			//if ((i + 1 - doc_beg) < 5)
			//	listPositiveEntityScores(doc_idx, dst_vecs[doc_idx], vecs1);
		}

		//if (i == doc_beg + 10)
		//	break;
	}

	delete[] tmp_neu1e;
}

void EntitySetTrainer::trainDocVectorWithFixedEntityVecs(int num_entities, int *entities, int *entity_cnts,
	float **entity_vecs, float *tmp_neu1e, NegativeSamplingTrainer &ns_trainer,
	std::default_random_engine &generator, float *dst_vec)
{
	int vec_len = vec_dict_.vec_len();
	//makeRandomVec(dst_vec, vec_len);

	//testDocVec(entities, num_entities, dst_vec);

	float alpha = starting_alpha_;  // kDefAlpha;
	if (num_entities < 15)
	{
		float x0 = 1, x1 = 15, y0 = 0.5f, y1 = alpha;
		alpha = (num_entities - 1) / (x1 - x0) * (y1 - y0) + y0;
		//alpha = 0.13f;
	}


	for (int i = 0; i < 100; ++i)
	{
		for (int j = 0; j < num_entities; ++j)
		{
			for (int k = 0; k < entity_cnts[j]; ++k)
				ns_trainer.TrainPrediction(dst_vec, entities[j], entity_vecs,
					alpha, tmp_neu1e, generator, true, false);
		}
		alpha *= 0.97f;
		if (alpha < starting_alpha_ * 0.1f)
			alpha = starting_alpha_ * 0.1f;
	}
}

void EntitySetTrainer::trainMatrix(int *doc_indices, int num_docs, float *matrix, 
	MatrixTrainer &matrix_trainer, float **doc_vecs, float **vecs1, std::default_random_engine &generator)
{
	float alpha = 0.01f;

	for (int i = 0; i < num_docs; ++i)
	{
		if ((i + 1) % 2500 == 0)
			printf("%d\n", (i + 1));

		int doc_idx = doc_indices[i];
		if (nums_doc_entities_[doc_idx] == 0)
			continue;
		for (int j = 0; j < nums_doc_entities_[doc_idx]; ++j)
		{
			for (int k = 0; k < doc_entity_cnts_[doc_idx][j]; ++k)
				matrix_trainer.TrainMatrix(doc_vecs[doc_idx], doc_entities_[doc_idx][j], vecs1,
					matrix, alpha, generator);
		}
	}
}

float *EntitySetTrainer::pretrainMatrix(MatrixTrainer &matrix_trainer)
{
	int vec1_dim = vec_dict_.vec_len();
	float *matrix = new float[dst_dim_ * vec1_dim];
	std::fill(matrix, matrix + dst_dim_ * vec1_dim, 0.0f);

	float *tmp_neu1e = new float[dst_dim_];
	float *doc_vec = new float[dst_dim_];

	return matrix;
}

void EntitySetTrainer::trainDocVectorsWithNegativeSamplingM(int vec_dim, float *matrix, const char *dst_file_name,
	MatrixTrainer &matrix_trainer)
{
	FILE *fp = fopen(dst_file_name, "wb");
	assert(fp != NULL);

	int vec1_dim = vec_dict_.vec_len();
	//ExpTable exp_table;
	//double *entity_sample_weights = getEntitySampleWeights();
	//entity_sample_dist_ = std::discrete_distribution<int>(entity_sample_weights,
	//	entity_sample_weights + vec_dict_.num_vectors());
	//MatrixTrainer matrix_trainer(&exp_table, vec_dim, vec1_dim, vec_dict_.num_vectors(), kNumNegSamples,
	//	&entity_sample_dist_);

	std::default_random_engine generator;

	fwrite(&num_docs_, sizeof(int), 1, fp);
	fwrite(&vec_dim, sizeof(int), 1, fp);

	float *no_entity_doc_vec = new float[vec_dim];
	//makeRandomVec(no_entity_doc_vec, vec_len);
	NegativeSamplingTrainer::InitVec0Def(no_entity_doc_vec, vec_dim);

	//float *matrix = new float[vec_dim * vec1_dim];
	//std::fill(matrix, matrix + vec_dim * vec1_dim, 0.0f);
	//for (int i = 0; i < vec_dim * vec1_dim; ++i)
	//	matrix[i] = rand() / RAND_MAX - 0.5f;

	float *tmp_neu1e = new float[vec_dim];
	float *doc_vec = new float[vec_dim];

	for (int i = 0; i < num_docs_; ++i)
	{
		//if (i < 12)
		//	continue;
		if ((i + 1) % 1000 == 0)
			printf("doc %d\n", (i + 1));

		if (nums_doc_entities_[i] == 0)
		{
			fwrite(no_entity_doc_vec, sizeof(float), vec_dim, fp);
		}
		else
		{
			//listPositiveEntityScores(doc_entities_[i], nums_doc_entities_[i], doc_vec);
			//testDocVec(doc_entities_[i], nums_doc_entities_[i], doc_vec);
			trainDocVectorWithMatrix(nums_doc_entities_[i], doc_entities_[i], doc_entity_cnts_[i], tmp_neu1e,
				doc_vec, matrix, matrix_trainer, generator);
			//printf("%f\n", MathUtils::NormSqr(matrix, vec_dim * vec1_dim));
			printf("%d\n", i);
			matrix_trainer.ListScores(nums_doc_entities_[i], doc_entities_[i], doc_vec,
				vec_dict_.vecs(), matrix);
			//testDocVec(doc_entities_[i], nums_doc_entities_[i], doc_vec);
			//listPositiveEntityScores(doc_entities_[i], nums_doc_entities_[i], doc_vec);
			fwrite(doc_vec, sizeof(float), vec_dim, fp);
		}

		if (i == 5)
			break;
	}

	fclose(fp);
	delete[] tmp_neu1e;
	delete[] doc_vec;
	delete[] no_entity_doc_vec;
}

void EntitySetTrainer::trainDocVectorWithMatrix(int num_entities, int *entities, int *entity_cnts, float *tmp_neu1e,
	float *dst_vec, float *matrix, MatrixTrainer &matrix_trainer, std::default_random_engine &generator)
{
	int vec_len = matrix_trainer.vec0_dim();
	NegativeSamplingTrainer::InitVec0Def(dst_vec, vec_len);
	//makeRandomVec(dst_vec, vec_len);

	//for (int i = 0; i < vec_len; ++i)
	//	printf("%f ", dst_vec[i]);
	//printf("\n");

	//testDocVec(entities, num_entities, dst_vec);

	float alpha = starting_alpha_;  // kDefAlpha;
	if (num_entities < 15)
	{
		float x0 = 1, x1 = 15, y0 = 0.5f, y1 = alpha;
		alpha = (num_entities - 1) / (x1 - x0) * (y1 - y0) + y0;
		//alpha = 0.13f;
	}

	for (int i = 0; i < 70; ++i)
	{
		for (int j = 0; j < num_entities; ++j)
		{
			for (int k = 0; k < entity_cnts[j]; ++k)
				matrix_trainer.TrainPrediction(dst_vec, entities[j], vec_dict_.vecs(), matrix,
					alpha, tmp_neu1e, generator, true, false, false);
		}
		alpha *= 0.97f;
		if (alpha < starting_alpha_ * 0.1f)
			alpha = starting_alpha_ * 0.1f;
	}

	//for (int i = 0; i < vec_len; ++i)
	//	printf("%f ", dst_vec[i]);
	//printf("\n");
}

void EntitySetTrainer::readDocEntities(const char *doc_entity_list_file_name)
{
	FILE *fp = fopen(doc_entity_list_file_name, "r");
	assert(fp != 0);

	num_docs_ = getNumDocs(fp);
	printf("%d documents. reading entity indices.\n", num_docs_);

	release();

	doc_names_ = new char*[num_docs_];
	doc_entities_ = new int*[num_docs_];
	doc_entity_cnts_ = new int*[num_docs_];
	nums_doc_entities_ = new int[num_docs_];
	entity_freqs_ = new int[vec_dict_.num_vectors()];
	std::fill(entity_freqs_, entity_freqs_ + vec_dict_.num_vectors(), 0);

	for (int i = 0; i < num_docs_; ++i)
	{
		doc_names_[i] = new char[kMaxDocNameLen];
		fscanf(fp, "%s", doc_names_[i]);

		fscanf(fp, "%d", &nums_doc_entities_[i]);
		//printf("%d\n", num_entities);
		doc_entities_[i] = new int[nums_doc_entities_[i]];
		doc_entity_cnts_[i] = new int[nums_doc_entities_[i]];
		for (int j = 0; j < nums_doc_entities_[i]; ++j)
		{
			fscanf(fp, "%d %d", &doc_entities_[i][j], &doc_entity_cnts_[i][j]);
			--doc_entities_[i][j];
			++entity_freqs_[doc_entities_[i][j]];
		}
	}

	fclose(fp);
	printf("done.\n");
}

int EntitySetTrainer::getNumDocs(FILE *doc_entity_list_fp)
{
	const int BUF_LEN = 1024 * 32;
	char buf[BUF_LEN];
	size_t len = 0;
	int cnt = 0;
	while ((len = fread(buf, 1, BUF_LEN, doc_entity_list_fp)) > 0)
	{
		//printf("%d\n", len);
		for (int i = 0; i < len; ++i)
		{
			if (buf[i] == '\n')
			{
				++cnt;
			}
		}
	}

	fseek(doc_entity_list_fp, SEEK_SET, 0);

	return cnt;
}

void EntitySetTrainer::sampleDocs(int num_docs, int *dst_indices, std::default_random_engine &generator)
{
	std::uniform_int_distribution<int> distribution(0, num_docs_);
	for (int i = 0; i < num_docs; ++i)
	{
		dst_indices[i] = distribution(generator);
	}
}

double *EntitySetTrainer::getEntitySampleWeights()
{
	const double power = 0.75;
	const int &num_entities = vec_dict_.num_vectors();
	double *weights = new double[num_entities];
	for (int i = 0; i < num_entities; ++i)
		weights[i] = pow(entity_freqs_[i], power);
	return weights;
}

void EntitySetTrainer::testDocVec(int *entities, int num_entities, float *doc_vec)
{
	int vec_len = vec_dict_.vec_len();
	float obj_val = 0;
	for (int i = 0; i < num_entities; ++i)
	{
		float *vec = vec_dict_.GetVector(entities[i]);
		float dp = MathUtils::DotProduct(vec, doc_vec, vec_len);
		obj_val += MathUtils::Sigma(dp) - MathUtils::Sigma(-dp);
		//float dp = dotProduct(vec, doc_vec, vec_len);
		//printf("dp: %f\n", dp);
	}

	for (int i = 0; i < vec_dict_.num_vectors(); ++i)
	{
		float *vec = vec_dict_.GetVector(i);
		float dp = MathUtils::DotProduct(vec, doc_vec, vec_len);
		obj_val += MathUtils::Sigma(-dp);
	}

	printf("%f\n", obj_val);
}

void EntitySetTrainer::listPositiveEntityScores(int doc_idx, float *doc_vec, float **entity_vecs)
{
	int *entities = doc_entities_[doc_idx];
	int num_entities = nums_doc_entities_[doc_idx];

	float obj_val = 0;
	int cnt = 0;
	for (int i = 0; i < num_entities; ++i)
	{
		float *vec = entity_vecs[entities[i]];
		float dp = MathUtils::DotProduct(vec, doc_vec, dst_dim_);
		if (dp > 0)
			++cnt;
	}
	printf("%d positive: %d %d %f\n", doc_idx, cnt, num_entities, (float)cnt / num_entities);

	//printf("\n");
	const int num_test_neg_samples = 100;
	cnt = 0;
	for (int i = 0; i < num_test_neg_samples; ++i)
	{
		float *vec = entity_vecs[nextRandom() % vec_dict_.num_vectors()];
		float dp = MathUtils::DotProduct(vec, doc_vec, dst_dim_);
		//printf("dp: %f\n", dp);
		if (dp < 0)
			++cnt;
	}
	printf("%d negative: %d %d %f\n", doc_idx, cnt, num_test_neg_samples,
		(float)cnt / num_test_neg_samples);
}

void EntitySetTrainer::saveVectors(float **vecs, int num_vecs, int dim, const char *dst_file_name)
{
	FILE *fp = fopen(dst_file_name, "wb");
	assert(fp != 0);

	fwrite(&num_vecs, 4, 1, fp);
	fwrite(&dim, 4, 1, fp);

	for (int i = 0; i < num_vecs; ++i)
		fwrite(vecs[i], 4, dim, fp);

	fclose(fp);
}
