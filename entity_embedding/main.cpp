#include <cstdio>
#include <ctime>
#include <random>
#include <array>
#include <cassert>
#include <thread>
#include <iostream>

//#include "entity_set_trainer.h"
#include "io_utils.h"
//#include "joint_trainer.h"
#include "math_utils.h"
#include "net_edge_sampler.h"
#include "eadocvectrainer.h"

void TrainDocWordVectors()
{
	int vec_dim = 50;
	int num_threads = 4;
	int num_rounds = 20;
	int num_negative_samples = 10;
	float starting_alpha = 0.06f;

	//const char *doc_words_file_name = "e:/dc/20ng_bydate/all_docs_dw_net_short.bin";
	//const char *word_cnts_file = "e:/dc/20ng_bydate/word_cnts.bin";
	//const char *word_vecs_file_name = "e:/dc/20ng_bydate/vecs/word_vecs.bin";
	//const char *dst_vec_file_name = "e:/dc/20ng_bydate/vecs/doc_vec_cpp_ea.bin";

	//const char *doc_words_file_name = "e:/dc/el/tac/tac_2010_train_docs_bow.bin";
	//const char *word_cnts_file = "e:/dc/el/wiki/word_cnts.bin";
	//const char *word_vecs_file_name = "e:/dc/el/vecs/word_vecs.bin";
	//const char *dst_vec_file_name = "e:/dc/el/vecs/tac_2010_train_dw_vecs.bin";

	//const char *doc_words_file_name = "e:/dc/el/tac/tac_2009_eval_docs_bow.bin";
	//const char *word_cnts_file = "e:/dc/el/wiki/word_cnts.bin";
	//const char *word_vecs_file_name = "e:/dc/el/vecs/word_vecs.bin";
	//const char *dst_vec_file_name = "e:/dc/el/vecs/tac_2009_eval_dw_vecs.bin";

	//const char *doc_words_file_name = "e:/dc/el/tac/tac_2010_train_entities.bin";
	//const char *word_cnts_file = "e:/dc/el/wiki/entity_cnts.bin";
	//const char *word_vecs_file_name = "e:/dc/el/vecs/entity_vecs.bin";
	//const char *dst_vec_file_name = "e:/dc/el/vecs/tac_2010_train_entity_vecs.bin";

	const char *doc_words_file_name = "e:/dc/el/tac/tac_2009_eval_entities.bin";
	const char *word_cnts_file = "e:/dc/el/wiki/entity_cnts.bin";
	const char *word_vecs_file_name = "e:/dc/el/vecs/entity_vecs.bin";
	const char *dst_vec_file_name = "e:/dc/el/vecs/tac_2009_eval_entity_vecs.bin";

	EADocVecTrainer trainer(num_rounds, num_threads, num_negative_samples, starting_alpha);
	trainer.TrainDocWordFixedWordVecs(doc_words_file_name, word_cnts_file, word_vecs_file_name, 
		vec_dim, dst_vec_file_name);
}

void JointTrainingNYT()
{
	const char *entity_net_file_name = "e:/dc/nyt/sentence_based_weighted_edge_list.txt";
	const char *doc_entity_net_file_name = "e:/dc/nyt/doc_entities_lo_f2012.bin";
	const char *doc_words_file_name = "e:/dc/nyt/line_docs/doc_word_indices_rm_ssw_lo_2012.bin";
	//const char *dst_doc_vec_file_name = "e:/dc/nyt/vecs/doc_vec_lo_f2012_joint_100.bin";
	const int entity_vec_dim = 64;
	const int word_vec_dim = 64;
	const int doc_vec_dim = 128;
	const int num_rounds = 5;
	const int num_threads = 4;
	const int num_negative_samples = 5;
	char dst_doc_vec_file_name[256];
	sprintf(dst_doc_vec_file_name, "e:/dc/nyt/vecs/doc_vec_lo_f2012_joint_%d.bin", doc_vec_dim);
	//JointTrainer jt(entity_net_file_name, doc_entity_net_file_name, doc_words_file_name);
	//jt.JointTrainingThreaded(entity_vec_dim, word_vec_dim, doc_vec_dim, num_rounds, num_threads, 
	//	num_negative_samples, dst_doc_vec_file_name);
}

void JointTraining20NG()
{
	const char *entity_net_file_name = "e:/dc/20ng_bydate/weighted_entity_edge_list.txt";
	const char *doc_entity_net_file_name = "e:/dc/20ng_bydate/doc_entities.bin";
	const char *doc_words_file_name = "e:/dc/20ng_bydate/all_docs_dw_net.bin";

	//const char *entity_net_file_name = "e:/dc/20ng_data/split/train_weighted_entity_edge_list.txt";
	//const char *doc_entity_net_file_name = "e:/dc/20ng_data/split/train_doc_entities.bin";
	//const char *doc_words_file_name = "e:/dc/20ng_data/split/train_docs_dw_net.bin";

	//const char *entity_net_file_name = "e:/dc/20ng_data/split/train_weighted_entity_edge_list.txt";
	//const char *doc_entity_net_file_name = "e:/dc/20ng_data/split/test_doc_entities.bin";
	//const char *doc_words_file_name = "e:/dc/20ng_data/split/test_docs_dw_net.bin";

	const int entity_vec_dim = 100;
	const int word_vec_dim = 100;
	const int doc_vec_dim = 200;
	const int num_rounds = 5;
	const int num_threads = 4;
	const int num_negative_samples = 5;
	char dst_doc_vec_file_name[256];
	sprintf(dst_doc_vec_file_name, "e:/dc/20ng_bydate/vecs/all_doc_vec_joint_%d.bin", doc_vec_dim);

	//JointTrainer jt(entity_net_file_name, doc_entity_net_file_name, doc_words_file_name);
	//jt.JointTrainingThreaded(entity_vec_dim, word_vec_dim, doc_vec_dim, num_rounds, num_threads,
	//	num_negative_samples, dst_doc_vec_file_name);
}

void EATrainDW(int argc, char **argv)
{
	//const char *entity_net_file_name = "e:/dc/20ng_bydate/entity_net_adj_list.bin";
	//const char *doc_entity_net_file_name = "e:/dc/20ng_bydate/doc_entities_short.bin";
	const char *doc_words_file_name = "e:/dc/20ng_bydate/all_docs_dw_net_short.bin";
	const char *entity_cnts_file = "e:/dc/20ng_bydate/entity_cnts.bin";
	const char *word_cnts_file = "e:/dc/20ng_bydate/word_cnts.bin";
	const char *dst_doc_vecs_file_name = "e:/dc/20ng_bydate/vecs/dw_vecs.bin";
	const char *dst_word_vec_file_name = "e:/dc/20ng_bydate/vecs/dw_word_vecs.bin";

	//const char *entity_net_file_name = "/home/dhl/data/dc/20ng_bydate/weighted_entity_edge_list.txt";
	//const char *doc_entity_net_file_name = "/home/dhl/data/dc/20ng_bydate/doc_entities.bin";
	//const char *doc_words_file_name = "/home/dhl/data/dc/20ng_bydate/all_docs_dw_net.bin";
	//const char *dst_dedw_vecs_file_name = "/home/dhl/data/dc/20ng_bydate/dedw_vecs.bin";

	int doc_vec_dim = 400;
	int num_rounds = 40;
	int num_threads = 4;
	int num_negative_samples = 10;
	float starting_alpha = 0.06f;
	float min_alpha = 0.0001f;

	printf("vec_dim: %d\nnum_rounds: %d\nnum_threads: %d\nnum_neg_samples: %d\nstarting_alpha: %f\nmin_alpha: %f\n",
		doc_vec_dim, num_rounds, num_threads, num_negative_samples, starting_alpha, min_alpha);

	EADocVecTrainer eatrain(num_rounds, num_threads, num_negative_samples, starting_alpha, min_alpha);
	eatrain.TrainDocWord(doc_words_file_name, word_cnts_file, doc_vec_dim, dst_doc_vecs_file_name);

	//JointTrainer jt(entity_net_file_name, doc_entity_net_file_name, doc_words_file_name);
	//jt.TrainCMThreaded(doc_vec_dim, num_rounds, num_threads, num_negative_samples,
	//	starting_alpha, min_alpha, dst_doc_vecs_file_name);
}

void EATrain(int argc, char **argv)
{
	const char *entity_net_file_name, *doc_entity_net_file_name, *doc_words_file_name, 
		*entity_cnts_file, *word_cnts_file, *dst_doc_vecs_file_name, *dst_word_vec_file_name, 
		*dst_entityvec_file_name;

	bool is_windows = 1;
	bool is_wiki = 0;
	bool share_doc_vec = 0;

	if (is_windows)
	{
		entity_net_file_name = "e:/dc/20ng_bydate/entity_net_adj_list.bin";
		doc_entity_net_file_name = "e:/dc/20ng_bydate/doc_entities_short.bin";
		doc_words_file_name = "e:/dc/20ng_bydate/all_docs_dw_net_short.bin";
		entity_cnts_file = "e:/dc/20ng_bydate/entity_cnts.bin";
		word_cnts_file = "e:/dc/20ng_bydate/word_cnts.bin";
		dst_doc_vecs_file_name = "e:/dc/20ng_bydate/vecs/dedw_vecs.bin";
		dst_word_vec_file_name = "e:/dc/20ng_bydate/vecs/word_vecs.bin";
		dst_entityvec_file_name = "e:/dc/20ng_bydate/vecs/entity_vecs.bin";
	}
	else
	{
		if (is_wiki)
		{
			entity_net_file_name = "/home/dhl/data/dc/el/entity_net_adj_list.bin";
			doc_entity_net_file_name = "/home/dhl/data/dc/el/doc_entities.bin";
			doc_words_file_name = "/home/dhl/data/dc/el/wiki_bow.bin";
			entity_cnts_file = "/home/dhl/data/dc/el/entity_cnts.bin";
			word_cnts_file = "/home/dhl/data/dc/el/word_cnts.bin";
			dst_doc_vecs_file_name = "/home/dhl/data/dc/el/vecs/wiki_vecs.bin";
			dst_word_vec_file_name = "/home/dhl/data/dc/el/vecs/word_vecs.bin";
			dst_entityvec_file_name = "/home/dhl/data/dc/el/vecs/entity_vecs.bin";
		}
		else
		{
			entity_net_file_name = "/home/dhl/data/dc/20ng_bydate/entity_net_adj_list.bin";
			doc_entity_net_file_name = "/home/dhl/data/dc/20ng_bydate/doc_entities_short.bin";
			doc_words_file_name = "/home/dhl/data/dc/20ng_bydate/all_docs_dw_net_short.bin";
			entity_cnts_file = "/home/dhl/data/dc/20ng_bydate/entity_cnts.bin";
			word_cnts_file = "/home/dhl/data/dc/20ng_bydate/word_cnts.bin";
			dst_doc_vecs_file_name = "/home/dhl/data/dc/20ng_bydate/vecs/dedw_vecs.bin";
			dst_word_vec_file_name = "/home/dhl/data/dc/20ng_bydate/vecs/word_vecs.bin";
			dst_entityvec_file_name = "/home/dhl/data/dc/20ng_bydate/vecs/entity_vecs.bin";
		}
	}

	int doc_vec_dim = 50;
	int num_rounds = 20;
	int num_threads = 4;
	int num_negative_samples = 10;
	float starting_alpha = 0.06f;
	float min_alpha = 0.0001f;

	if (argc >= 7)
	{
		doc_vec_dim = atoi(argv[1]);
		num_rounds = atoi(argv[2]);
		num_threads = atoi(argv[3]);
		num_negative_samples = atoi(argv[4]);
		starting_alpha = (float)atof(argv[5]);
		min_alpha = (float)atof(argv[6]);
		if (argc == 8)
			dst_doc_vecs_file_name = argv[7];
	}

	printf("vec_dim: %d\nnum_rounds: %d\nnum_threads: %d\nnum_neg_samples: %d\nstarting_alpha: %f\nmin_alpha: %f\n",
		doc_vec_dim, num_rounds, num_threads, num_negative_samples, starting_alpha, min_alpha);

	EADocVecTrainer eatrain(num_rounds, num_threads, num_negative_samples, starting_alpha, min_alpha);
	eatrain.AllJointThreaded(entity_net_file_name, doc_entity_net_file_name, doc_words_file_name,
		entity_cnts_file, word_cnts_file, doc_vec_dim, share_doc_vec, dst_doc_vecs_file_name, dst_word_vec_file_name,
		dst_entityvec_file_name);
	//eatrain.TrainEnergyMT(entity_net_file_name, doc_entity_net_file_name, doc_words_file_name,
	//	entity_cnts_file, word_cnts_file, doc_vec_dim, dst_doc_vecs_file_name, dst_word_vec_file_name,
	//	dst_entityvec_file_name);
}

void Test()
{
	std::default_random_engine generator(43);
	MultinomialSampler sampler;
	const int len = 3;
	int weights[] = { 3, 1, 2 };
	int cnts[len];
	for (int i = 0; i < len; ++i)
		cnts[i] = 0;
	sampler.Init(weights, 3);
	for (int i = 0; i < 10000; ++i)
		++cnts[sampler.Sample(generator)];
	for (int i = 0; i < len; ++i)
		printf("%d %d\n", i, cnts[i]);
	//delete[] weights;
}

int main(int argc, char **argv)
{
	time_t t = time(0);

	//Test();
	//JointTrainingNYT();
	//JointTraining20NG();
	//JointTrainingOML20NG(argc, argv);
	//JointTrainingCM20NG(argc, argv);

	//TrainDocWordVectors();
	//EATrainDW(argc, argv);
	EATrain(argc, argv);

	time_t et = time(0) - t;
	printf("\n%lld s. %lld m. %lld h.\n", et, et / 60, et / 3600);

	return 0;
}
