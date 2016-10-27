#include <cstdio>
#include <ctime>
#include <random>
#include <array>
#include <cassert>
#include <thread>
#include <iostream>
#include <map>

#include "ioutils.h"
#include "mathutils.h"
#include "pairsampler.h"
#include "eadocvectrainer.h"

enum DataSet {
	WIN_NYT,
	WIN_20NG,
	LINUX_WIKI,
	LINUX_20NG
};

void TrainDocWordVectors()
{
	int vec_dim = 50;
	int num_threads = 4;
	int num_rounds = 20;
	int num_negative_samples = 10;
	float starting_alpha = 0.06f;

	const char *doc_words_file_name = "e:/dc/el/tac/2010/train/dw.bin";
	const char *word_cnts_file = "e:/dc/el/wiki/word_cnts.bin";
	const char *word_vecs_file_name = "e:/dc/el/vecs/word_vecs_3.bin";
	const char *dst_vec_file_name = "e:/dc/el/vecs/2010/train_3_dw_vecs.bin";

	EADocVecTrainer trainer(num_rounds, num_threads, num_negative_samples, starting_alpha);
	trainer.TrainDocWordFixedWordVecs(doc_words_file_name, word_cnts_file, word_vecs_file_name, 
		vec_dim, dst_vec_file_name);
}

void EATrainDWEFixed()
{

	const char *de_file_name = "e:/dc/el/tac/2010/train/de.bin";
	const char *doc_words_file_name = "e:/dc/el/tac/2010/train/dw.bin";
	const char *dst_doc_vecs_file_name = "e:/dc/el/vecs/tac_2010_train_vecs_3.bin";

	const char *word_cnts_file = "e:/dc/el/wiki/word_cnts.bin";
	const char *entity_cnts_file = "e:/dc/el/wiki/entity_cnts.bin";
	const char *word_vecs_file_name = "e:/dc/el/vecs/word_vecs_3.bin";
	const char *entity_vecs_file_name = "e:/dc/el/vecs/entity_vecs_3.bin";

	int doc_vec_dim = 50;
	int num_rounds = 20;
	int num_threads = 4;
	int num_negative_samples = 10;
	float starting_alpha = 0.06f;
	float min_alpha = 0.0001f;

	printf("vec_dim: %d\nnum_rounds: %d\nnum_threads: %d\nnum_neg_samples: %d\nstarting_alpha: %f\nmin_alpha: %f\n",
		doc_vec_dim, num_rounds, num_threads, num_negative_samples, starting_alpha, min_alpha);

	EADocVecTrainer eatrain(num_rounds, num_threads, num_negative_samples, starting_alpha, min_alpha);
	eatrain.TrainWEFixed(doc_words_file_name, de_file_name, word_cnts_file, entity_cnts_file,
		word_vecs_file_name, entity_vecs_file_name, doc_vec_dim, dst_doc_vecs_file_name);
}

void configFilesDW(DataSet data_set, const char *&dw_file, const char *&word_cnts_file,
	const char *&dst_doc_vecs_file, const char *&dst_word_vecs_file)
{
	switch (data_set)
	{
	case WIN_20NG:
		dw_file = "e:/dc/20ng_bydate/bin/all_docs_dw_short.bin";
		word_cnts_file = "e:/dc/20ng_bydate/bin/word_cnts.bin";
		dst_doc_vecs_file = "e:/dc/20ng_bydate/vecs/dw-vecs.bin";
		dst_word_vecs_file = "e:/dc/20ng_bydate/vecs/dw-word-vecs.bin";
		break;
	case WIN_NYT:
		dw_file = "e:/dc/nyt-world-full/processed/bin/dw.bin";
		word_cnts_file = "e:/dc/nyt-world-full/processed/bin/word-cnts.bin";
		dst_doc_vecs_file = "e:/dc/nyt-world-full/processed/vecs/dw-vecs.bin";
		dst_word_vecs_file = "e:/dc/nyt-world-full/processed/vecs/dw-word-vecs.bin";
		break;
	case LINUX_20NG:
		dw_file = "/home/dhl/data/dc/20ng_bydate/all_docs_dw_short.bin";
		word_cnts_file = "/home/dhl/data/dc/20ng_bydate/word_cnts.bin";
		dst_doc_vecs_file = "/home/dhl/data/dc/20ng_bydate/vecs/dedw_vecs_tmp.bin";
		dst_word_vecs_file = "/home/dhl/data/dc/20ng_bydate/vecs/word_vecs_tmp.bin";
		break;
	case LINUX_WIKI:
		dw_file = "/home/dhl/data/dc/el/wiki_bow.bin";
		word_cnts_file = "/home/dhl/data/dc/el/word_cnts.bin";
		dst_doc_vecs_file = "/home/dhl/data/dc/el/vecs/wiki_vecs.bin";
		dst_word_vecs_file = "/home/dhl/data/dc/el/vecs/word_vecs.bin";
		break;
	default:
		break;
	}
}

void EATrainDW(int argc, char **argv)
{
	//const char *doc_words_file_name = "/home/dhl/data/dc/20ng_bydate/all_docs_dw.bin";
	//const char *dst_dedw_vecs_file_name = "/home/dhl/data/dc/20ng_bydate/dedw_vecs.bin";

	const char *dw_file, *word_cnts_file, 
		*dst_doc_vecs_file, *dst_word_vecs_file;

	DataSet data_set = DataSet::WIN_NYT;
	configFilesDW(data_set, dw_file, word_cnts_file, dst_doc_vecs_file, dst_word_vecs_file);

	int doc_vec_dim = 100;
	int num_rounds = 10;
	int num_threads = 4;
	int num_negative_samples = 10;
	float starting_alpha = 0.06f;
	float min_alpha = 0.0001f;

	printf("vec_dim: %d\nnum_rounds: %d\nnum_threads: %d\nnum_neg_samples: %d\nstarting_alpha: %f\nmin_alpha: %f\n",
		doc_vec_dim, num_rounds, num_threads, num_negative_samples, starting_alpha, min_alpha);

	EADocVecTrainer eatrain(num_rounds, num_threads, num_negative_samples, starting_alpha, min_alpha);
	eatrain.TrainDocWord(dw_file, word_cnts_file, doc_vec_dim, dst_doc_vecs_file);
}

void configFiles(DataSet data_set, const char *&ee_file, const char *&de_file, const char *&dw_file,
	const char *&entity_cnts_file, const char *&word_cnts_file, const char *&dst_doc_vecs_file,
	const char *&dst_word_vecs_file, const char *&dst_entity_vecs_file)
{
	switch (data_set)
	{
	case WIN_20NG:
		//ee_file = "e:/dc/20ng_bydate/bin/entity_adj_list.bin";
		//de_file = "e:/dc/20ng_bydate/bin/doc_entities_short.bin";
		//entity_cnts_file = "e:/dc/20ng_bydate/bin/entity_cnts.bin";
		ee_file = "e:/data/emadr/20ng_bydate/bin/ee-ner.bin";
		de_file = "e:/data/emadr/20ng_bydate/bin/de-ner.bin";
		dw_file = "e:/data/emadr/20ng_bydate/bin/dw.bin";
		entity_cnts_file = "e:/data/emadr/20ng_bydate/bin/entity-cnts-ner.bin";
		word_cnts_file = "e:/data/emadr/20ng_bydate/bin/word-cnts.bin";
		dst_doc_vecs_file = "e:/data/emadr/20ng_bydate/vecs/dew-vecs.bin";
		dst_word_vecs_file = "e:/data/emadr/20ng_bydate/vecs/word-vecs-ner.bin";
		dst_entity_vecs_file = "e:/data/emadr/20ng_bydate/vecs/entity-vecs-ner.bin";
		break;
	case WIN_NYT:
		dw_file = "e:/dc/nyt-world-full/processed/bin/dw.bin";
		ee_file = "e:/dc/nyt-world-full/processed/bin/ee-ner.bin";
		de_file = "e:/dc/nyt-world-full/processed/bin/de-ner.bin";
		entity_cnts_file = "e:/dc/nyt-world-full/processed/bin/entity-cnts-ner.bin";
		word_cnts_file = "e:/dc/nyt-world-full/processed/bin/word-cnts.bin";
		dst_doc_vecs_file = "e:/dc/nyt-world-full/processed/vecs/dedw4-vecs-012.bin";
		dst_word_vecs_file = "e:/dc/nyt-world-full/processed/vecs/word4-vecs.bin";
		dst_entity_vecs_file = "e:/dc/nyt-world-full/processed/vecs/entity4-vecs.bin";
		break;
	case LINUX_20NG:
		ee_file = "/home/dhl/data/dc/20ng_bydate/entity_adj_list.bin";
		de_file = "/home/dhl/data/dc/20ng_bydate/doc_entities_short.bin";
		dw_file = "/home/dhl/data/dc/20ng_bydate/all_docs_dw_short.bin";
		entity_cnts_file = "/home/dhl/data/dc/20ng_bydate/entity_cnts.bin";
		word_cnts_file = "/home/dhl/data/dc/20ng_bydate/word_cnts.bin";
		dst_doc_vecs_file = "/home/dhl/data/dc/20ng_bydate/vecs/dedw_vecs_tmp.bin";
		dst_word_vecs_file = "/home/dhl/data/dc/20ng_bydate/vecs/word_vecs_tmp.bin";
		dst_entity_vecs_file = "/home/dhl/data/dc/20ng_bydate/vecs/entity_vecs_tmp.bin";
		break;
	case LINUX_WIKI:
		ee_file = "/home/dhl/data/dc/el/entity_adj_list.bin";
		de_file = "/home/dhl/data/dc/el/doc_entities.bin";
		dw_file = "/home/dhl/data/dc/el/wiki_bow.bin";
		entity_cnts_file = "/home/dhl/data/dc/el/entity_cnts.bin";
		word_cnts_file = "/home/dhl/data/dc/el/word_cnts.bin";
		dst_doc_vecs_file = "/home/dhl/data/dc/el/vecs/wiki_vecs.bin";
		dst_word_vecs_file = "/home/dhl/data/dc/el/vecs/word_vecs.bin";
		dst_entity_vecs_file = "/home/dhl/data/dc/el/vecs/entity_vecs.bin";
		break;
	default:
		break;
	}
}

char *GetArgValue(int argc, char **argv, const char *arg)
{
	for (int i = 0; i < argc - 1; ++i)
	{
		if (strcmp(argv[i], arg) == 0)
			return argv[i + 1];
	}
	return 0;
}

int GetIntArgValue(int argc, char **argv, const char *arg, int def_val)
{
	char *arg_val = GetArgValue(argc, argv, arg);
	if (arg_val)
		return atoi(arg_val);
	return def_val;
}

float GetFloatArgValue(int argc, char **argv, const char *arg, float def_val)
{
	char *arg_val = GetArgValue(argc, argv, arg);
	if (arg_val)
		return atof(arg_val);
	return def_val;
}

void EATrain(int argc, char **argv)
{
	const char *ee_file, *de_file, *dw_file, *entity_cnts_file, *word_cnts_file, 
		*dst_doc_vecs_file, *dst_word_vecs_file,
		*dst_entity_vecs_file;

	bool share_doc_vec = 1;

	int doc_vec_dim = GetIntArgValue(argc, argv, "-d", 100);
	int num_rounds = GetIntArgValue(argc, argv, "-r", 10);
	int num_threads = GetIntArgValue(argc, argv, "-t", 2);
	int num_negative_samples = GetIntArgValue(argc, argv, "-n", 10);
	float starting_alpha = GetFloatArgValue(argc, argv, "-sa", 0.06f);
	float min_alpha = GetFloatArgValue(argc, argv, "-ma", 0.0001f);

	ee_file = GetArgValue(argc, argv, "-ee");
	de_file = GetArgValue(argc, argv, "-de");
	dw_file = GetArgValue(argc, argv, "-dw");
	entity_cnts_file = GetArgValue(argc, argv, "-ecnt");
	word_cnts_file = GetArgValue(argc, argv, "-wcnt");
	dst_doc_vecs_file = GetArgValue(argc, argv, "-docvec");
	dst_word_vecs_file = GetArgValue(argc, argv, "-wordvec");
	dst_entity_vecs_file = GetArgValue(argc, argv, "-entityvec");

	if (!ee_file)
	{
		DataSet data_set = DataSet::WIN_20NG;
		configFiles(data_set, ee_file, de_file, dw_file, entity_cnts_file, word_cnts_file,
			dst_doc_vecs_file, dst_word_vecs_file, dst_entity_vecs_file);
	}

	//if (argc >= 7)
	//{
	//	doc_vec_dim = atoi(argv[1]);
	//	num_rounds = atoi(argv[2]);
	//	num_threads = atoi(argv[3]);
	//	num_negative_samples = atoi(argv[4]);
	//	starting_alpha = (float)atof(argv[5]);
	//	min_alpha = (float)atof(argv[6]);
	//	if (argc == 8)
	//		dst_doc_vecs_file = argv[7];
	//}
	//else
	//{
	//	DataSet data_set = DataSet::WIN_20NG;
	//	configFiles(data_set, ee_file, de_file, dw_file, entity_cnts_file, word_cnts_file,
	//		dst_doc_vecs_file, dst_word_vecs_file, dst_entity_vecs_file);
	//}

	printf("ee_file: %s\nde_file: %s\ndw_file: %s\n", ee_file, de_file, dw_file);
	printf("vec_dim: %d\nnum_rounds: %d\nnum_threads: %d\nnum_neg_samples: %d\nstarting_alpha: %f\nmin_alpha: %f\n",
		doc_vec_dim, num_rounds, num_threads, num_negative_samples, starting_alpha, min_alpha);

	EADocVecTrainer eatrain(num_rounds, num_threads, num_negative_samples, starting_alpha, min_alpha);
	eatrain.AllJointThreaded(ee_file, de_file, dw_file,
		entity_cnts_file, word_cnts_file, doc_vec_dim, share_doc_vec, dst_doc_vecs_file, dst_word_vecs_file,
		dst_entity_vecs_file);
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

	//TrainDocWordVectors();
	//EATrainDWEFixed();
	//EATrainDW(argc, argv);
	EATrain(argc, argv);

	time_t et = time(0) - t;
	printf("\n%lld s. %lld m. %lld h.\n", et, et / 60, et / 3600);

	return 0;
}
