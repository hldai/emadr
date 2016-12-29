#include <cstdio>
#include <ctime>
#include <random>
#include <array>
#include <cassert>
#include <thread>
#include <iostream>
#include <cstring>
#include <map>

#include "ioutils.h"
#include "mathutils.h"
#include "pairsampler.h"
#include "eadocvectrainer.h"

enum DataSet {
	NYT_ARTS,
	NYT_WORLD,
	NYT_BUSINESS,
	NYT_SPORTS,
	TWNG
};

void TrainDocWordVectors()
{
	int vec_dim = 50;
	int num_threads = 1;
	int num_rounds = 5;
	int num_negative_samples = 5;
	float starting_alpha = 0.06f;

	const char *doc_words_file_name = "e:/data/emadr/el/tac/2010/eval/dw.bin";
	const char *word_cnts_file = "e:/data/emadr/el/wiki/word_cnts.bin";
	const char *word_vecs_file_name = "e:/data/emadr/el/vecs/word_vecs_3.bin";
	const char *dst_vec_file_name = "e:/data/emadr/el/tac/2010/eval/train_3_dw_vecs.bin";

	EADocVecTrainer trainer(num_rounds, num_threads, num_negative_samples, starting_alpha);
	trainer.TrainDocWordFixedWordVecs(doc_words_file_name, word_cnts_file, word_vecs_file_name, 
		vec_dim, dst_vec_file_name);
}

void EATrainDWEFixed()
{
	//const char *de_file_name = "e:/data/emadr/el/tac/2010/eval/de.bin";
	//const char *doc_words_file_name = "e:/data/emadr/el/tac/2010/eval/dw.bin";
	//const char *dst_doc_vecs_file_name = "e:/data/emadr/el/tac/2010/eval/doc_vecs_4.bin";

	//const char *de_file_name = "e:/data/emadr/el/tac/2009/eval/de.bin";
	//const char *doc_words_file_name = "e:/data/emadr/el/tac/2009/eval/dw.bin";
	//const char *dst_doc_vecs_file_name = "e:/data/emadr/el/tac/2009/eval/doc_vecs_4.bin";

	//const char *de_file_name = "e:/data/emadr/el/tac/2011/eval/de.bin";
	//const char *doc_words_file_name = "e:/data/emadr/el/tac/2011/eval/dw.bin";
	//const char *dst_doc_vecs_file_name = "e:/data/emadr/el/tac/2011/eval/doc_vecs_4.bin";

	const char *de_file_name = "e:/data/emadr/el/tac/2014/eval/de.bin";
	const char *doc_words_file_name = "e:/data/emadr/el/tac/2014/eval/dw.bin";
	const char *dst_doc_vecs_file_name = "e:/data/emadr/el/tac/2014/eval/doc_vecs_3.bin";

	const char *word_cnts_file = "e:/data/emadr/el/wiki/word_cnts.bin";
	const char *entity_cnts_file = "e:/data/emadr/el/wiki/entity_cnts.bin";

	const char *word_vecs_file_name = "e:/data/emadr/el/vecs/word_vecs_3.bin";
	const char *entity_vecs_file_name = "e:/data/emadr/el/vecs/entity_vecs_3.bin";
	//const char *word_vecs_file_name = "e:/data/emadr/el/vecs/word_vecs_4.bin";
	//const char *entity_vecs_file_name = "e:/data/emadr/el/vecs/entity_vecs_4.bin";

	int dw_vec_dim = 50;
	int doc_vec_dim = 100;
	int num_rounds = 20;
	int num_threads = 4;
	int num_negative_samples = 10;
	float starting_alpha = 0.06f;
	float min_alpha = 0.0001f;
	bool share_vecs = false;
	//bool share_vecs = true;

	printf("vec_dim: %d\nnum_rounds: %d\nnum_threads: %d\nnum_neg_samples: %d\nstarting_alpha: %f\nmin_alpha: %f\n",
		doc_vec_dim, num_rounds, num_threads, num_negative_samples, starting_alpha, min_alpha);

	EADocVecTrainer eatrain(num_rounds, num_threads, num_negative_samples, starting_alpha, min_alpha);
	if (share_vecs)
		eatrain.TrainWEFixed(doc_words_file_name, de_file_name, word_cnts_file, entity_cnts_file,
			word_vecs_file_name, entity_vecs_file_name, doc_vec_dim, dst_doc_vecs_file_name);
	else
		eatrain.TrainEmadrNewDocs2(doc_words_file_name, de_file_name, word_cnts_file, entity_cnts_file,
			word_vecs_file_name, entity_vecs_file_name, dw_vec_dim, dst_doc_vecs_file_name);
}

void configFilesDW(DataSet data_set, const char *&dw_file, const char *&word_cnts_file,
	const char *&dst_doc_vecs_file, const char *&dst_word_vecs_file)
{
	switch (data_set)
	{
	case TWNG:
		dw_file = "e:/dc/20ng_bydate/bin/all_docs_dw_short.bin";
		word_cnts_file = "e:/dc/20ng_bydate/bin/word_cnts.bin";
		dst_doc_vecs_file = "e:/dc/20ng_bydate/vecs/dw-vecs.bin";
		dst_word_vecs_file = "e:/dc/20ng_bydate/vecs/dw-word-vecs.bin";
		break;
	case NYT_WORLD:
		dw_file = "e:/dc/nyt-world-full/processed/bin/dw.bin";
		word_cnts_file = "e:/dc/nyt-world-full/processed/bin/word-cnts.bin";
		dst_doc_vecs_file = "e:/dc/nyt-world-full/processed/vecs/dw-vecs.bin";
		dst_word_vecs_file = "e:/dc/nyt-world-full/processed/vecs/dw-word-vecs.bin";
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

	DataSet data_set = DataSet::NYT_WORLD;
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

void configFiles(DataSet data_set, char *&ee_file, char *&de_file, char *&dw_file,
	char *&entity_cnts_file, char *&word_cnts_file, char *&dst_doc_vecs_file,
	char *&dst_word_vecs_file, char *&dst_entity_vecs_file)
{
	const char *datadir = 0;
	//const char *datadir = "e:/data/emadr/20ng_bydate/";
	//const char *datadir = "e:/data/emadr/nyt-world-full/processed/";
	switch (data_set)
	{
	case NYT_WORLD:
		datadir = "e:/data/emadr/nyt-all/arts";
		break;
	case NYT_ARTS:
		datadir = "e:/data/emadr/nyt-all/arts";
		break;
	default:
		break;
	}
	const int kPathLen = 256;
	ee_file = new char[kPathLen];
	de_file = new char[kPathLen];
	dw_file = new char[kPathLen];
	entity_cnts_file = new char[kPathLen];
	word_cnts_file = new char[kPathLen];
	dst_doc_vecs_file = new char[kPathLen];
	dst_word_vecs_file = new char[kPathLen];
	dst_entity_vecs_file = new char[kPathLen];

	sprintf(ee_file, "%s/ee.bin", datadir);
	sprintf(de_file, "%s/de.bin", datadir);
	sprintf(dw_file, "%s/dw-2.bin", datadir);
	sprintf(entity_cnts_file, "%s/entity-cnts.bin", datadir);
	sprintf(word_cnts_file, "%s/word-cnts-2.bin", datadir);
	sprintf(dst_doc_vecs_file, "%s/vecs/dew-vecs.bin", datadir);
	sprintf(dst_word_vecs_file, "%s/vecs/word-vecs.bin", datadir);
	sprintf(dst_entity_vecs_file, "%s/vecs/entity-vecs.bin", datadir);
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
	char *ee_file, *de_file, *dw_file, *entity_cnts_file, *word_cnts_file, 
		*dst_doc_vecs_file, *dst_word_vecs_file,
		*dst_entity_vecs_file;

	bool share_doc_vec = 1;

	int doc_vec_dim = GetIntArgValue(argc, argv, "-d", 100);
	int num_rounds = GetIntArgValue(argc, argv, "-r", 10);
	int num_threads = GetIntArgValue(argc, argv, "-t", 4);
	int num_negative_samples = GetIntArgValue(argc, argv, "-n", 10);
	float starting_alpha = GetFloatArgValue(argc, argv, "-sa", 0.06f);
	float weight_ee = GetFloatArgValue(argc, argv, "-wee", 1);
	float weight_de = GetFloatArgValue(argc, argv, "-wde", 1);
	float weight_dw = GetFloatArgValue(argc, argv, "-wdw", 1);
	float min_alpha = GetFloatArgValue(argc, argv, "-ma", 0.0001f);

	ee_file = GetArgValue(argc, argv, "-ee");
	de_file = GetArgValue(argc, argv, "-de");
	dw_file = GetArgValue(argc, argv, "-dw");
	entity_cnts_file = GetArgValue(argc, argv, "-ecnt");
	word_cnts_file = GetArgValue(argc, argv, "-wcnt");
	dst_doc_vecs_file = GetArgValue(argc, argv, "-docvec");
	dst_word_vecs_file = GetArgValue(argc, argv, "-wordvec");
	dst_entity_vecs_file = GetArgValue(argc, argv, "-entityvec");

	//if (!ee_file)
	//{
	//	DataSet data_set = DataSet::WIN_20NG;
	//	configFiles(data_set, ee_file, de_file, dw_file, entity_cnts_file, word_cnts_file,
	//		dst_doc_vecs_file, dst_word_vecs_file, dst_entity_vecs_file);
	//}

	if (!dw_file)
	{
		DataSet data_set = DataSet::NYT_ARTS;
		configFiles(data_set, ee_file, de_file, dw_file, entity_cnts_file, word_cnts_file,
			dst_doc_vecs_file, dst_word_vecs_file, dst_entity_vecs_file);
	}

	printf("vec_dim: %d\nnum_rounds: %d\nnum_threads: %d\nnum_neg_samples: %d\nstarting_alpha: %f\nmin_alpha: %f\n",
		doc_vec_dim, num_rounds, num_threads, num_negative_samples, starting_alpha, min_alpha);
	printf("wee: %f\twde: %f\twdw: %f\n", weight_ee, weight_de, weight_dw);
	printf("ee_file: %s\nde_file: %s\ndw_file: %s\n", ee_file, de_file, dw_file);
	printf("dst_doc_vec_file: %s\n", dst_doc_vecs_file);

	EADocVecTrainer eatrain(num_rounds, num_threads, num_negative_samples, starting_alpha, min_alpha);
	eatrain.AllJointThreaded(ee_file, de_file, dw_file, entity_cnts_file, word_cnts_file, doc_vec_dim, share_doc_vec, 
		weight_ee, weight_de, weight_dw, dst_doc_vecs_file, dst_word_vecs_file,
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
