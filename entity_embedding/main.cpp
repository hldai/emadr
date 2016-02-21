#include <cstdio>
#include <ctime>
#include <random>
#include <array>
#include <cassert>
#include <thread>
#include <iostream>

#include "entity_set_trainer.h"
#include "entity_vec_trainer.h"
#include "io_utils.h"
#include "joint_trainer.h"
#include "math_utils.h"
#include "net_edge_sampler.h"
#include "paragraph_vector.h"
#include "eadocvectrainer.h"

void TrainParagraphVectors()
{
	const char *dw_net_file_name = "e:/dc/20ng_bydate/all_docs_dw_net.bin";

	//const char *doc_word_indices_file_name = "e:/dc/20ng_data/doc_word_indices.txt";
	//const char *dict_file_name = "e:/dc/20ng_data/doc_word_indices_dict.txt";

	const char *dst_vec_file_name = "e:/dc/20ng_bydate/vecs/doc_vec_cpp_100.bin";
	const char *dst_word_vec_file_name = "e:/dc/20ng_bydate/vecs/word_vecs_cpp_100.bin";
	ParagraphVector pv(dw_net_file_name, 0.02f, 5);
	int vec_dim = 200;
	int num_threads = 4;
	int num_rounds = 15;
	pv.Train(vec_dim, num_threads, num_rounds, dst_vec_file_name, dst_word_vec_file_name);
}

void TrainEntityVectors()
{
	const char *entity_net_file_name = "e:/dc/nyt/sentence_based_weighted_edge_list.txt";
	const char *doc_entity_net_file_name = "e:/dc/nyt/doc_entities_lo_f2012.bin";
	const char *doc_words_file_name = "e:/dc/nyt/line_docs/doc_word_indices_rm_ssw_lo_2012.bin";
	const char *dst_vec_file_name = "e:/dc/nyt/entity_vecs/entity_vecs_64_tmp.bin";
	const char *dst_output_vec_file_name = "e:/dc/nyt/entity_vecs/syn1_64.bin";
	const int vec_dim = 64;
	const int num_rounds = 10;
	const int num_threads = 4;
	const int num_negative_samples = 5;
	//EntityVecTrainer trainer(file_name);
	//trainer.ThreadedTrain(vec_dim, num_rounds, num_threads, num_negative_samples, 
	//	dst_vec_file_name, dst_output_vec_file_name);
	JointTrainer jt(entity_net_file_name, doc_entity_net_file_name, doc_words_file_name);
	jt.TrainEntityNetThreaded(vec_dim, num_rounds, num_threads, num_negative_samples, dst_vec_file_name,
		dst_output_vec_file_name);
}

void TrainEntitySetVectors()
{
	const char *entity_vec_file_name = "e:/dc/nyt/entity_vecs/entity_vecs_64_tmp.bin";
	const char *doc_entity_list_file_name = "e:/dc/nyt/doc_entities_lo_f2012.txt";
	const char *dst_entity_set_vec_file_name = "e:/dc/nyt/vecs/es_doc_vec_64_lo_f2012_tmp.bin";

	EntitySetTrainer trainer(entity_vec_file_name);
	trainer.Train(doc_entity_list_file_name, dst_entity_set_vec_file_name);
}

void TrainEntitySetVectorsJoint()
{
	const char *doc_entity_net_file_name = "e:/dc/nyt/doc_entities_lo_f2012.bin";
	const char *entity_vec_file_name = "e:/dc/nyt/entity_vecs/entity_vecs_64_tmp.bin";
	const char *doc_words_file_name = "e:/dc/nyt/line_docs/doc_word_indices_rm_ssw_lo_2012.bin";
	const char *dst_doc_vec_file_name = "e:/dc/nyt/vecs/es_doc_vec_64_lo_f2012_tmp.bin";
	const int vec_dim = 64;
	const int num_rounds = 10;
	const int num_threads = 4;
	const int num_negative_samples = 5;
	JointTrainer jt(0, doc_entity_net_file_name, doc_words_file_name);
	jt.TrainDocEntityNetThreaded(entity_vec_file_name, num_rounds, num_negative_samples, num_threads,
		dst_doc_vec_file_name);
}

void TrainEntitySetVectorsM()
{
	const char *entity_vec_file_name = "e:/dc/nyt/entity_vecs/entity_vecs_64.bin";
	//const char *doc_entity_list_file_name = "e:/dc/nyt/doc_entities.txt";
	const char *doc_entity_list_file_name = "e:/dc/nyt/doc_entities_lo_f2012.txt";
	const char *dst_entity_set_vec_file_name = "e:/dc/nyt/vecs/es_doc_vec_64_lo_f2012_m.bin";

	EntitySetTrainer trainer(entity_vec_file_name);
	trainer.TrainM(doc_entity_list_file_name, 64, dst_entity_set_vec_file_name);
	//trainer.TrainDocVectors(doc_entity_list_file_name, dst_entity_set_vec_file_name);
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
	JointTrainer jt(entity_net_file_name, doc_entity_net_file_name, doc_words_file_name);
	jt.JointTrainingThreaded(entity_vec_dim, word_vec_dim, doc_vec_dim, num_rounds, num_threads, 
		num_negative_samples, dst_doc_vec_file_name);
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
	//sprintf(dst_doc_vec_file_name, "e:/dc/20ng_data/vecs/train_doc_vec_joint_%d.bin", doc_vec_dim);
	//sprintf(dst_doc_vec_file_name, "e:/dc/20ng_data/vecs/test_doc_vec_joint_%d.bin", doc_vec_dim);
	JointTrainer jt(entity_net_file_name, doc_entity_net_file_name, doc_words_file_name);
	jt.JointTrainingThreaded(entity_vec_dim, word_vec_dim, doc_vec_dim, num_rounds, num_threads,
		num_negative_samples, dst_doc_vec_file_name);
}

void JointTrainingOML20NG(int argc, char **argv)
{
	const char *entity_net_file_name = "e:/dc/20ng_bydate/weighted_entity_edge_list.txt";
	const char *doc_entity_net_file_name = "e:/dc/20ng_bydate/doc_entities.bin";
	const char *doc_words_file_name = "e:/dc/20ng_bydate/all_docs_dw_net_short.bin";
	const char *dst_dedw_vecs_file_name = "e:/dc/20ng_bydate/vecs/dedw_vecs.bin";

	//const char *entity_net_file_name = "/home/dhl/data/dc/20ng_bydate/weighted_entity_edge_list.txt";
	//const char *doc_entity_net_file_name = "/home/dhl/data/dc/20ng_bydate/doc_entities.bin";
	//const char *doc_words_file_name = "/home/dhl/data/dc/20ng_bydate/all_docs_dw_net.bin";
	//const char *dst_dedw_vecs_file_name = "/home/dhl/data/dc/20ng_bydate/dedw_vecs.bin";

	int doc_vec_dim = 50;
	int num_rounds = 10;
	int num_threads = 4;
	int num_negative_samples = 5;
	float starting_alpha = 0.06f;
	float ws_rate = 0.7f;
	float min_alpha = 0.0001f;

	if (argc >= 8)
	{
		doc_vec_dim = atoi(argv[1]);
		num_rounds = atoi(argv[2]);
		num_threads = atoi(argv[3]);
		num_negative_samples = atoi(argv[4]);
		starting_alpha = (float)atof(argv[5]);
		ws_rate = (float)atof(argv[6]);
		min_alpha = (float)atof(argv[7]);
		if (argc == 9)
			dst_dedw_vecs_file_name = argv[8];
	}

	printf("vec_dim: %d\nnum_rounds: %d\nnum_threads: %d\nnum_neg_samples: %d\nstarting_alpha: %f\nws_rate: %f\nmin_alpha: %f\n",
		doc_vec_dim, num_rounds, num_threads, num_negative_samples, starting_alpha, ws_rate, min_alpha);

	JointTrainer jt(entity_net_file_name, doc_entity_net_file_name, doc_words_file_name);
	//jt.JointTrainingOMLThreaded(doc_vec_dim, num_rounds, num_threads, num_negative_samples, 
	//	starting_alpha, ws_rate, min_alpha, dst_dedw_vecs_file_name, dst_mixed_vecs_file_name, dst_word_vecs_file_name,
	//	dst_entity_vecs_file_name);
	jt.JointTrainingOMLThreaded(doc_vec_dim, num_rounds, num_threads, num_negative_samples,
		starting_alpha, ws_rate, min_alpha, dst_dedw_vecs_file_name);
}

void JointTrainingCM20NG(int argc, char **argv)
{
	const char *entity_net_file_name = "e:/dc/20ng_bydate/weighted_entity_edge_list.txt";
	const char *doc_entity_net_file_name = "e:/dc/20ng_bydate/doc_entities.bin";
	const char *doc_words_file_name = "e:/dc/20ng_bydate/all_docs_dw_net.bin";
	const char *dst_doc_vecs_file_name = "e:/dc/20ng_bydate/vecs/doc_vecs_0.bin";

	//const char *entity_net_file_name = "/home/dhl/data/dc/20ng_bydate/weighted_entity_edge_list.txt";
	//const char *doc_entity_net_file_name = "/home/dhl/data/dc/20ng_bydate/doc_entities.bin";
	//const char *doc_words_file_name = "/home/dhl/data/dc/20ng_bydate/all_docs_dw_net.bin";
	//const char *dst_dedw_vecs_file_name = "/home/dhl/data/dc/20ng_bydate/dedw_vecs.bin";

	int doc_vec_dim = 50;
	int num_rounds = 10;
	int num_threads = 4;
	int num_negative_samples = 5;
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

	JointTrainer jt(entity_net_file_name, doc_entity_net_file_name, doc_words_file_name);
	//jt.JointTrainingOMLThreaded(doc_vec_dim, num_rounds, num_threads, num_negative_samples, 
	//	starting_alpha, ws_rate, min_alpha, dst_dedw_vecs_file_name, dst_mixed_vecs_file_name, dst_word_vecs_file_name,
	//	dst_entity_vecs_file_name);
	jt.TrainCMThreaded(doc_vec_dim, num_rounds, num_threads, num_negative_samples,
		starting_alpha, min_alpha, dst_doc_vecs_file_name);
}

void EATrain(int argc, char **argv)
{
	//const char *entity_net_file_name = "/home/dhl/data/dc/el/entity_net_adj_list.bin";
	//const char *doc_entity_net_file_name = "/home/dhl/data/dc/el/doc_entities.bin";
	//const char *doc_words_file_name = "/home/dhl/data/dc/el/wiki_bow.bin";
	//const char *dst_doc_vecs_file_name = "/home/dhl/data/dc/el/vecs/wiki_vecs.bin";
	//const char *dst_word_vec_file_name = "/home/dhl/data/dc/el/vecs/word_vecs.bin";
	//const char *dst_entityvec_file_name = "/home/dhl/data/dc/el/vecs/entity_vecs.bin";

	//const char *entity_net_file_name = "/home/dhl/data/dc/20ng_bydate/entity_net_adj_list.bin";
	//const char *doc_entity_net_file_name = "/home/dhl/data/dc/20ng_bydate/doc_entities_short.bin";
	//const char *doc_words_file_name = "/home/dhl/data/dc/20ng_bydate/all_docs_dw_net_short.bin";
	//const char *dst_doc_vecs_file_name = "/home/dhl/data/dc/20ng_bydate/vecs/dedw_vecs.bin";
	//const char *dst_word_vec_file_name = "/home/dhl/data/dc/20ng_bydate/vecs/word_vecs.bin";
	//const char *dst_entityvec_file_name = "/home/dhl/data/dc/20ng_bydate/vecs/entity_vecs.bin";

	//const char *entity_net_file_name = "e:/dc/el/entity_net_adj_list.bin";
	//const char *doc_entity_net_file_name = "e:/dc/el/doc_entities.bin";
	//const char *doc_words_file_name = "e:/dc/el/wiki_bow.bin";
	//const char *dst_doc_vecs_file_name = "e:/dc/el/vecs/wiki_vecs.bin";

	const char *entity_net_file_name = "e:/dc/20ng_bydate/entity_net_adj_list.bin";
	const char *doc_entity_net_file_name = "e:/dc/20ng_bydate/doc_entities_short.bin";
	const char *doc_words_file_name = "e:/dc/20ng_bydate/all_docs_dw_net_short.bin";
	const char *dst_doc_vecs_file_name = "e:/dc/20ng_bydate/vecs/dedw_vecs_0.bin";
	const char *dst_word_vec_file_name = "e:/dc/20ng_bydate/vecs/word_vecs.bin";
	const char *dst_entityvec_file_name = "e:/dc/20ng_bydate/vecs/entity_vecs.bin";

	int doc_vec_dim = 50;
	int num_rounds = 10;
	int num_threads = 4;
	int num_negative_samples = 5;
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

	EADocVecTrainer eatrain;
	eatrain.Init(entity_net_file_name, doc_entity_net_file_name, doc_words_file_name);
	eatrain.AllJointThreaded(doc_vec_dim, num_rounds, num_threads, num_negative_samples,
		starting_alpha, min_alpha, dst_doc_vecs_file_name, dst_word_vec_file_name,
		dst_entityvec_file_name);
}

void Test()
{
	//const char *doc_words_file_name = "e:/dc/20ng_bydate/all_docs_dw_net_short.bin";
	////const char *doc_words_file_name = "/home/dhl/data/dc/20ng_bydate/all_docs_dw_net_short.bin";
	//NetEdgeSampler sampler(doc_words_file_name);
	//int lidx, ridx;
	std::default_random_engine generator(43);
	//int *cnts = new int[sampler.num_vertex_right()];
	//for (int i = 0; i < sampler.num_vertex_right(); ++i)
	//	cnts[i] = 0;
	//for (int i = 0; i < sampler.sum_weights(); ++i)
	//{
	//	sampler.SampleEdge(lidx, ridx, generator);
	//	//printf("%d %d\n", lidx, ridx);
	//	if (lidx == 0)
	//		++cnts[ridx];
	//}
	//for (int i = 0; i < sampler.num_vertex_right(); ++i)
	//{
	//	if (cnts[i] > 0)
	//		printf("%d %d\n", i, cnts[i]);
	//}
	//delete[] cnts;


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
	//TrainEntityVectors();
	//TrainEntitySetVectors();
	//TrainEntitySetVectorsM();
	//TrainParagraphVectors();
	//JointTrainingNYT();
	//JointTraining20NG();
	//JointTrainingOML20NG(argc, argv);
	//JointTrainingCM20NG(argc, argv);
	EATrain(argc, argv);

	time_t et = time(0) - t;
	printf("\n%lld s. %lld m. %lld h.\n", et, et / 60, et / 3600);

	return 0;
}
