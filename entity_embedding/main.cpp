#include <cstdio>
#include <ctime>
#include <random>
#include <array>
#include <cassert>
#include <thread>
#include <iostream>

#include "entity_set_trainer.h"
#include "entity_vec_trainer.h"
#include "joint_trainer.h"
#include "math_utils.h"
#include "paragraph_vector.h"
#include "io_utils.h"

void TrainParagraphVectors()
{
	const char *dw_net_file_name = "e:/dc/20ng_bydate/all_docs_dw_net.bin";

	//const char *doc_word_indices_file_name = "e:/dc/20ng_data/doc_word_indices.txt";
	//const char *dict_file_name = "e:/dc/20ng_data/doc_word_indices_dict.txt";

	const char *dst_vec_file_name = "e:/dc/20ng_bydate/vecs/doc_vec_cpp_100.bin";
	const char *dst_word_vec_file_name = "e:/dc/20ng_bydate/vecs/word_vecs_cpp_100.bin";
	ParagraphVector pv(dw_net_file_name, 0.045f, 5);
	int vec_dim = 100;
	int num_threads = 4;
	pv.Train(vec_dim, num_threads, dst_vec_file_name, dst_word_vec_file_name);
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
	sprintf(dst_doc_vec_file_name, "e:/dc/20ng_bydate/vecs/all_doc_vec_joint_%d.bin", doc_vec_dim,
		num_rounds, num_negative_samples);
	//sprintf(dst_doc_vec_file_name, "e:/dc/20ng_data/vecs/train_doc_vec_joint_%d.bin", doc_vec_dim);
	//sprintf(dst_doc_vec_file_name, "e:/dc/20ng_data/vecs/test_doc_vec_joint_%d.bin", doc_vec_dim);
	JointTrainer jt(entity_net_file_name, doc_entity_net_file_name, doc_words_file_name);
	jt.JointTrainingThreaded(entity_vec_dim, word_vec_dim, doc_vec_dim, num_rounds, num_threads,
		num_negative_samples, dst_doc_vec_file_name);
}

void JointTrainingOML20NG()
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

	const int doc_vec_dim = 100;
	const int num_rounds = 20;
	const int num_threads = 4;
	const int num_negative_samples = 5;
	char dst_dedw_vecs_file_name[256];
	char dst_mixed_vecs_file_name[256];
	//sprintf(dst_dedw_vecs_file_name, "e:/dc/20ng_bydate/vecs/all_doc_vec_joint_oml_%d.bin", doc_vec_dim,
	//	num_rounds, num_negative_samples);
	//sprintf(dst_mixed_vecs_file_name, "e:/dc/20ng_bydate/vecs/all_doc_vec_joint_oml_mixed_%d.bin", doc_vec_dim,
	//	num_rounds, num_negative_samples);
	sprintf(dst_dedw_vecs_file_name, "e:/dc/20ng_bydate/vecs/all_doc_vec_joint_oml_%d_test.bin", doc_vec_dim,
		num_rounds, num_negative_samples);
	sprintf(dst_mixed_vecs_file_name, "e:/dc/20ng_bydate/vecs/all_doc_vec_joint_oml_mixed_%d_test.bin", doc_vec_dim,
		num_rounds, num_negative_samples);
	JointTrainer jt(entity_net_file_name, doc_entity_net_file_name, doc_words_file_name);
	jt.JointTrainingOMLThreaded(doc_vec_dim, num_rounds, num_threads,
		num_negative_samples, dst_dedw_vecs_file_name, dst_mixed_vecs_file_name);
}

void Test()
{
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, 6);
	for (int i = 0; i < 100; ++i)
	{
		printf("%d ", distribution(generator));
	}
}

int main()
{
	time_t t = time(0);

	//TrainEntityVectors();
	//TrainEntitySetVectors();
	//TrainEntitySetVectorsM();
	//TrainParagraphVectors();
	//JointTrainingNYT();
	//JointTraining20NG();
	JointTrainingOML20NG();
	//Test();

	int et = time(0) - t;
	printf("\n%d s. %d m. %d h.\n", et, et / 60, et / 3600);

	return 0;
}
