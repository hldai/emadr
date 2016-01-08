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
	const char *doc_word_indices_file_name = "e:/dc/nyt/line_docs/doc_word_indices_rm_ssw_lo_2012.txt";
	const char *dict_file_name = "e:/dc/nyt/doc_word_indices_dict.txt";
	const char *dst_vec_file_name = "e:/dc/nyt/vecs/doc_vec_cpp_64.bin";
	ParagraphVector pv(doc_word_indices_file_name, dict_file_name);
	int vec_dim = 64;
	int num_threads = 4;
	pv.Train(vec_dim, num_threads, dst_vec_file_name);
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

void JointTraining()
{
	const char *entity_net_file_name = "e:/dc/nyt/sentence_based_weighted_edge_list.txt";
	const char *doc_entity_net_file_name = "e:/dc/nyt/doc_entities_lo_f2012.bin";
	const char *doc_words_file_name = "e:/dc/nyt/line_docs/doc_word_indices_rm_ssw_lo_2012.bin";
	const char *dst_doc_vec_file_name = "e:/dc/nyt/vecs/es_doc_vec_64_lo_f2012_joint_128bin";
	const int entity_vec_dim = 64;
	const int word_vec_dim = 64;
	const int doc_vec_dim = 128;
	const int num_rounds = 5;
	const int num_threads = 4;
	const int num_negative_samples = 5;
	JointTrainer jt(entity_net_file_name, doc_entity_net_file_name, doc_words_file_name);
	jt.JointTrainingThreaded(entity_vec_dim, word_vec_dim, doc_vec_dim, num_rounds, num_threads, 
		num_negative_samples, dst_doc_vec_file_name);
}

int main()
{
	time_t t = time(0);

	//TrainEntityVectors();
	//TrainEntitySetVectors();
	//TrainEntitySetVectorsM();
	//TrainParagraphVectors();
	JointTraining();

	int et = time(0) - t;
	printf("%d s. %d m. %d h.\n", et, et / 60, et / 3600);

	return 0;
}
