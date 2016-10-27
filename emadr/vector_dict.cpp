#include "vector_dict.h"

#include <cstdio>
#include <cassert>
#include <cstring>

//VectorDict::VectorDict(const char *entity_vec_file_name, bool is_bin_file)
//{
//	if (is_bin_file)
//		loadBinDataFile(entity_vec_file_name);
//	else
//		loadTextDataFile(entity_vec_file_name);
//	printf("done reading entity vectors.\n");
//}

VectorDict::~VectorDict()
{
	if (vecs_ != 0)
	{
		for (int i = 0; i < num_vecs_; ++i)
			delete[] vecs_[i];
		delete[] vecs_;
	}
}

void VectorDict::SaveAsBinFile(const char *file_name)
{
	FILE *fp = fopen(file_name, "wb");
	assert(fp != NULL);

	fwrite(&num_vecs_, sizeof(int), 1, fp);
	fwrite(&vec_len_, sizeof(int), 1, fp);
	for (int i = 0; i < num_vecs_; ++i)
	{
		// write entity vector of index (i + 1)
		fwrite(vecs_[i], sizeof(float), vec_len_, fp);
	}

	fclose(fp);
}

void VectorDict::loadBinDataFile(const char *file_name)
{
	FILE *fp = fopen(file_name, "rb");
	assert(fp != NULL);

	fread(&num_vecs_, sizeof(int), 1, fp);
	fread(&vec_len_, sizeof(int), 1, fp);
	printf("%d %d\nreading vectors...\n", num_vecs_, vec_len_);
	vecs_ = new float*[num_vecs_];
	for (int i = 0; i < num_vecs_; ++i)
	{
		vecs_[i] = new float[vec_len_];
		fread(vecs_[i], sizeof(float), vec_len_, fp);
	}

	fclose(fp);
}

void VectorDict::loadTextDataFile(const char *file_name)
{
	if (file_name == NULL)
		return;

	FILE *fin = fopen(file_name, "r");
	assert(fin != 0);

	fscanf(fin, "%d %d", &num_vecs_, &vec_len_);
	fgetc(fin);
	printf("%d %d\nreading vectors...\n", num_vecs_, vec_len_);

	vecs_ = new float*[num_vecs_];
	for (int i = 0; i < num_vecs_; ++i)
		vecs_[i] = new float[vec_len_];

	for (int i = 0; i < num_vecs_; ++i)
	{
		int idx = 0;

		fscanf(fin, "%d", &idx);
		for (int j = 0; j < vec_len_; ++j)
		{
			fscanf(fin, "%f", &vecs_[idx - 1][j]);
		}
		//if (i % 10000 == 0)
		//	printf("%d\n", i);
	}
	//delete[] buf;

	fclose(fin);
}
