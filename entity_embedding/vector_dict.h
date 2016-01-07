#ifndef VECTOR_DICT_H_
#define VECTOR_DICT_H_

class VectorDict
{
public:
	VectorDict(const char *entity_vec_file_name, bool is_bin_file);
	~VectorDict();

	float *GetVector(int idx) { return vecs_[idx]; }

	void SaveAsBinFile(const char *file_name);

	float **vecs() { return vecs_; }
	int num_vectors() { return num_vecs_; }
	int vec_len() { return vec_len_; }

private:
	void loadBinDataFile(const char *file_name);
	void loadTextDataFile(const char *file_name);

private:
	float **vecs_ = 0;
	int num_vecs_;
	int vec_len_;
};

#endif
