#ifndef IOUTILS_H_
#define IOUTILS_H_

class IOUtils
{
public:
	static void SaveVectors(float **vecs, int vec_dim, int num_vecs,
		const char *dst_file_name);
	static void LoadVectors(const char *file_name, int &num_vecs,
		int &vec_dim, float **&vecs);

	static void LoadCountsFile(const char *file_name, int &num, int *&cnts);

	static void LoadPairsAdjListText(const char *file_name, int &num_vertices,
		int *&num_adj_vertices, int **&adj_vertices, int **&weights);
	static void LoadPairsAdjListBin(const char *file_name, int &num_vertices,
		int *&num_adj_vertices, int **&adj_vertices, int **&weights);
};

#endif
