#ifndef IO_UTILS_H_
#define IO_UTILS_H_

#include "edge_net.h"

class IOUtils
{
public:
	static void SaveVectors(float **vecs, int vec_dim, int num_vecs,
		const char *dst_file_name);
	static void LoadVectors(const char *file_name, int &num_vecs,
		int &vec_dim, float **&vecs);

	static void LoadNetEdgeListText(const char *file_name, Edge *&dst_edges,
		int *&weights, int &num_edges, int &num_objs_left, int &num_objs_right);

	static void LoadNetAdjListText(const char *file_name, int &num_vertices,
		int *&num_adj_vertices, int **&adj_vertices, int **&weights);
	static void LoadNetAdjListBin(const char *file_name, int &num_vertices,
		int *&num_adj_vertices, int **&adj_vertices, int **&weights);
};

#endif
