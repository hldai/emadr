#ifndef ADJ_LIST_NET_H_
#define ADJ_LIST_NET_H_

#include "edge_net.h"

struct AdjListNet
{
	~AdjListNet();

	int num_vertices_left = -1;
	int num_vertices_right = -1;
	int **adj_vertices = 0;
	int **weights;
	int *num_adj_vertices = 0;

	void LoadBinFile(const char *file_name);
	void ToEdgeNet(EdgeNet &edge_net);
	// count the occurrences of the vertices on the right side
	int *CountRightVertices();
};

#endif
