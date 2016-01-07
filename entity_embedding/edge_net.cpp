#include "edge_net.h"

#include <cstdio>
#include <cassert>

EdgeNet::~EdgeNet()
{
	if (edges != 0)
	{
		delete[] edges;
		delete[] weights;
	}
}

void EdgeNet::LoadTextFile(const char *file_name)
{
	if (file_name == 0)
		return;

	printf("Loading edge list file %s ...\n", file_name);

	FILE *fp = fopen(file_name, "r");
	assert(fp != 0);

	fscanf(fp, "%d %d %d", &num_vertices_left, &num_vertices_right, &num_edges);
	edges = new Edge[num_edges];
	weights = new int[num_edges];

	for (int i = 0; i < num_edges; ++i)
		fscanf(fp, "%d %d %d", &edges[i].va, &edges[i].vb, &weights[i]);

	fclose(fp);

	printf("Done.\n");
}
