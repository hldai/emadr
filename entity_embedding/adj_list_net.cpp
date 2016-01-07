#include "adj_list_net.h"

#include <cstdio>
#include <cassert>
#include <algorithm>

#include "mem_utils.h"

AdjListNet::~AdjListNet()
{
	if (adj_vertices != 0)
	{
		MemUtils::Release(adj_vertices, num_vertices_left);
		MemUtils::Release(weights, num_vertices_left);
		delete[] num_adj_vertices;
	}
}

void AdjListNet::LoadBinFile(const char *file_name)
{
	FILE *fp = fopen(file_name, "rb");
	assert(fp != 0);

	fread(&num_vertices_left, 4, 1, fp);
	fread(&num_vertices_right, 4, 1, fp);
	printf("%d left vertices. %d right vertices.\n", num_vertices_left, num_vertices_right);
	num_adj_vertices = new int[num_vertices_left];
	adj_vertices = new int*[num_vertices_left];
	weights = new int*[num_vertices_left];
	for (int i = 0; i < num_vertices_left; ++i)
	{
		fread(&num_adj_vertices[i], 4, 1, fp);
		adj_vertices[i] = new int[num_adj_vertices[i]];
		weights[i] = new int[num_adj_vertices[i]];
		fread(adj_vertices[i], 4, num_adj_vertices[i], fp);
		fread(weights[i], 4, num_adj_vertices[i], fp);
	}

	fclose(fp);
}

void AdjListNet::ToEdgeNet(EdgeNet &edge_net)
{
	edge_net.num_edges = 0;
	for (int i = 0; i < num_vertices_left; ++i)
		edge_net.num_edges += num_adj_vertices[i];

	edge_net.edges = new Edge[edge_net.num_edges];
	edge_net.weights = new int[edge_net.num_edges];
	edge_net.num_vertices_left = num_vertices_left;
	int idx = 0;
	for (int i = 0; i < num_vertices_left; ++i)
	{
		for (int j = 0; j < num_adj_vertices[i]; ++j)
		{
			edge_net.edges[idx].va = i;
			edge_net.edges[idx].vb = adj_vertices[i][j];
			edge_net.weights[idx] = weights[i][j];
			++idx;
		}
	}
}

int *AdjListNet::CountRightVertices()
{
	int *cnts = new int[num_vertices_right];
	std::fill(cnts, cnts + num_vertices_right, 0);
	for (int i = 0; i < num_vertices_left; ++i)
		for (int j = 0; j < num_adj_vertices[i]; ++j)
			cnts[adj_vertices[i][j]] += weights[i][j];

	return cnts;
}
