#include "io_utils.h"

#include <cstdio>
#include <cassert>

void IOUtils::SaveVectors(float **vecs, int vec_dim, int num_vecs,
	const char *dst_file_name)
{
	FILE *fp = fopen(dst_file_name, "wb");
	assert(fp != 0);

	fwrite(&num_vecs, 4, 1, fp);
	fwrite(&vec_dim, 4, 1, fp);

	for (int i = 0; i < num_vecs; ++i)
		fwrite(vecs[i], 4, vec_dim, fp);

	fclose(fp);
}

void IOUtils::LoadVectors(const char *file_name, int &num_vecs, int &vec_dim, 
	float **&vecs)
{
	FILE *fp = fopen(file_name, "rb");
	assert(fp != 0);

	fread(&num_vecs, 4, 1, fp);
	fread(&vec_dim, 4, 1, fp);

	vecs = new float*[num_vecs];
	for (int i = 0; i < num_vecs; ++i)
	{
		vecs[i] = new float[vec_dim];
		fread(vecs[i], 4, vec_dim, fp);
	}

	fclose(fp);
}

void IOUtils::LoadCountsFile(const char *file_name, int &num, int *&cnts)
{
	FILE *fp = fopen(file_name, "rb");
	fread(&num, 4, 1, fp);
	cnts = new int[num];
	fread(cnts, 4, num, fp);
	fclose(fp);
}

//void IOUtils::LoadNetEdgeListText(const char *file_name, Edge *&dst_edges, int *&weights, 
//	int &num_edges, int &num_objs_left, int &num_objs_right)
//{
//	if (file_name == 0)
//		return;
//
//	printf("Loading edge list file %s ...\n", file_name);
//
//	FILE *fp = fopen(file_name, "r");
//	assert(fp != 0);
//
//	fscanf(fp, "%d %d %d", &num_objs_left, &num_objs_right, &num_edges);
//	dst_edges = new Edge[num_edges];
//	weights = new int[num_edges];
//
//	for (int i = 0; i < num_edges; ++i)
//		fscanf(fp, "%d %d %d", &dst_edges[i].va, &dst_edges[i].vb, &weights[i]);
//
//	fclose(fp);
//
//	printf("Done.\n");
//}

void IOUtils::LoadNetAdjListText(const char *file_name, int &num_vertices,
	int *&num_adj_vertices, int **&adj_vertices, int **&weights)
{

}

void IOUtils::LoadNetAdjListBin(const char * file_name, int & num_vertices, 
	int *& num_adj_vertices, int **& adj_vertices, int **& weights)
{
	FILE *fp = fopen(file_name, "rb");
	assert(fp != 0);

	fread(&num_vertices, 4, 1, fp);
	printf("%d vertices\n", num_vertices);
	num_adj_vertices = new int[num_vertices];
	adj_vertices = new int*[num_vertices];
	weights = new int*[num_vertices];
	for (int i = 0; i < num_vertices; ++i)
	{
		fread(&num_adj_vertices[i], 4, 1, fp);
		adj_vertices[i] = new int[num_adj_vertices[i]];
		weights[i] = new int[num_adj_vertices[i]];
		fread(adj_vertices[i], 4, num_adj_vertices[i], fp);
		fread(weights[i], 4, num_adj_vertices[i], fp);
	}

	fclose(fp);
}
