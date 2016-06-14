#include "ioutils.h"

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

void IOUtils::LoadPairsAdjListText(const char *file_name, int &num_vertices,
	int *&num_adj_vertices, int **&adj_vertices, int **&weights)
{

}

void IOUtils::LoadPairsAdjListBin(const char * file_name, int & num_vertices,
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
