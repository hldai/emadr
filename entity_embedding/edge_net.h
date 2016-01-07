#ifndef EDGE_NET_H_
#define EDGE_NET_H_

struct Edge
{
	int va;
	int vb;
};

struct EdgeNet
{
	~EdgeNet();
	void LoadTextFile(const char *file_name);

	int num_vertices_left = 0;
	int num_vertices_right= 0;
	int num_edges = 0;
	Edge *edges = 0;
	int *weights = 0;
};

#endif
