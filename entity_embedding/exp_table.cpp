#include "exp_table.h"

#include <cmath>

ExpTable::ExpTable(int table_size, float max_exp): table_size_(table_size), max_exp_(max_exp)
{
	exp_table_ = new float[table_size + 1];
	for (int i = 0; i < table_size; ++i)
	{
		exp_table_[i] = exp((i / (float)table_size * 2 - 1) * max_exp);
		exp_table_[i] = exp_table_[i] / (exp_table_[i] + 1);
	}
}

ExpTable::~ExpTable()
{
	delete[] exp_table_;
}
