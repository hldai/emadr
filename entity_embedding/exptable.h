#ifndef EXPTABLE_H_
#define EXPTABLE_H_

class ExpTable
{
public:
	ExpTable(int table_size = 10000, float max_exp = 6);
	~ExpTable();

	float getSigmaValue(float x)
	{
		if (x > max_exp_)
			return 1;
		else if (x < -max_exp_)
			return 0;
		return exp_table_[(int)((x + max_exp_) * (table_size_ / max_exp_ / 2))];
	}

private:
	int table_size_;
	float max_exp_;
	float *exp_table_ = 0;
};

#endif
