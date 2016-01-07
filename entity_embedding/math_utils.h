#ifndef MATH_UTILS_H_
#define MATH_UTILS_H_

#include <cmath>

namespace MathUtils
{
	static int Sum(int *arr, int len)
	{
		int s = 0;
		for (int i = 0; i < len; ++i)
			s += arr[i];
		return s;
	}

	static float DotProduct(float *vec0, float *vec1, int len)
	{
		float dot_prod = 0;
		for (int i = 0; i < len; ++i)
			dot_prod += vec0[i] * vec1[i];
		return dot_prod;
	}

	static void ElementWiseDivide(float *vec, int len, float divisor)
	{
		for (int i = 0; i < len; ++i)
		{
			vec[i] /= divisor;
		}
	}

	static float Sigma(float x)
	{
		return 1 / (1 + exp(-x));
	}

	static float Dist(float *vec0, float *vec1, int len)
	{
		float rslt = 0;
		for (int i = 0; i < len; ++i)
		{
			rslt += (vec0[i] - vec1[i]) * (vec0[i] - vec1[i]);
		}
		return rslt;
	}

	static float Norm(float *vec, int len)
	{
		float val = 0;
		for (int i = 0; i < len; ++i)
			val += vec[i] * vec[i];
		return sqrt(val);
	}

	static float Cosine(float *vec0, float *vec1, int len)
	{
		float dp = DotProduct(vec0, vec1, len);
		return dp / Norm(vec0, len) / Norm(vec1, len);
	}

	// matrix: [ x11, x12, ..., x1n, x21, ..., xmn ]
	static float XMY(float *vec0, int len0, float *vec1, int len1, float *matrix)
	{
		float result = 0;
		for (int i = 0; i < len0; ++i)
			for (int j = 0; j < len1; ++j)
				result += vec0[i] * matrix[i * len1 + j] * vec1[j];

		return result;
	}

	// matrix: [ x11, x12, ..., x1n, x21, ..., xmn ]
	static void MY(float *matrix, float *vec, int m, int n, float *dst_vec)
	{
		for (int i = 0; i < m; ++i)
		{
			dst_vec[i] = 0;
			for (int j = 0; j < n; ++j)
				dst_vec[i] += matrix[i * n + j] * vec[j];
		}
	}

	static float NormSqr(float *vec, int len)
	{
		float result = 0;
		for (int i = 0; i < len; ++i)
			result += vec[i] * vec[i];
		return result;
	}
}

#endif
