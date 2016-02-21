#ifndef RAND_GEN_H_
#define RAND_GEN_H_

class RandGen
{
public:
	RandGen() {}
	RandGen(unsigned long long seed) : next_random_(seed) {}

	void SetSeed(unsigned long long seed)
	{
		next_random_ = seed;
	}

	unsigned long long NextRandom()
	{
		next_random_ = next_random_ * (unsigned long long)25214903917 + 11;
		return next_random_;
	}

private:
	unsigned long long next_random_ = 1;
};

#endif
