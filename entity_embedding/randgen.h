#ifndef RANDGEN_H_
#define RANDGEN_H_

class RandGen
{
public:
	RandGen() {}
	RandGen(unsigned long long seed) : next_random_(seed) {}

	void SetSeed(unsigned long long seed)
	{
		next_random_ = seed;
	}

	long long NextRandom()
	{
		next_random_ = next_random_ * (unsigned long long)25214903917 + 11;
		return (long long)(next_random_ << 1);
	}

private:
	unsigned long long next_random_ = 1;
};

#endif
