#include <iostream>
#include <random>
#include <cmath>

using namespace std;

#define N_POINT 500000

int main(int argc, char const *argv[])
{
	random_device rand_dev;
	mt19937 gen(rand_dev());
	uniform_real_distribution<float> dist(0.0f, 1.0f);

	int nWithinCircle = 0;
	for(int i = 0; i < N_POINT; i++){
		float x = dist(gen);
		float y = dist(gen);
		nWithinCircle += sqrt(x * x + y * y) <= 1.0f ? 1 : 0;
	}
	float pi = (4.0f * nWithinCircle / N_POINT);
	cout << "PI = " << pi << endl;
	return 0;
}