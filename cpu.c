#include <stdio.h>
#include <stdlib.h>

int main(const int argc, const char *argv[])
{
	if (argc != 4) {
		printf("usage: m n k\n");
		return -1;
	}
	
	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	float *A, *B, *C;

	A = (float*)malloc(sizeof(float)*M*K);
	B = (float*)malloc(sizeof(float)*K*N);
	C = (float*)malloc(sizeof(float)*M*N);

	for (int i = 0; i < M*K; i++) {
		A[i] = rand()/(float)RAND_MAX - rand()/(float)RAND_MAX;
	}
	for (int i = 0; i < K*N; i++) {
		B[i] = rand()/(float)RAND_MAX - rand()/(float)RAND_MAX;
	}

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			float val = 0;
			for (int k = 0; k < K; k++) {
				val += A[i*K+k] * B[k*N+j];
			}
			C[i*N+j] = val;
		}
	}

	FILE *fp = fopen("cpu.out", "w");
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			fprintf(fp, "%f\n", C[i*N+j]);
		}
	}

	fclose(fp);

	return 0;
}
