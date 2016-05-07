#include <stdio.h>
#include <stdlib.h>

int main(const int argc, const char *argv[])
{
	if (argc != 5) {
		printf("usage: m n k batch\n");
		return -1;
	}
	
	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);
	int batch = atoi(argv[4]);

	float *A, *B, *C;

	A = (float*)malloc(sizeof(float)*M*K*batch);
	B = (float*)malloc(sizeof(float)*K*N*batch);
	C = (float*)malloc(sizeof(float)*M*N*batch);

	for (int i = 0; i < M*K*batch; i++) {
		A[i] = rand()/(float)RAND_MAX - rand()/(float)RAND_MAX;
	}
	for (int i = 0; i < K*N*batch; i++) {
		B[i] = rand()/(float)RAND_MAX - rand()/(float)RAND_MAX;
	}

	for (int b = 0; b < batch; b++) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				float val = 0;
				for (int k = 0; k < K; k++) {
					val += A[b*M*K+k*M+i] * B[b*K*N+k*N+j];
				}
				C[b*M*N+i*N+j] = val;
			}
		}
	}

	FILE *fp = fopen("cpu.out", "w");
	for (int b = 0; b < batch; b++) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				fprintf(fp, "%f\n", C[b*M*N+i*N+j]);
			}
		}
	}

	fclose(fp);

	return 0;
}
