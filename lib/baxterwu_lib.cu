#include "baxterwu_lib.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>



DECLSPEC bool between(float x, float a, float b) {
    return (x <= b && x >= a) || (x >= b && x <= a);
}

DECLSPEC void gpuAssert(cudaError_t code, const char* file, int line, bool abort) {
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

DECLSPEC __host__ __device__ struct neibors_indexes SLF(int j, int L, int N) {
    //spin lookup function
    struct neibors_indexes result;
    result.right = (j + 1) % L + L * (j / L);
    result.down = (j + L) % N;
    result.diag = (j + L + 1) % N;
    return result;
}

// hardcoded spin suggestion for init
DECLSPEC __device__ int suggestSpin(curandStatePhilox4_32_10_t* state, int r) {
    return (2 * (curand(&state[r]) % 2)) - 1;
};

DECLSPEC __global__ void initializePopulation(curandStatePhilox4_32_10_t* state, int* s, int N, initializePopulationMode mode, int s_a, int s_b, int s_c) {
/*---------------------------------------------------------------------------------------------
	Initializes population on gpu(!) by randomly assigning each spin a value from suggestSpin function or using by-hand seeding
----------------------------------------------------------------------------------------------*/
	int r = threadIdx.x + blockIdx.x * blockDim.x;
    for (int k = 0; k < N; k++) {
        int arrayIndex = r * N + k;
        if (mode == RANDOM) {
            int spin = suggestSpin(state, r);
        }
        else if (mode == BY_SUBLATTICE) {
            s[arrayIndex] = (arrayIndex % 3 == 0) * s_a + (arrayIndex % 3 == 1) * s_b + (arrayIndex % 3 == 2) * s_c;
        }
        else {
        };
    }
};

DECLSPEC void printSpinSample(int* s, int L, int N, int r) {
    int arrayIndex = r * N;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            printf("%d ", s[arrayIndex]);
            arrayIndex++;
        }
        printf("\n");
    }
    printf("\n");
};


//DECLSPEC __device__ struct neibors get_neibors_values(int* s, struct neibors_indexes n_i, int replica_shift) {
//	return {
//		s[n_i.right + replica_shift],
//		s[n_i.down + replica_shift],
//        s[n_i.diag + replica_shift]
//	};
//}
    
//DECLSPEC __host__ __device__ struct energy_parts localEnergyParts(char currentSpin, struct neibors n) {
//	// Computes energy of spin i with neighbors a, b, c, d 
//	// it summirezes each join 2 times
//	return {
//		-(currentSpin * n.up)
//		- (currentSpin * n.right)
//		- (currentSpin * n.down)
//		- (currentSpin * n.left)
//		, (currentSpin * currentSpin)
//	};
//}
//
//__device__ struct energy_parts addEnergyParts(struct energy_parts A, struct energy_parts B) {
//	return { A.Ising + B.Ising, A.Blume + B.Blume };
//}
//
//__device__ struct energy_parts subEnergyParts(struct energy_parts A, struct energy_parts B) {
//	return { A.Ising - B.Ising, A.Blume - B.Blume };
//}
//
//__device__ struct energy_parts calcEnergyParts(char* s, int L, int N, int r) {
//	struct energy_parts sum = { 0, 0 };
//	int replica_shift = r * N;
//
//	for (int j = 0; j < N; j++) {
//		// do not forget double joint summarization!
//		char i = s[j + replica_shift]; // current spin value
//		struct neibors_indexes n_i = SLF(j, L, N);
//		struct neibors n = get_neibors_values(s, n_i, replica_shift); // we look into r replica and j spin
//		struct energy_parts tmp = localEnergyParts(i, n);
//		sum = addEnergyParts(sum, tmp);
//	}
//	return sum;
//}
//
//__device__ int calcEnergyFromParts(struct energy_parts energyParts, int D_div, int D_base) { // D = D_div / D_base
//	return (D_base * energyParts.Ising / 2) + (D_div * energyParts.Blume); // div 2 because of double joint summarization
//}
//
//__global__ void deviceEnergy(char* s, int* E, int L, int N, int D_div, int D_base) {
//	int r = threadIdx.x + blockIdx.x * blockDim.x;
//	struct energy_parts sum = calcEnergyParts(s, L, N, r);
//	E[r] = calcEnergyFromParts(sum, D_div, D_base);
//}
//
//
//// hardcoded spin suggestion for equilibration
//__device__ char suggestSpinSwap(curandStatePhilox4_32_10_t* state, int r, char currentSpin) {
//	return (currentSpin + 2 + (curand(&state[r]) % 2)) % 3 - 1; // little trick
//}
//
//#define FULL_MASK 0xffffffff
//
//__device__ float warpReduceSum(float val)
//{
//	for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
//		val += __shfl_down_sync(FULL_MASK, val, offset);
//	return val;
//}
//
//__global__ void equilibrate(curandStatePhilox4_32_10_t* state, char* s, int* E, int L, int N, int R, int q, int nSteps, int U, int D_div, int D_base, bool heat) {//, int* acceptance_number) {
//	/*---------------------------------------------------------------------------------------------
//		Main Microcanonical Monte Carlo loop.  Performs update sweeps on each replica in the
//		population;
//		There, one could change calcEnergyParts for system of carrying arrays of energy parts,
//		but:
//			1. This is not the bottleneck (which is for loop over N * nSteps
//	---------------------------------------------------------------------------------------------*/
//
//	int r = threadIdx.x + blockIdx.x * blockDim.x;
//	int replica_shift = r * N;
//
//	struct energy_parts baseEnergyParts = calcEnergyParts(s, L, N, r);
//
//	for (int k = 0; k < N * nSteps; k++)
//	{
//		int j = curand(&state[r]) % N;
//		char currentSpin = s[j + replica_shift];
//		char suggestedSpin = suggestSpinSwap(state, r, currentSpin);
//		//char suggestedSpin = curand(&state[r]) % 3 - 1;
//		struct neibors_indexes n_i = SLF(j, L, N);
//		struct neibors n = get_neibors_values(s, n_i, replica_shift);
//		struct energy_parts suggestedLocalEnergyParts = localEnergyParts(suggestedSpin, n);
//
//		struct energy_parts currentLocalEnergyParts = localEnergyParts(currentSpin, n);
//		struct energy_parts deltaLocalEnergyParts = subEnergyParts(suggestedLocalEnergyParts, currentLocalEnergyParts);
//		//local energy delta calculated for single spin; but should be for whole lattice
//		//thus we need to count Ising part twice - for the neibors change in energy as well
//		//but not Blume part!
//		deltaLocalEnergyParts.Ising *= 2;
//		struct energy_parts suggestedEnergyParts = addEnergyParts(baseEnergyParts, deltaLocalEnergyParts);
//		int suggestedEnergy = calcEnergyFromParts(suggestedEnergyParts, D_div, D_base);
//
//		if ((!heat && (suggestedEnergy + EPSILON < U)) || (heat && (suggestedEnergy - EPSILON > U))) {
//			baseEnergyParts = suggestedEnergyParts;
//			E[r] = suggestedEnergy;
//			s[j + replica_shift] = suggestedSpin;
//		}
//	}
//}
//
//__global__ void calcMagnetization(char* s, int* E, int L, int N, int R, int U, int* deviceMagnetization) {//, int* acceptance_number) {
//
//	int r = threadIdx.x + blockIdx.x * blockDim.x;
//	//printf("kernel %d reporting: E = %i, U = %i\n", r, E[r], U);
//	int replica_shift = r * N;
//	if (E[r] == U) {
//		int m = 0;
//		int c = N;
//
//
//		for (int j = 0; j < N; j++) {
//			char i = s[j + replica_shift]; // current spin value
//			m += i;
//			c -= i * i;
//		}
//		//printf("kernel %d reporting success: E = %i, U = %i; m=%i, m_sq=%i\n", r, E[r], U, m, m_sq);
//
//		deviceMagnetization[r * STATISTICS_NUMBER + 0] = 1; // number of replicas with E = U
//		deviceMagnetization[r * STATISTICS_NUMBER + 1] = m; // | magnetization |
//		deviceMagnetization[r * STATISTICS_NUMBER + 2] = c; // | concentration |
//
//		struct energy_parts sum = calcEnergyParts(s, L, N, r);
//		deviceMagnetization[r * STATISTICS_NUMBER + 3] = sum.Ising / 2; // | ising energy part |
//		deviceMagnetization[r * STATISTICS_NUMBER + 4] = sum.Blume; // | blume energy part |
//	}
//}
//
//
//void CalcPrintAvgE(FILE* efile, int* E, int R, int U, int D_base) {
//	float avg = 0.0;
//	for (int i = 0; i < R; i++) {
//		avg += E[i];
//	}
//	avg /= R;
//	fprintf(efile, "%f %f\n", 1.0 * U / D_base, avg);
//	printf("E: %f\n", avg);
//}
//
//void CalculateRhoT(const int* replicaFamily, FILE* ptfile, int R, int U, int D_base) {
//	// histogram of family sizes
//	int* famHist = (int*)calloc(R, sizeof(int));
//	for (int i = 0; i < R; i++) {
//		famHist[replicaFamily[i]]++;
//	}
//	double sum = 0;
//	for (int i = 0; i < R; i++) {
//		sum += famHist[i] * famHist[i];
//	}
//	sum /= R;
//	fprintf(ptfile, "%f %f\n", 1.0 * U / D_base, sum);
//	sum /= R;
//	printf("RhoT:\t%f\n", sum);
//	free(famHist);
//}
//
//
//void Swap(int* A, int i, int j) {
//	int temp = A[i];
//	A[i] = A[j];
//	A[j] = temp;
//}
//
//void quicksort(int* E, int* O, int left, int right, int direction) {
//	int Min = (left + right) / 2;
//	int i = left;
//	int j = right;
//	double pivot = direction * E[O[Min]];
//
//	while (left < j || i < right)
//	{
//		while (direction * E[O[i]] > pivot)
//			i++;
//		while (direction * E[O[j]] < pivot)
//			j--;
//
//		if (i <= j) {
//			Swap(O, i, j);
//			i++;
//			j--;
//		}
//		else {
//			if (left < j)
//				quicksort(E, O, left, j, direction);
//			if (i < right)
//				quicksort(E, O, i, right, direction);
//			return;
//		}
//	}
//}
//
//int resample(int* E, int* O, int* update, int* replicaFamily, int R, int* U, int D_base, FILE* Xfile, bool heat) {
//	//std::sort(O, O + R, [&E](int a, int b) {return E[a] > E[b]; }); // greater sign for descending order
//	quicksort(E, O, 0, R - 1, 1 - 2 * heat); //Sorts O by energy
//
//	int nCull = 0;
//	//fprintf(e2file, "%f %i\n", 1.0 * (*U) / D_base, E[O[0]]);
//
//	//update energy seiling to the highest available energy
//	int U_old = *U;
//	int U_new;
//
//	for (int i = 0; i < R; i++) {
//		U_new = E[O[i]];
//		if ((!heat && U_new < U_old - EPSILON) || (heat && U_new > U_old + EPSILON)) {
//			*U = U_new;
//			break;
//		}
//	}
//
//	if (fabs(*U - U_old) <= EPSILON) {
//		return 1; // out of replicas
//	}
//
//	while ((!heat && E[O[nCull]] >= *U - EPSILON) || (heat && E[O[nCull]] <= *U + EPSILON)) {
//		nCull++;
//		if (nCull == R) {
//			break;
//		}
//	}
//	// culling fraction
//	double X = nCull;
//	X /= R;
//	//printf("U: %d %d %d\n", U_new, U_old, *U);
//	fprintf(Xfile, "%f %f\n", 1.0 * (*U) / D_base, X);
//	fflush(Xfile);
//	printf("Culling fraction:\t%f\n", X);
//	fflush(stdout);
//	for (int i = 0; i < R; i++)
//		update[i] = i;
//	if (nCull < R) {
//		for (int i = 0; i < nCull; i++) {
//			// random selection of unculled replica
//			int r = (rand() % (R - nCull)) + nCull; // different random number generator for resampling
//			update[O[i]] = O[r];
//			replicaFamily[O[i]] = replicaFamily[O[r]];
//		}
//	}
//
//	return 0;
//}
//
//void PrintMagnetization(int U, int D_base, FILE* mfile, int R, int N, int* hostMagnetization) {
//	int culled_replica_number = 0;
//	double m, c, e_ising, e_blume;
//	double n[STATISTICS_NUMBER_2];
//	for (int j = 0; j < STATISTICS_NUMBER_2; j++) {
//		n[j] = 0.0;
//	}
//
//	for (int i = 0; i < R; i++) {
//		culled_replica_number += hostMagnetization[i * STATISTICS_NUMBER + 0];
//		m = double(hostMagnetization[i * STATISTICS_NUMBER + 1]);
//		c = double(hostMagnetization[i * STATISTICS_NUMBER + 2]);
//
//		n[0] += abs(m / N);
//		n[1] += (m / N) * (m / N);
//		n[2] += (m / N) * (m / N) * abs(m / N);
//		n[3] += (m / N) * (m / N) * (m / N) * (m / N);
//		n[4] += abs(c / N);
//		n[5] += (c / N) * (c / N);
//		n[6] += (c / N) * (c / N) * abs(c / N);
//		n[7] += (c / N) * (c / N) * (c / N) * (c / N);
//
//		e_ising = double(hostMagnetization[i * STATISTICS_NUMBER + 3]);
//		e_blume = double(hostMagnetization[i * STATISTICS_NUMBER + 4]);
//
//		n[8] += e_ising / N;
//		n[9] += e_blume / N;
//	}
//
//
//	fprintf(mfile, "%f %i ", 1.0 * U / D_base, culled_replica_number);
//	for (int j = 0; j < STATISTICS_NUMBER_2; j++) {
//		fprintf(mfile, "%f ", n[j] / culled_replica_number);
//	}
//	fprintf(mfile, "\n");
//	fflush(mfile);
//	fflush(mfile);
//}
//
//__global__ void updateReplicas(char* s, int* E, int* update, int N) {
//	/*---------------------------------------------------------------------------------------------
//		Updates the population after the resampling step (done on cpu) by replacing indicated
//		replicas by the proper other replica
//	-----------------------------------------------------------------------------------------------*/
//	int r = threadIdx.x + blockIdx.x * blockDim.x;
//	int replica_shift = r * N;
//	int source_r = update[r];
//	int source_replica_shift = source_r * N;
//	if (source_r != r) {
//		for (int j = 0; j < N; j++) {
//			s[j + replica_shift] = s[j + source_replica_shift];
//		}
//		E[r] = E[update[r]];
//	}
//}
//
//__global__ void setup_kernel(curandStatePhilox4_32_10_t* state, int seed)
//{
//	int id = threadIdx.x + blockIdx.x * blockDim.x;
//	/* Each thread gets same seed, a different sequence
//	   number, no offset */
//	curand_init(seed, id, 0, state + id);
//}
//







// All other function implementations go here
//}


