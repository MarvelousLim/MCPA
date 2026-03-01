//#pragma once
//
//
//#define NNEIBORS 4 // number of nearest neighbors, is 4 for 2d lattice
//#define STATISTICS_NUMBER 5		// primary statistics to calc for each replica
//#define STATISTICS_NUMBER_2 10	// secondary statistics calced from primary
//#define EPSILON 0
//
//
//int main(int argc, char* argv[]) {
//
//	clock_t start, end;
//	double cpu_time_used;
//
//	start = clock();
//
//	// Parameters:
//
//	int run_number = atoi(argv[1]);	// A number to label this run of the algorithm, used for data keeping purposes, also, a seed
//	int seed = run_number;
//	//int grid_width = atoi(argv[2]);	// should not be more than 256 due to MTGP32 limits
//	int L = atoi(argv[2]);	// Lattice size
//	int N = L * L;
//	//int R = grid_width * BLOCKS;	// Population size
//	int BLOCKS = atoi(argv[3]);
//	int THREADS = atoi(argv[4]);
//	int nSteps = atoi(argv[5]);
//
//	int R = BLOCKS * THREADS;
//
//
//	//Blume-Capel model parameter
//	int D_div = atoi(argv[6]);
//	int D_base = atoi(argv[7]);
//	float D = (float)D_div / D_base;
//	bool heat = atoi(argv[8]); // 0 if cooling (default) and 1 if heating
//
//
//	// initializing files to write in
//	const char* heating = heat ? "Heating" : "";
//
//	printf("running 2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de.txt\n", heating, q, D, N, R, nSteps, run_number);
//
//	char s[100];
//	const char* prefix = "test";// "2DBlume";
//	/*
//	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de.txt", prefix, heating, q, D, N, R, nSteps, run_number);
//	FILE* efile = fopen(s, "w");	// average energy
//	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de2.txt", prefix, heating, q, D, N, R, nSteps, run_number);
//	FILE* e2file = fopen(s, "w");	// surface (culled) energy
//	*/
//	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dX.txt", prefix, heating, q, D, N, R, nSteps, run_number);
//	FILE* Xfile = fopen(s, "w");	// culling fraction
//	/*
//	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dpt.txt", prefix, heating, q, D, N, R, nSteps, run_number);
//	FILE* ptfile = fopen(s, "w");	// rho t
//	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dn.txt", prefix, heating, q, D, N, R, nSteps, run_number);
//	FILE* nfile = fopen(s, "w");	// number of sweeps
//	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dch.txt", prefix, heating, q, D, N, R, nSteps, run_number);
//	FILE* chfile = fopen(s, "w");	// cluster size histogram
//	*/
//	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dm.txt", prefix, heating, q, D, N, R, nSteps, run_number);
//	FILE* mfile = fopen(s, "w");	// magnetization in format m = sum(sigma), m^2=sum(sigma^2), N
//	/*
//	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%de3.txt", heating, q, D, N, R, nSteps, run_number);
//	FILE* e3file = fopen(s, "w");
//	*/
//	/*
//	sprintf(s, "datasets//hysteresis//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%ds.txt", heating, q, D, N, R, nSteps, run_number);
//	FILE* sfile = fopen(s, "w");	// spin system sample
//	*/
//
//	size_t fullLatticeByteSize = R * N * sizeof(char);
//
//	// Allocate space on host
//	//char* hostSpin = (char*)malloc(fullLatticeByteSize);
//	int* hostE = (int*)malloc(R * sizeof(int));
//	int* hostUpdate = (int*)malloc(R * sizeof(int));
//	int* replicaFamily = (int*)malloc(R * sizeof(int));
//	int* energyOrder = (int*)malloc(R * sizeof(int));
//	for (int i = 0; i < R; i++) {
//		energyOrder[i] = i;
//		replicaFamily[i] = i;
//	}
//	int MagnetizationArraySize = R * STATISTICS_NUMBER;
//	int* hostMagnetization = (int*)malloc(MagnetizationArraySize * sizeof(int));
//
//	// Allocate memory on device
//	char* deviceSpin; // s, d_s
//	int* deviceE;
//	int* deviceUpdate;
//	int* deviceMagnetization;
//	gpuErrchk(cudaMalloc((void**)&deviceSpin, fullLatticeByteSize));
//	gpuErrchk(cudaMalloc((void**)&deviceE, R * sizeof(int)));
//	gpuErrchk(cudaMalloc((void**)&deviceUpdate, R * sizeof(int)));
//	gpuErrchk(cudaMalloc((void**)&deviceMagnetization, MagnetizationArraySize * sizeof(int)));
//
//	// Allocate memory for histogram calculation
//	/*
//	int* hostClusterSizeArray = (int*)malloc(N * sizeof(int));
//	bool* deviceVisited;
//	int* deviceClusterSizeArray;
//	int* deviceStack;
//	gpuErrchk( cudaMalloc((void**)&deviceVisited, N * R * sizeof(bool)) );
//	gpuErrchk( cudaMalloc((void**)&deviceClusterSizeArray, N * sizeof(int)) );
//	gpuErrchk( cudaMalloc((void**)&deviceStack, N * R * sizeof(int)) );
//	*/
//
//	// Init Philox
//	curandStatePhilox4_32_10_t* devStates;
//	gpuErrchk(cudaMalloc((void**)&devStates, R * sizeof(curandStatePhilox4_32_10_t)));
//	setup_kernel << < BLOCKS, THREADS >> > (devStates, seed);
//	gpuErrchk(cudaPeekAtLastError());
//	gpuErrchk(cudaDeviceSynchronize());
//
//	// Init std random generator for little host part
//	srand(seed);
//
//	// Actually working part
//	initializePopulation << < BLOCKS, THREADS >> > (devStates, deviceSpin, N, q);
//	gpuErrchk(cudaPeekAtLastError());
//	gpuErrchk(cudaDeviceSynchronize());
//	cudaMemset(deviceE, 0, R * sizeof(int));
//
//	//init testing values
//	/*
//	deviceEnergy <<< BLOCKS, THREADS >>> (deviceSpin, deviceE, L, N, D);
//	gpuErrchk(cudaPeekAtLastError());
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk( cudaMemcpy(hostE, deviceE, R * sizeof(int), cudaMemcpyDeviceToHost) );
//
//	char* hostSpin = (char*)malloc(N * sizeof(char)); // test shit
//	gpuErrchk(cudaMemcpy(hostSpin, deviceSpin, N * sizeof(char), cudaMemcpyDeviceToHost)); // take one replica (first)
//	for (int i = 0; i < L; i++) {
//		for (int j = 0; j < L; j++) {
//			printf("%i ", hostSpin[i * L + j]);
//		}
//		printf("\n");
//	}
//
//
//	int host_acceptance_number = 0;
//	int* device_acceptance_number;
//	gpuErrchk(cudaMalloc((void**)&device_acceptance_number, sizeof(int)));
//	*/
//
//
//	int upper_energy = N * abs(D_div) + 2 * N * D_base;
//	int lower_energy = -N * abs(D_div) - 2 * N * D_base;
//
//	int U = (heat ? lower_energy : upper_energy);	// U is energy ceiling
//
//	//CalcPrintAvgE(efile, hostE, R, U);
//
//	while ((U >= lower_energy && !heat) || (U <= upper_energy && heat)) {
//
//		//fprintf(nfile, "%d %d\n", U, nSteps);
//		printf("U:\t%f out of %d; nSteps: %d;\n", 1.0 * U / D_base, -2 * N, nSteps);
//
//		equilibrate << < BLOCKS, THREADS >> > (devStates, deviceSpin, deviceE, L, N, R, q, nSteps, U, D_div, D_base, heat);// , device_acceptance_number);
//		gpuErrchk(cudaPeekAtLastError());
//		gpuErrchk(cudaDeviceSynchronize());
//		gpuErrchk(cudaMemcpy(hostE, deviceE, R * sizeof(int), cudaMemcpyDeviceToHost));
//
//		// record average energy and rho t
//		//CalcPrintAvgE(efile, hostE, R, U, D_base);
//		//CalculateRhoT(replicaFamily, ptfile, R, U, D_base);
//		// perform resampling step on cpu
//		// also lowers energy seiling U
//
//		int error = resample(hostE, energyOrder, hostUpdate, replicaFamily, R, &U, D_base, Xfile, heat);
//
//
//		cudaMemset(deviceMagnetization, 0, MagnetizationArraySize * sizeof(int));
//		calcMagnetization << < BLOCKS, THREADS >> > (deviceSpin, deviceE, L, N, R, U, deviceMagnetization);
//		gpuErrchk(cudaPeekAtLastError());
//		gpuErrchk(cudaDeviceSynchronize());
//		gpuErrchk(cudaMemcpy(hostMagnetization, deviceMagnetization, MagnetizationArraySize * sizeof(int), cudaMemcpyDeviceToHost));
//		PrintMagnetization(U, D_base, mfile, R, N, hostMagnetization);
//
//
//		if (error)
//		{
//			printf("Process ended with zero replicas\n");
//			break;
//		}
//		// copy list of replicas to update back to gpu
//		gpuErrchk(cudaMemcpy(deviceUpdate, hostUpdate, R * sizeof(int), cudaMemcpyHostToDevice));
//		updateReplicas << < BLOCKS, THREADS >> > (deviceSpin, deviceE, deviceUpdate, N);
//		gpuErrchk(cudaPeekAtLastError());
//		gpuErrchk(cudaDeviceSynchronize());
//		printf("\n");
//
//
//	}
//
//
//
//	// Free memory and close files
//
//	cudaFree(deviceSpin);
//	cudaFree(deviceE);
//	cudaFree(deviceUpdate);
//	cudaFree(deviceMagnetization);
//	//cudaFree(deviceClusterSizeArray);
//	//cudaFree(deviceStack);
//	//cudaFree(deviceVisited);
//	//cudaFree(device_acceptance_number);
//
//	free(hostMagnetization);
//	//free(hostSpin);
//	free(hostE);
//	free(hostUpdate);
//	free(replicaFamily);
//	free(energyOrder);
//	//free(hostClusterSizeArray);
//
//	//fclose(efile);
//	//fclose(e2file);
//	fclose(Xfile);
//	//fclose(ptfile);
//	//fclose(nfile);
//	//fclose(chfile);
//	fclose(mfile);
//
//	//fclose(e3file);
//
//	end = clock();
//	cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//
//	sprintf(s, "datasets//%s//2DBlume%s_q%d_D%f_N%d_R%d_nSteps%d_run%dtime.txt", prefix, heating, q, D, N, R, nSteps, run_number);
//	FILE* timefile = fopen(s, "w");
//
//	fprintf(timefile, "%f seconds", cpu_time_used);
//
//	fclose(timefile);
//
//	// End
//	return 0;
//}
