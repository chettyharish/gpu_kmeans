
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <math.h>
#include <time.h>

#define iterations 10						// Max times to run kmeans
#define convergecount 20					// Max kmeans step to avoid flip-flops
#define clusterDimension 16
#define numPoints (1*1024*1024)
#define numClusters (1024)
#define FLOAT_MAX 1e+37
#define ConstantMemFloats (64*1024)/4			//	64KB/4
#define rand_range 100


#define PC 1
#if PC == 0
/*Works on Windows!*/
double microtime() { return (double)time(NULL); }
#else
/*Woks on Linux*/
#include <sys/time.h>
double microtime(void)
{
	struct timeval t;
	gettimeofday(&t, 0);
	return 1.0e6*t.tv_sec + (double)t.tv_usec;
}
#endif



__constant__ float d_cons_centers[ConstantMemFloats];

__host__ void printDeviceInfo(){
	FILE * fp;
	fp = fopen("specifications.txt", "w");
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		fprintf(fp, "Name = %s\n", prop.name);
		fprintf(fp, "Number of Multi-processors = %d\n", (int)prop.multiProcessorCount);
		fprintf(fp, "Max threads per Block = %d\n", (int)prop.maxThreadsPerBlock);
		fprintf(fp, "Max threads per SM = %d\n", (int)prop.maxThreadsPerMultiProcessor);

		fprintf(fp, "Global Memory = %u B\n", (int)prop.totalGlobalMem);
		fprintf(fp, "L2 SIZE = %d bytes\n", (int)prop.l2CacheSize);
		fprintf(fp, "Shared memory per SM = %d B\n", (int)prop.sharedMemPerBlock);
		fprintf(fp, "Total Constant Memory = %d B\n", (int)prop.totalConstMem);

		fprintf(fp, "Registers per SM = %d\n", (int)prop.regsPerBlock);
		fprintf(fp, "Average Registers per Thread = %d\n", (int)prop.regsPerBlock / (int)prop.maxThreadsPerMultiProcessor);
		fprintf(fp, "Warp size = %d threads\n", (int)prop.warpSize);


		fprintf(fp, "Version = %d.%d\n", (int)prop.major, (int)prop.minor);
	}

	fclose(fp);
}


__host__ void generate_random_points_transpose(float *h_points){
	/*Avoiding rand() since its not truly random*/
	//Randomly generating points using rand()
	//srand((unsigned int)time(0));

	int count = 0;
	for (int j = 0; j < clusterDimension; j++)
	{
		for (int i = 0; i < numPoints; i++){
			h_points[j*numPoints + i] =  (float)(rand() % rand_range);
			h_points[j*numPoints + i] = (float)(count++);
		}
	}


	/*int flagger = 0;
	int sameCount = 0;
	for (int k = 0; k < numClusters; k++){
		for (int i = 0; i < numClusters; i++){
			flagger = 0;
			for (int j = 0; j < clusterDimension; j++){
				if (!(h_points[j*numPoints + k] == h_points[j*numPoints + i])){
					flagger = 1;
				}
			}

			if (k == i)
				flagger = 1;

			if (flagger == 0){
				sameCount++;
			}
		}
	}

	printf("SAME POINTS = %d\n", sameCount);*/

}

__host__ void generate_random_centers_transpose(float *h_points, float *h_centers_old, float *h_centers_new){
	//Selecting random points using Floyd's Algorithm
	int *rand_idx = (int *)malloc(numClusters*sizeof(int));
	int *flag = (int *)malloc(numPoints*sizeof(int));

	memset(rand_idx, 0, numClusters*sizeof(int));
	memset(flag, 0, numPoints*sizeof(int));

	int n = numPoints;
	int m = 0;
	for (n = (numPoints - numClusters); n < numPoints && m < numClusters; n++, m++){
		int r = rand() % (n + 1);

		if (flag[r] == 1){
			/*Works since previous iteration had rand() % n , and thus
			it is not possible that n was chosen!*/
			r = n;
		}
		rand_idx[m] = r;
		flag[r] = 1;
	}


	for (int i = 0; i < numClusters; i++){
		for (int j = 0; j < clusterDimension; j++)
		{
			h_centers_old[i*clusterDimension + j] = h_points[j*numPoints + rand_idx[i]];
			h_centers_new[i*clusterDimension + j] = h_points[j*numPoints + rand_idx[i]];
		}
	}

	/*int flagger = 0;
	int sameCount = 0;

	for (int k = 0; k < numClusters; k++){
		for (int i = 0; i < numClusters; i++){
			flagger = 0;
			for (int j = 0; j < clusterDimension; j++){
				if (!(h_centers_new[k*clusterDimension + j] == h_centers_new[i*clusterDimension + j])){
					flagger = 1;
				}
			}

			if (k == i)
				flagger = 1;

			if (flagger == 0){
				sameCount++;
			}
		}
	}


	printf("SAME CENTERS  = %d\n", sameCount);*/


}



__host__ float calculate_norm(float *h_centers_old, float *h_centers_new){
	float diff_norm = 0.0f;
	for (int i = 0; i < numClusters; i++){
		for (int j = 0; j < clusterDimension; j++){
			diff_norm += fabsf(h_centers_old[i*clusterDimension + j] - h_centers_new[i*clusterDimension + j]);
		}
	}

	return diff_norm;
}

__host__ void member_division(float *h_centers_new, int *h_memberCounter){
	for (int i = 0; i < numClusters; i++){
		for (int j = 0; j < clusterDimension; j++){
			/*New center is same as old if no points in the cluster*/
			if (h_memberCounter[i] != 0)
				h_centers_new[i*clusterDimension + j] /= h_memberCounter[i];
		}
	}
}

__host__ void copy_centers(float * h_centers_old, float* h_centers_new){
	for (int i = 0; i < numClusters; i++){
		for (int j = 0; j < clusterDimension; j++){
			h_centers_old[i*clusterDimension + j] = h_centers_new[i*clusterDimension + j];

		}
	}
}


/*
A modified kernel which tries to take advantage of both the orders
Uses column ordering for points and row ordering for centers
*/

__global__ void calc_distance_mixed(float *d_points, int *d_clusterIdx, float *d_mindistances, int step, int num_copy, int max_cached){

	int i = blockDim.x*blockIdx.x + threadIdx.x;

	/*Getting the value from previous iterations*/
	if (i < numPoints){ 
		int min_pos = -1;
		float points[clusterDimension];
		for (int j = 0; j < clusterDimension; j++){
			points[j] = d_points[j*numPoints + i];
		}
		float min_dist = d_mindistances[i];
		float old_min_dist = d_mindistances[i];

		for (int k = 0; k < num_copy; k++){
			float distance = 0.0f;
			for (int j = 0; j < clusterDimension; j++){
				distance += fabsf(points[j] - d_cons_centers[k*clusterDimension + j]);
			}

			if (distance < min_dist){
				min_dist = distance;
				min_pos = k;
			}
		}

		
		/*Only update if there were changes!!*/
		if (min_dist < old_min_dist){
			d_mindistances[i] = min_dist;
			d_clusterIdx[i] = step*max_cached + min_pos;
		}
	}
}

__global__ void generate_new_center_transpose(float *d_points, float *d_centers, int *d_clusterIdx, int * d_member_counter){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < numPoints){
		int clusterId = d_clusterIdx[i];
		for (int j = 0; j < clusterDimension; j++){
			atomicAdd(&d_centers[clusterDimension*clusterId + j], d_points[j*numPoints + i]);
		}
		atomicAdd(&d_member_counter[clusterId], 1);
	}

}

__global__ void add_array(float *d_mindistances)
{
	const int threads_per_block = 1024;
	__shared__ float partialSum[2 * threads_per_block];
	int idx = threadIdx.x;
	int start = 2 * blockIdx.x * threads_per_block;
	if (start < numPoints){
		partialSum[idx] = d_mindistances[start + idx];
		if (start + threads_per_block < numPoints)
			partialSum[idx + threads_per_block] = d_mindistances[start + idx + threads_per_block];
		else
			partialSum[idx + threads_per_block] = 0;

		for (unsigned int stride = threads_per_block; stride >= 1; stride >>= 1)
		{
			__syncthreads();
			if (idx < stride)
				partialSum[idx] += partialSum[idx + stride];
		}

		d_mindistances[start + idx] = partialSum[idx];
	}
}

int main(int argc, char **argv){

	printDeviceInfo();
	double clk1, clk2, mclk1, mclk2, kmeansclk1, kmeansclk2;
	mclk1 = microtime();

	/*For maintaining multiple iterations of kmeans*/
	/*Usese RSS(Residual sum of squares) to determine best iteration*/
	/*RSS_score = sum((di)^2) for all di , where di is distance from cluster center*/
	double RSS_score = 0.0;
	double minRSS_score = (float)FLOAT_MAX;
	int *h_min_RSS_clusterIdx = (int *)malloc(numPoints*sizeof(int));

	/*For carrying out one iteration of Kmeans*/
	float *d_points, *d_centers, *d_mindistances;
	int *d_clusterIdx, *d_member_counter;
	float diff_norm = (float)FLOAT_MAX;

	//Calculations for calc_distance()
	/*Block fitting*/
	int ThreadsPerBlock = 1024;
	if ((48 * 1024) / (clusterDimension*sizeof(float)) < 1024)
		ThreadsPerBlock = (48 * 1024) / (clusterDimension*sizeof(float));
	int NumBlocks = (int)ceil(numPoints*1.0f / ThreadsPerBlock);

	/*Shared memory max fitting*/
	int smem_size = ThreadsPerBlock*clusterDimension*sizeof(float);

	/*Constant memory max fitting*/
	int max_cached, distance_steps;
	if (numClusters*clusterDimension * 4.0 <= 64 * 1024 * 1.0f){
		/*I can fit all the centers!*/
		max_cached = numClusters;
	}
	else{
		/*One Point takes clusterDimension*4 memory , how mant can I fit in 64k?*/
		max_cached = (int)floor(64 * 1024 * 1.0f / (clusterDimension * 4));
	}

	distance_steps = (int)ceil(numClusters*1.0f / max_cached*1.0f);
	printf("Calculations for max_distance() function!\n");
	printf("numClusters = %d \tThreadsPerBlock = %d \t NumBlocks = %d\t smem_size = %d\n", numClusters, ThreadsPerBlock, NumBlocks, smem_size);
	printf("max_cached %d\t distance_steps %d\n\n", max_cached, distance_steps);

	//Calculations for generate_new_centers()
	int split_size = (int)floor((48 * 1024) / (numClusters * 4.0f));
	int split_steps = (int)ceil(clusterDimension *1.0f / split_size);
	printf("Calculations for generate_new_centers() function!\n");
	printf("split_size = %d \t split_steps = %d \n\n", split_size, split_steps);

	/*Stores the points*/
	float *h_points = (float *)malloc(clusterDimension*numPoints*sizeof(float));

	/*Need two arrays one for old centers, and one for new  for calculating NORM*/
	float *h_centers_old = (float *)malloc(clusterDimension*numClusters*sizeof(float));
	float *h_centers_new = (float *)malloc(clusterDimension*numClusters*sizeof(float));
	float *h_centers_transpose = (float *)malloc(clusterDimension*max_cached*sizeof(float));

	/*Stores cluster indexes of all the points*/
	float *h_mindistances = (float *)malloc(numPoints*sizeof(float));
	int *h_clusterIdx = (int *)malloc(numPoints*sizeof(int));

	/*Consists number of members in a clusters*/
	int *h_member_counter = (int *)malloc(numClusters*sizeof(int));

	cudaMalloc((void**)& d_points, clusterDimension*numPoints*sizeof(float));
	cudaMalloc((void**)& d_centers, clusterDimension*numClusters*sizeof(float));
	cudaMalloc((void**)& d_mindistances, numPoints*sizeof(float));
	cudaMalloc((void **)&d_clusterIdx, numPoints*sizeof(int));
	cudaMalloc((void **)&d_member_counter, numClusters*sizeof(int));
	generate_random_points_transpose(h_points);


	cudaMemcpy(d_points, h_points, clusterDimension*numPoints*sizeof(float), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	int count = 0;
	int con_count = 0;
	while (count < iterations){

		generate_random_centers_transpose(h_points, h_centers_old, h_centers_new);
		/*Each co-ordinate has a change less than 0.001 on average!*/
		kmeansclk1 = microtime();
		while (diff_norm >(numPoints*clusterDimension) / 1000.0 && con_count < convergecount ){
			printf("\nIteration = %d \t Ccount = %d\n", count, con_count);
			clk1 = microtime();
			cudaMemset(d_clusterIdx, 1, numPoints*sizeof(int));
			cudaMemset(d_mindistances, 99, numPoints*sizeof(float));
			cudaMemset(d_member_counter, 0, numClusters*sizeof(int));
			/*
			cudaMemcpy(d_clusterIdx, h_clusterIdx, numPoints*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_mindistances, h_mindistances, numPoints*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_member_counter, h_member_counter, numClusters*sizeof(int), cudaMemcpyHostToDevice);*/
			cudaDeviceSynchronize();
			clk2 = microtime();
			printf("PART 1 :Time = %g µs\n", (double)(clk2 - clk1));

			clk1 = microtime();
			for (int step = 0; step < distance_steps; step++){
				//To adjust for last iteration!
				int num_copy = (max_cached <= numClusters - step*max_cached) ? max_cached : numClusters - step*max_cached;
				cudaMemcpyToSymbol(d_cons_centers, h_centers_new + step*max_cached*clusterDimension, clusterDimension*num_copy*sizeof(float));
				cudaDeviceSynchronize();
				calc_distance_mixed << <(int)ceil(numPoints / 1024.0), 1024 >> >(d_points, d_clusterIdx,
					d_mindistances, step, num_copy, max_cached);
				cudaDeviceSynchronize();


				//printf("num_cpy = %d\tcalc_distance_mixed: %s\n", num_copy, cudaGetErrorString(cudaGetLastError()));
			}
			clk2 = microtime();
			printf("PART 2 :Time = %g µs\n", (double)(clk2 - clk1));

			clk1 = microtime();
			cudaMemset(d_centers, 0, clusterDimension*numClusters*sizeof(float));
			generate_new_center_transpose << <(int)ceil(numPoints / 1024.0), 1024 >> >(d_points, d_centers, d_clusterIdx, d_member_counter);
			cudaDeviceSynchronize();
			//printf("generate_new_center_transpose: %s\n", cudaGetErrorString(cudaGetLastError()));
			clk2 = microtime();
			printf("PART 3 :Time = %g µs\n", (double)(clk2 - clk1));

			clk1 = microtime();
			cudaMemcpy(h_centers_new, d_centers, clusterDimension*numClusters*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_member_counter, d_member_counter, numClusters*sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			clk2 = microtime();
			printf("PART 4 :Time = %g µs\n", (double)(clk2 - clk1));

			clk1 = microtime();
			/*
			int zero_count = 0;
			for (int i = 0; i < numClusters; i++){
				if (h_member_counter[i] == 0)
					zero_count++;
			}
			printf("Zero Count = %d\n", zero_count);*/
			member_division(h_centers_new, h_member_counter);
			diff_norm = calculate_norm(h_centers_old, h_centers_new);
			copy_centers(h_centers_old, h_centers_new);
			con_count++;
			clk2 = microtime();
			printf("PART 5 :Time = %g µs\n", (double)(clk2 - clk1));
			printf("Diff Norm = %f\n", diff_norm);
		}
		kmeansclk2 = microtime();
		printf("Kmeans Total Time = %g seconds\n\n", (double)((kmeansclk2 - kmeansclk1) / 1000000));

#if numPoints > 64*1024*1024
		add_array << <(int)ceil(numPoints / 1024.0), 1024 >> >(d_mindistances);
		cudaMemcpy(h_mindistances, d_mindistances, numPoints*sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		for (int i = 0; i < (int)ceil(numPoints / 1024.0); i++){
			RSS_score += h_mindistances[i * 2 * 1024];
		}
#else
		cudaMemcpy(h_mindistances, d_mindistances, numPoints*sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		for (int i = 0; i < numPoints; i++){
			RSS_score += h_mindistances[i];
		}
#endif
		printf("Iteration = %d\t RSS_score is : %f\n", count, RSS_score);


		if (RSS_score < minRSS_score){
			/*Storing back the clusters which exhibit the lowest RSS_score*/
			minRSS_score = RSS_score;
			cudaMemcpy(h_min_RSS_clusterIdx, d_clusterIdx, numPoints*sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

		}
		RSS_score = 0.0;
		con_count = 0;
		diff_norm = (float)FLOAT_MAX;
		count++;

	}


	FILE * fp;
	fp = fopen("clusters.txt", "w");
	fprintf(fp, "RSS_score = %g\n", minRSS_score);
	for (int i = 0; i < numPoints; i++){
		fprintf(fp, "%d\t%d\n", i, h_min_RSS_clusterIdx[i]);
	}
	fclose(fp);
	printf("Done\n");
	cudaFree(d_member_counter);
	cudaFree(d_clusterIdx);
	cudaFree(d_points);
	cudaFree(d_centers);
	free(h_points);
	free(h_centers_old);
	free(h_centers_new);
	free(h_clusterIdx);
	free(h_member_counter);
	mclk2 = microtime();

	printf("Total Time = %g seconds\n", (double)((mclk2 - mclk1) / 1000000));
	return 0;
}
