#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <math.h>
#include <time.h>

#define FLOAT_MAX 1e+37
#define numPoints 1024*1024
#define clusterDimension 16
#define numClusters (64*1024)/(4*clusterDimension) //Must fit in const cache!
#define ConstantMemFloats (64*1024)/4			//	64KB/4
#define rand_range 100
#define PC 1
#if PC == 0
	double microtime() { return 0.0; }
#else
	#include <sys/time.h>
	double microtime(void)
	{
		struct timeval t;
		gettimeofday(&t, 0);
		return 1.0e6*t.tv_sec + (double)t.tv_usec;
	}
#endif



__constant__ float d_cons_centers[ConstantMemFloats];

__host__ void generate_random_data(float *h_points, float *h_centers_old, float *h_centers_new){

	//Randomly generating points using rand()
	srand((unsigned int)time(0));
	for (int i = 0; i < numPoints; i++){
		for (int j = 0; j < clusterDimension; j++)
		{
			h_points[i*clusterDimension + j] = (float)(rand() % rand_range);
		}
	}

	//Selecting the first numClusters points as the starting centers
	int k = 0;
	for (int i = 0; i < numClusters; i++, k++){
		for (int j = 0; j < clusterDimension; j++)
		{
			h_centers_old[i*clusterDimension + j] = h_points[k*clusterDimension + j];
			h_centers_new[i*clusterDimension + j] = h_points[k*clusterDimension + j];
		}
	}

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



//Costs 500 µ seconds! do not use!
__device__ float distance_func(float *point1, float *point2){
	float distance = 0.0f;
	for (int k = 0; k < clusterDimension; k++){
		distance += sqrtf((point1[k] - point2[k]) * (point1[k] - point2[k]));
	}
	return distance;

}

__global__ void calc_distance(float *d_points, int *d_clusterIdx){

	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int tx = threadIdx.x;
	int min_pos = -1;
	float min_dist = FLOAT_MAX;
	extern __shared__ float s_points[];
	
	if (i < numPoints){
		for (int j = 0; j < clusterDimension; j++){
			s_points[tx*clusterDimension + j] = d_points[i*clusterDimension + j];
		}
		
		for (int k = 0; k < numClusters; k++){float distance = 0.0f;
			for (int j = 0; j < clusterDimension; j++){
				distance += fabsf(s_points[tx*clusterDimension + j] - d_cons_centers[k*clusterDimension + j]);
			}

			if (distance < min_dist){
				min_dist = distance;
				min_pos = k;
			}

		}
		
		d_clusterIdx[i] = min_pos;
	}
}

__global__ void generate_new_center(float *d_points, float *d_centers, int *d_clusterIdx, int * d_member_counter){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < numPoints){
		int clusterId = d_clusterIdx[i];
		for (int j = 0; j < clusterDimension; j++){
			atomicAdd(&d_centers[clusterDimension*clusterId + j], d_points[i*clusterDimension + j]);
		}
		atomicAdd(&d_member_counter[clusterId], 1);
	}

}

int main(int argc, char **argv){
	double clk1, clk2, mclk1, mclk2;
	mclk1 = microtime();


	float *d_points , *d_centers;
	int *d_clusterIdx, *d_member_counter;
	float diff_norm = rand_range + 1;
	
	int ThreadsPerBlock = (48 * 1024) / (clusterDimension*sizeof(float))/2;
	int NumBlocks = (int)ceil(numPoints*1.0f / ThreadsPerBlock);

	int smem_size = ThreadsPerBlock*clusterDimension*sizeof(float);

	printf("numClusters = %d \tThreadsPerBlock = %d \t NumBlocks = %d\t smem_size = %d\n",numClusters, ThreadsPerBlock, NumBlocks, smem_size);
	/*Stores the points*/
	float *h_points = (float *)malloc(clusterDimension*numPoints*sizeof(float));

	/*Need two arrays one for old centers, and one for new  for calculating NORM*/
	float *h_centers_old = (float *)malloc(clusterDimension*numClusters*sizeof(float));
	float *h_centers_new = (float *)malloc(clusterDimension*numClusters*sizeof(float));

	/*Stores cluster indexes of all the points*/
	int *h_clusterIdx = (int *)malloc(numPoints*sizeof(int));

	/*Consists number of members in a clusters*/
	int *h_member_counter = (int *)malloc(numClusters*sizeof(int));

	cudaMalloc((void**)& d_points, clusterDimension*numPoints*sizeof(float));
	cudaMalloc((void**)& d_centers, clusterDimension*numClusters*sizeof(float));
	cudaMalloc((void **)&d_clusterIdx, numPoints*sizeof(int));
	cudaMalloc((void **)&d_member_counter, numClusters*sizeof(int));
	memset(h_clusterIdx,1, numPoints*sizeof(int));

	generate_random_data(h_points, h_centers_old, h_centers_new);


	cudaMemcpy(d_points, h_points, clusterDimension*numPoints*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_clusterIdx, h_clusterIdx, numPoints*sizeof(int), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();


	int count = 0;

	while (diff_norm > rand_range){
	//while (count < 200){
		clk1 = microtime();
		cudaMemcpyToSymbol(d_cons_centers, h_centers_new, clusterDimension*numClusters*sizeof(float)); 
		cudaMemcpy(d_centers, h_centers_new, clusterDimension*numClusters*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_member_counter, h_member_counter, numClusters*sizeof(int), cudaMemcpyHostToDevice);

		cudaThreadSynchronize();
		clk2 = microtime();
		printf("PART 1 :Count = %d\t Time = %g µs\n", count, (double)(clk2 - clk1));

		clk1 = microtime();
		calc_distance << <NumBlocks, ThreadsPerBlock, smem_size >> >(d_points, d_clusterIdx);
		cudaThreadSynchronize();
		clk2 = microtime();
		printf("PART 2 :Count = %d\t Time = %g µs\n", count, (double)(clk2 - clk1));
		
		clk1 = microtime();
		generate_new_center << <NumBlocks, ThreadsPerBlock >> >(d_points, d_centers, d_clusterIdx, d_member_counter);
		cudaThreadSynchronize();
		clk2 = microtime();
		printf("PART 3 :Count = %d\t Time = %g µs\n", count, (double)(clk2 - clk1));
		
		clk1 = microtime();
		cudaMemcpy(h_centers_new, d_centers, clusterDimension*numClusters*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_member_counter, d_member_counter, numClusters*sizeof(int), cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
		clk2 = microtime();
		printf("PART 4 :Count = %d\t Time = %g µs\n", count, (double)(clk2 - clk1));

		clk1 = microtime();

		member_division(h_centers_new, h_member_counter);
		diff_norm = calculate_norm(h_centers_old, h_centers_new);
		copy_centers(h_centers_old, h_centers_new);

		clk2 = microtime();
		printf("PART 5 :Count = %d\t Time = %g\t DN = %g \n", count, (double)(clk2 - clk1), diff_norm);
		
		count++;
	}



	cudaMemcpy(h_clusterIdx, d_clusterIdx, numPoints*sizeof(int), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	for (int i = 0; i < numClusters; i++){
		fprintf(stderr,"%d\t", h_clusterIdx[i]);
	}

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

	printf("Total Time = %g seconds\n", (double)((mclk2 - mclk1)/1000000));
	return 0;
}
