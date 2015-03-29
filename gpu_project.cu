#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <time.h>
//#include <microtime.h>

#define FLOAT_MAX 1e+37
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <math.h>


#define numPoints 1024*1024
#define numClusters 1024
#define clusterDimension 16
#define ThreadsPerBlock 1024
#define ConstantMemFloats 64*1024/4

__constant__ float d_cons_centers[ConstantMemFloats];
double microtime(){ return 0.0; }

__host__ void generate_random_data(float *h_points, float *h_centers_old, float *h_centers_new){

	//Randomly generating points using rand()
	srand((unsigned int)time(0));
	for (int i = 0; i < numPoints; i++){
		for (int j = 0; j < clusterDimension; j++)
		{
			h_points[i*clusterDimension + j] = (float)(rand());
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

__device__ float distance_func(float *point1, float *point2){
	float distance = 0.0f;
	for (int k = 0; k < clusterDimension; k++){
		distance += sqrtf((point1[k] - point2[k]) * (point1[k] - point2[k]));
	}
	return distance;

}



__global__ void calc_distance(float *d_points, float *d_centers, int *d_clusterIdx, int max_fitting , int numele)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int min_pos = -1;
	float min_dist = FLOAT_MAX, distance = 0.0f;
	
	if (i < numPoints){
		
		for (int j = 0; j < numele; j++){
			distance = distance_func(d_points + i*clusterDimension, d_cons_centers + j*clusterDimension);
			if (distance < min_dist){
				min_dist = distance;
				min_pos = j;
			}
		}

		d_clusterIdx[i] = min_pos;
	}


}

/*
__global__ void calc_distance(float *d_points, float *d_centers, int *d_clusterIdx, int max_fitting){

	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int tx = threadIdx.x;
	int min_pos = -1;
	float min_dist = FLOAT_MAX, distance = 0.0f;

	int max_steps = ceil(numClusters*1.0f / max_fitting);
	extern __shared__ float s_centers[];

	if (i < numPoints){
		for (int step = 0; step < max_steps; step++){
			if (tx < fminf(max_fitting, numClusters - step*max_fitting)){
				for (int j = 0; j < clusterDimension; j++){
					s_centers[tx*clusterDimension + j] = d_centers[step*max_fitting*clusterDimension + tx*clusterDimension + j];
				}
			}

			__syncthreads();

			for (int j = 0; j < max_fitting; j++){
				if (step*max_fitting + j < numClusters){
					distance = distance_func(d_points + i*clusterDimension, s_centers + j*clusterDimension);

					if (distance < min_dist){
						min_dist = distance;
						min_pos = step*max_fitting + j;
					}
				}
			}
			__syncthreads();

		} // step for
		d_clusterIdx[i] = min_pos;

	} // if

	
	int i = blockDim.x*blockIdx.x + threadIdx.x, min_pos = -1;
	float min_dist = FLOAT_MAX, distance = 0.0f;
	if (i < numPoints){
		for (int j = 0; j < numClusters; j++){
			distance = distance_func(d_points + i*clusterDimension, d_centers + j*clusterDimension);
			if (distance < min_dist){
				min_dist = distance;
				min_pos = j;
			}
		}
	}
	d_clusterIdx[i] = min_pos;
	
	

}
*/

__global__ void generate_new_center(float *d_points, float *d_centers, int *d_clusterIdx, int * d_member_counter){
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	/*
	__shared__ float new_d_centers[numClusters*clusterDimension];
	__shared__ int member_counter[numClusters];

	if (i < numClusters){
	member_counter[i] = 0;

	for (int j = 0; j < clusterDimension; j++){
	new_d_centers[i*clusterDimension + j] = 0;
	}
	}

	int clusterId = d_clusterIdx[i];
	int step = i*clusterDimension;
	for (int j = 0; j < clusterDimension; j++){
	new_d_centers[clusterDimension*clusterId + j] += d_points[i*clusterDimension + j];
	}
	member_counter[clusterId] += 1;
	*/
	int clusterId = d_clusterIdx[i];

	for (int j = 0; j < clusterDimension; j++){
		atomicAdd(&d_centers[clusterDimension*clusterId + j], d_points[i*clusterDimension + j]);
	}
	atomicAdd(&d_member_counter[clusterId], 1);

}



int main()
{

	double time1, time2, clk1, clk2;
	time1 = microtime();
	int NumBlocks = (int)ceil(numPoints*1.0f / ThreadsPerBlock);
	float diff_norm = 0.0f;

	int max_fitting = floor((64 * 1024 * 1.0f) / (clusterDimension*sizeof(float)));
	int smem_size = max_fitting*clusterDimension*sizeof(float);

	printf("max_fitting = %d \t smem = %d\n", max_fitting, smem_size);
	/*Stores the pointss*/
	float *h_points = (float *)malloc(clusterDimension*numPoints*sizeof(float));

	/*Need two arrays one for old centers, and one for new  for calculating NORM*/
	float *h_centers_old = (float *)malloc(clusterDimension*numClusters*sizeof(float));
	float *h_centers_new = (float *)malloc(clusterDimension*numClusters*sizeof(float));

	/*Stores cluster indexes of all the points*/
	int *h_clusterIdx = (int *)malloc(numPoints*sizeof(int));
	/*Consists number of members in a clusters*/
	int *h_memberCounter = (int *)malloc(numClusters*sizeof(int));
	float *d_points, *d_centers;
	int *d_clusterIdx, *d_member_counter;


	cudaMalloc((void**)& d_points, clusterDimension*numPoints*sizeof(float));
	cudaMalloc((void**)& d_centers, clusterDimension*numClusters*sizeof(float));
	cudaMalloc((void **)&d_clusterIdx, numPoints*sizeof(int));
	cudaMalloc((void **)&d_member_counter, numClusters*sizeof(int));
	memset(h_clusterIdx, 0, numPoints*sizeof(int));

	generate_random_data(h_points, h_centers_old, h_centers_new);

	cudaMemcpy(d_points, h_points, clusterDimension*numPoints*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_clusterIdx, h_clusterIdx, numPoints*sizeof(int), cudaMemcpyHostToDevice);



	int count = 0;
	while (count < 10)
	{
		clk1 = clock();

		memset(h_memberCounter, 0, numClusters*sizeof(int));
		cudaMemcpy(d_centers, h_centers_new, clusterDimension*numClusters*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_member_counter, h_memberCounter, numClusters*sizeof(int), cudaMemcpyHostToDevice);
		cudaThreadSynchronize();
		clk2 = clock();
		printf("PART 1 :Count = %d\t Time = %g\n", count, (double)(clk2 - clk1) / CLOCKS_PER_SEC);


		clk1 = clock();
		for (int step = 0; step < ceil(numClusters*1.0f/max_fitting); step++){
			int numele = fminf(max_fitting, numClusters - step*max_fitting);
			cudaMemcpyToSymbol(d_cons_centers, h_centers_new + step*max_fitting*clusterDimension, smem_size);
			calc_distance << <NumBlocks,ThreadsPerBlock >> >(d_points, d_centers, d_clusterIdx, max_fitting , numele);
		}

		//calc_distance << <NumBlocks,ThreadsPerBlock,smem_size >> >(d_points, d_centers, d_clusterIdx, max_fitting);
		cudaThreadSynchronize();
		clk2 = clock();
		printf("PART 2 :Count = %d\t Time = %g\n", count, (double)(clk2 - clk1) / CLOCKS_PER_SEC);

		clk1 = clock();
		generate_new_center << <NumBlocks,ThreadsPerBlock >> >(d_points, d_centers, d_clusterIdx, d_member_counter);
		cudaThreadSynchronize();
		clk2 = clock();
		printf("PART 3 :Count = %d\t Time = %g\n", count, (double)(clk2 - clk1) / CLOCKS_PER_SEC);

		clk1 = clock();
		cudaMemcpy(h_centers_new, d_centers, clusterDimension*numClusters*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_memberCounter, d_member_counter, numClusters*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_clusterIdx, d_clusterIdx, numClusters*sizeof(int), cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
		clk2 = clock();
		printf("PART 4 :Count = %d\t Time = %g\n", count, (double)(clk2 - clk1) / CLOCKS_PER_SEC);


		clk1 = clock();

		member_division(h_centers_new, h_memberCounter);
		diff_norm = calculate_norm(h_centers_old, h_centers_new);
		copy_centers(h_centers_old, h_centers_new);

		clk2 = clock();
		printf("PART 5 :Count = %d\t Time = %g\t DN = %g \n", count, (double)(clk2 - clk1) / CLOCKS_PER_SEC, diff_norm);

		count++;
		}

	printf("Done\n");

	cudaFree(d_centers);
	cudaFree(d_member_counter);
	cudaFree(d_clusterIdx);
	cudaFree(d_points);

	time2 = microtime();
	return 0;
}


