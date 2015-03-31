#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <math.h>
#include <time.h>

#define iterations 1
#define FLOAT_MAX 1e+37
#define numPoints 1024*1024
#define clusterDimension 16
#define numClusters 1024*3
#define ConstantMemFloats (64*1024)/4			//	64KB/4
#define rand_range 100
#define PC 0
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

__global__ void calc_distance(float *d_points, int *d_clusterIdx, float *d_mindistances, int step, int num_copy, int max_cached){

	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int tx = threadIdx.x;
	int min_pos = -1;
	extern __shared__ float s_points[];

	/*Getting the value from previous iterations*/
	if (i < numPoints){
		float min_dist = d_mindistances[i];
		float old_min_dist = d_mindistances[i];

		for (int j = 0; j < clusterDimension; j++){
			s_points[tx*clusterDimension + j] = d_points[i*clusterDimension + j];
		}

		for (int k = 0; k < num_copy; k++){
			float distance = 0.0f;
			for (int j = 0; j < clusterDimension; j++){
				distance += fabsf(s_points[tx*clusterDimension + j] - d_cons_centers[k*clusterDimension + j]);
				//distance += j;
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


/*
Hoping that the point is in L1 cache!
Need to try this with 48KB L1 cache!
Probably saves a lot of index calculations and array accesses
Probably everything is in registers
*/
__global__ void calc_distance2(float *d_points, int *d_clusterIdx, float *d_mindistances, int step, int num_copy, int max_cached){

	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int min_pos = -1;

	/*Getting the value from previous iterations*/
	if (i < numPoints){
		float points[clusterDimension];
		for (int j = 0; j < clusterDimension; j++){
			points[j] = d_points[i*clusterDimension + j];
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

#if (48 * 1024) / (numClusters * 4) >= 5
__global__ void generate_new_center(float *d_points, float *d_centers, int *d_clusterIdx, int *d_member_counter, int split_steps, int split_size){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int tx = threadIdx.x;
	extern __shared__ float s_centers[];
	if (i < numPoints){
		int clusterId = d_clusterIdx[i];
		float points[clusterDimension];
		for (int j = 0; j < clusterDimension; j++){
			points[j] = d_points[i*clusterDimension + j];
		}

		for (int j = 0; j < split_steps; j++){
			/*
			Example....
			3*3 -> 9 to 11
			4*3 -> 12 to 14
			5*3 -> 15 to 17
			start -> 13 , 14 , 15 , 16
			end   -> 15 , 16 , 17 , 18
			diff  ->  3  , 2  , 1  , 0
			*/
			int max_length;
			int end_point = (j + 1)*split_size - 1;
			if (end_point <= clusterDimension - 1){
				max_length = split_size;
			}
			else{
				max_length = clusterDimension - j*split_size;
			}

			/*Collabaratively Load 0's in shared memory!*/
			/*Each thread 0's out ele_to_zero entries*/
			int total_ele = (48 * 1024) / (4);
			int ele_to_zero = (int)ceil(total_ele *1.0f / blockDim.x);

			for (int k = tx*ele_to_zero; k < (tx + 1)*ele_to_zero; k++){
				if (k < total_ele)
					s_centers[k] = 0;
			}
			__syncthreads();

			for (int k = 0; k < max_length; k++){
				atomicAdd(&s_centers[max_length*clusterId + k], points[j*split_size + k]);
			}
			__syncthreads();


			/*Collabaratively write back to d_centers!*/
			/*Not using all the threads . Need to find a smarter way to do this!*/
			int clus = ceil(1.0f*numClusters / blockDim.x);
			for (int l = tx*clus; l < (tx + 1)*clus; l++){
				if (l < numClusters)
					for (int k = 0; k < max_length; k++){
						atomicAdd(&d_centers[j*split_size + clusterDimension*l + k], s_centers[max_length*l + k]);
					}
			}
			__syncthreads();

		}
		atomicAdd(&d_member_counter[clusterId], 1);
	}
}
#else
__global__ void generate_new_center(float *d_points, float *d_centers, int *d_clusterIdx, int * d_member_counter, int split_steps, int split_size){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < numPoints){
		int clusterId = d_clusterIdx[i];
		for (int j = 0; j < clusterDimension; j++){
			atomicAdd(&d_centers[clusterDimension*clusterId + j], d_points[i*clusterDimension + j]);
		}
		atomicAdd(&d_member_counter[clusterId], 1);
	}

}

#endif

__host__ void printDeviceInfo(){
	FILE * fp;
	fp = fopen("specifications.txt", "w");
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		fprintf(fp, "Name = %s\n", prop.name);
		fprintf(fp, "Global Memory = %u B\n", (int)prop.totalGlobalMem);
		fprintf(fp, "Shared memory per SM = %d B\n", (int)prop.sharedMemPerBlock);
		fprintf(fp, "Registers per SM = %d\n", (int)prop.regsPerBlock);
		fprintf(fp, "Warp size = %d threads\n", (int)prop.warpSize);
		fprintf(fp, "Max threads per Block = %d\n", (int)prop.maxThreadsPerBlock);
		fprintf(fp, "Total Constant Memory = %d B\n", (int)prop.totalConstMem);

		fprintf(fp, "Texture alignment = %d\n", (int)prop.textureAlignment);
		fprintf(fp, "Device overlap = %d\n", (int)prop.deviceOverlap);
		fprintf(fp, "Number of Multi-processors = %d\n", (int)prop.multiProcessorCount);
		fprintf(fp, "L2 SIZE = %d bytes\n", (int)prop.l2CacheSize);
		fprintf(fp, "Bus Width = %d bits\n", (int)prop.memoryBusWidth);
	}


}

int main(int argc, char **argv){

	printDeviceInfo();
	double clk1, clk2, mclk1, mclk2;
	mclk1 = microtime();


	float *d_points, *d_centers, *d_mindistances;
	int *d_clusterIdx, *d_member_counter;
	float diff_norm = (float)FLOAT_MAX;

	//Calculations for calc_distance()
	/*Block fitting*/
	int ThreadsPerBlock = (48 * 1024) / (clusterDimension*sizeof(float));
	int NumBlocks = (int)ceil(numPoints*1.0f / ThreadsPerBlock);

	/*Shared memory max fitting*/
	int smem_size = ThreadsPerBlock*clusterDimension*sizeof(float);

	/*Constant memory max fitting*/
	int max_cached, distance_steps;
	if (numClusters*clusterDimension * 4.0 <= 64 * 1024 * 1.0f){
		/*I can fit all the centers!*/
		max_cached = numClusters;
		distance_steps = 1;
	}
	else{
		/*One Point takes clusterDimension*4 memory , how mant can I fit in 64k?*/
		max_cached = (int)floor(64 * 1024 * 1.0f / (clusterDimension * 4));
		distance_steps = (int)ceil(numClusters*1.0f / max_cached*1.0f);
	}
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
	float *h_centers_zero = (float *)malloc(clusterDimension*numClusters*sizeof(float));

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
	generate_random_data(h_points, h_centers_old, h_centers_new);


	cudaMemcpy(d_points, h_points, clusterDimension*numPoints*sizeof(float), cudaMemcpyHostToDevice);
	memset(h_centers_zero, 0, clusterDimension*numClusters*sizeof(float));
	cudaThreadSynchronize();

	int count = 0;
	/*Each co-ordinate has a change less than 0.001 on average!*/
	while (diff_norm > (numPoints*clusterDimension) / 1000.0){
		clk1 = microtime();

		for (int i = 0; i < numPoints; i++){
			h_clusterIdx[i] = INT_MAX;
			h_mindistances[i] = (float)FLOAT_MAX;
		}
		memset(h_member_counter, 0, numClusters*sizeof(int));
		cudaMemcpy(d_clusterIdx, h_clusterIdx, numPoints*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_mindistances, h_mindistances, numPoints*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_centers, h_centers_zero, clusterDimension*numClusters*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_member_counter, h_member_counter, numClusters*sizeof(int), cudaMemcpyHostToDevice);

		cudaThreadSynchronize();
		clk2 = microtime();
		printf("PART 1 :Count = %d\t Time = %g µs\n", count, (double)(clk2 - clk1));

		clk1 = microtime();

		for (int step = 0; step < distance_steps; step++){
			//To adjust for last iteration!
			int num_copy = (max_cached <= numClusters - step*max_cached) ? max_cached : numClusters - step*max_cached;
			printf("num_cpy = %d\n", num_copy);
			cudaMemcpyToSymbol(d_cons_centers, h_centers_new + step*max_cached*clusterDimension
				, clusterDimension*num_copy*sizeof(float));
			cudaThreadSynchronize();
			//calc_distance << <NumBlocks, ThreadsPerBlock, smem_size >> >(d_points, d_clusterIdx,
			//	d_mindistances, step, num_copy, max_cached);
			calc_distance2 << <(int)ceil(numPoints / 1024.0), 1024 >> >(d_points, d_clusterIdx,
				d_mindistances, step, num_copy, max_cached);
			cudaThreadSynchronize();
			clk2 = microtime();

		}
		printf("PART 2 :Count = %d\t Time = %g µs\n", count, (double)(clk2 - clk1));

		clk1 = microtime();
		generate_new_center << <NumBlocks, ThreadsPerBlock, smem_size >> >(d_points, d_centers, d_clusterIdx, d_member_counter, split_steps, split_size);

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
	}


	cudaMemcpy(h_clusterIdx, d_clusterIdx, numPoints*sizeof(int), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	for (int i = 0; i < numClusters; i++){
		fprintf(stderr, "%d\t", h_clusterIdx[i]);
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

	printf("Total Time = %g seconds\n", (double)((mclk2 - mclk1) / 1000000));
	return 0;
}
