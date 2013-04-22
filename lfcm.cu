/*
   lfcm.c
   A literal FCM implementation.

   $Id: lfcm.c,v 1.3 2002/07/12 20:48:48 eschrich Exp $
   Steven Eschrich

   Copyright (C) 2002 University of South Florida

   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the
   Free Software Foundation; either version 2 of the License, or (at
   your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <math.h>
#include <sys/times.h>
#include <sys/resource.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include "utils.h"

#ifndef HANDLE_ERROR
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#endif





#define U(i,j) U[j][i]
#define DIM 256


	float *U;
	float *V;
	float *X;

	int C;
	float m;
	int S;
	int N;
	/*
__device__ float *U_d;
__device__ float *V_d;
__device__ float *X_d;
*/

/* Variables are defined statically here. These are reasonable
   defaults, but can be changed via the parameter to lfcmCluster() */
float epsilon=0.25;

int number_of_iterations;
long seed;
int *max_value;



/* Public functions */
int lfcm(float** U_d,float** V_d,float* X_d);


/* Private functions */
int    update_centroids();
__global__ void update_umatrix(float*,float*,float*,float*,int,int,int,float);

/* Utilities */
int    init(float** U_d,float** V_d, float* X_d);
__device__ int    is_example_centroid(float*,float*,int k,int,int);
__device__ float distance(float *,float *,int,int,int);

int output_centroids(char*);
int output_umatrix(char*);
int output_members(char*);

/* External functions */
int load_test_data(float **ds,float **ds_d, int *s, int *n);
int load_atr_data(char *filename, float **ds,float **ds_d ,int *s, int *n);
int load_mri_data(char *filename, float **ds,float **ds_d, int *s, int *n);


/* For testing purposes, we hard-code the desired number of clusters */
#define ATR_NUMBER_OF_CLUSTERS 5
#define MRI_NUMBER_OF_CLUSTERS 10
#define TEST_NUMBER_OF_CLUSTERS 2
#define TEST  1
#define ATR   2
#define MRI   3


/* Global variables */
int     dataset_type=MRI;
int     write_centroids=0;
int     write_umatrix=0;
int     write_members=0;

/* Variables that must be defined for called functions */
int  vals[][3]={{DIM,DIM,DIM},{0,0,0},{DIM,DIM,DIM},{4096,4096,4096}};


/* Function prototypes */
float *timing_of(struct rusage,struct rusage);  /* Calculate time in seconds */




int main(int argc, char **argv)
{
	cudaDeviceReset();
	number_of_iterations=0;
	struct rusage start_usage, end_usage;
	int ch;
	m = 2.0;
	C=2;
	S=2;
	N=2;
	float *perf_times;
	char   *filename;
	float *U_d,*V_d,*X_d;
	epsilon=0.225;
	m=2.0;
	seed=2000;
	max_value=vals[dataset_type];


	while ( (ch=getopt(argc, argv,"hw:d:s:")) != EOF ) {
		switch (ch) {
		case 'h':
			fprintf(stderr,"Usage\n" \
					"-d [a|t|m|s] Use dataset atr, mri, test, seawifs\n"\
					"-w write cluster centers and memberships out\n"\
					"-s seed  Use seed as the random seed\n");
			exit(1);
		case 'w':
			if ( !strcmp(optarg,"umatrix") ) write_umatrix=1;
			if ( !strcmp(optarg,"centroids") ) write_centroids=1;
			if ( !strcmp(optarg,"members") ) write_members=1;
			if ( !strcmp(optarg,"all"))
				write_umatrix=write_centroids=write_members=1;
			break;
		case 'd':
			if ( *optarg == 'a' ) dataset_type=ATR;
			if ( *optarg == 'm' ) dataset_type=MRI;
			if ( *optarg == 't' ) dataset_type=TEST;
			max_value=vals[dataset_type];
			break;
		case 's':
			seed=atol(optarg);
			break;
		}
	}
	/* Print out main parameters for this run */
	fprintf(stdout,"FCM Parameters\n clusterMethod=literal fcm\n");
	filename=argv[optind];
	fprintf(stdout," file=%s\n\n",filename);



	/* Load the dataset, using one of a particular group of datasets. */
	switch (dataset_type) {
	case TEST:
		load_test_data(&X,&X_d, &S, &N);
		C=TEST_NUMBER_OF_CLUSTERS;
		break;
	case ATR:
		load_atr_data(argv[optind],&X,&X_d, &S, &N);
		C=ATR_NUMBER_OF_CLUSTERS;
		break;
	case MRI:
		load_mri_data(argv[optind], &X,&X_d, &S, &N);
		C=MRI_NUMBER_OF_CLUSTERS;
		break;
	}


	fprintf(stdout, "Beginning to cluster here...\n");


	/* Time the fcm algorithm */
	//getrusage(RUSAGE_SELF, &start_usage);
	lfcm(&U_d,&V_d,X_d);
	//getrusage(RUSAGE_SELF, &end_usage);


	/* Output whatever clustering results we need */
	if ( write_centroids ) output_centroids(filename);
	if ( write_umatrix   ) output_umatrix(filename);
	if ( write_members   ) output_members(filename);


	/* Output timing numbers */
	//perf_times=timing_of(start_usage, end_usage);
	///printf("Timing: %f user, %f system, %f total.\n",
	//perf_times[0], perf_times[1], perf_times[0] +
	//perf_times[1]);

	printf("Clustering required %d iterations.\n", number_of_iterations);


	return 0;
}






/* Main entry point into code. Cluster the dataset, given the details
   in the parameter block. */
int lfcm(float** U_d,float** V_d,float* X_d)
{


	float sqrerror[((N+DIM-1)/DIM)*(C/1)];
	float *sqrerror_d;
	float sqrerror_sum;

	sqrerror_sum= 2 * epsilon;
	/* Initialize code  */
	init(U_d,V_d,X_d);
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaMalloc(&sqrerror_d,((N+DIM-1)/DIM)*sizeof(float)));
	printf("Beginning GPU side code\n");

	/* Run the updates iteratively */
	while (sqrerror_sum > epsilon ) {
		number_of_iterations++;
		HANDLE_ERROR(cudaMemcpy(U,*U_d,N*C*sizeof(float),cudaMemcpyDeviceToHost));
		update_centroids();
		HANDLE_ERROR(cudaMemcpy(*V_d,V,C*S*sizeof(float),cudaMemcpyHostToDevice));
		update_umatrix<<<(N+DIM-1)/DIM,DIM>>>(sqrerror_d,*U_d,*V_d,X_d,C,N,S,m);
		HANDLE_ERROR(cudaGetLastError());
		HANDLE_ERROR(cudaMemcpy(sqrerror,sqrerror_d,((N+DIM-1)/DIM)*sizeof(float),cudaMemcpyDeviceToHost));
		sqrerror_sum=0;
		cudaDeviceSynchronize();
		for(int i=0; i<((N+DIM-1)/DIM); i++)
			sqrerror_sum+=sqrerror[i];
	}


	/* We go ahead and update the centroids - presumably this will not
      change much, since the overall square error in U is small */
	update_centroids();

	HANDLE_ERROR(cudaMemcpy(U,*U_d,N*C*sizeof(float),cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(V,*V_d,C*S*sizeof(float),cudaMemcpyDeviceToHost));
	return 0;
}



/*
   update_centroids()
    Given a membership matrix U, recalculate the cluster centroids as the
    "weighted" mean of each contributing example from the dataset. Each
    example contributes by an amount proportional to the membership value.
 */
int update_centroids()
{
	  int i,k,x;
	  float *numerator, *denominator;
	  numerator = (float *)malloc(S*sizeof(float));
	  denominator = (float *)malloc(S*sizeof(float));

	  /* For each cluster */
	  for (i=0; i < C; i++)  {
		/* Zero out numerator and denominator options */
		for (x=0; x < S; x++) {
			numerator[x]=0;
			denominator[x]=0;
		}

		/* Calculate numerator */
		for (k=0; k < N; k++) {
			for (x=0; x < S; x++)
				numerator[x] += powf(U[k*C+i], m) * X[k*S+x];
		}

		/* Calculate denominator */
		for (k=0; k < N; k++) {
			for (x=0; x < S; x++)
				denominator[x] += powf(U[k*C+i], m);
		}

		/* Calculate V */
		for (x=0; x < S; x++) {
			V[i*S+x]= numerator[x] / denominator[x];
		}

	}  /* endfor: C clusters */

	return 0;
}

__global__ void update_umatrix(float *sqrerror,float* U_d, float* V_d, float* X_d,int C,int N,int S,float m)
{

	int i,j,k;
	int example_is_centroid;
	float summation, D_ki, D_kj;
	float newU;

	__shared__ float tmp_sqrerror[DIM];
	/* For each example in the dataset */
	k = threadIdx.x + blockIdx.x*blockDim.x;
	int local_offset = threadIdx.x;
	tmp_sqrerror[local_offset]=0;
	if(k<N)
	{
		/* Special case: If Example is equal to a Cluster Centroid,
       then U=1.0 for that cluster and 0 for all others */
		if ( (example_is_centroid=is_example_centroid(V_d,X_d,k,S,C)) != -1 ) {
			for(int i=0; i<C; i++)
			{
			if ( i == example_is_centroid )
				U_d[k*C+i]=1.0;
			else
				U_d[k*C+i]=0.0;
			}
			return;
		}
	/* For each class */
	for(int i=0; i< C; i++)
	{
		summation=0;

		/* Calculate summation */
		for (j=0; j < C; j++) {
			D_ki=distance(X_d, V_d,k*S,i*S,S);
			D_kj=distance(X_d, V_d,k*S,j*S,S);
			summation += powf( D_ki / D_kj , (2.0/ (m-1)));
		}

		/* Weight is 1/sum */
		newU=1.0/summation;

		/* Add to the squareDifference */
		tmp_sqrerror[local_offset] += powf(U_d[k*C+i] - newU, 2);

		U_d[k*C+i]=newU;

	}

	}
	__syncthreads();
	int t= blockDim.x/2;
	while(t>0)
	{
		if(k+t < N && threadIdx.x<t)
			tmp_sqrerror[local_offset] += tmp_sqrerror[local_offset+t];
		t/=2;
		__syncthreads();
	}

	if(threadIdx.x==0)
		sqrerror[blockIdx.x] = tmp_sqrerror[0];

}

/*===================================================
  Utilities

  init()
  checkIfExampleIsCentroid()
  distance()

  ===================================================*/

/* Allocate storage for U and V dynamically. Also, copy over the
   variables that may have been externally set into short names,
   which are private and easier to access.
 */
int init(float** U_d, float** V_d, float* X_d)
{
	int i,j;

	/* Allocate necessary storage */
	V=(float *)CALLOC(S*C, sizeof(float));

	U=(float *)CALLOC(C*N,sizeof(float));
	HANDLE_ERROR(cudaMalloc(U_d,N*C*sizeof(float)));
	HANDLE_ERROR(cudaMalloc(V_d,C*S*sizeof(float)));
	/* Place random values in V, then update U matrix based on it */
	srand48(seed);
	for (i=0; i < C; i++) {
		for (j=0; j < S; j++) {
			V[i*S+j]=drand48() * max_value[j];
		}
	}
	float *dummy;
	cudaMalloc(&dummy,N*sizeof(float));
	HANDLE_ERROR(cudaMemcpy(*V_d,V,C*S*sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(X_d,X,N*S*sizeof(float),cudaMemcpyHostToDevice));
	/* Once values are populated in V, update the U Matrix for sane values */
	update_umatrix<<<(N+DIM-1)/DIM,DIM>>>(dummy,*U_d,*V_d,X_d,C,N,S,m);
cudaDeviceSynchronize();

HANDLE_ERROR(cudaGetLastError());
fprintf(stdout,"Initialization completed.\n");

	return 0;
}


/* If X[k] == V[i] for some i, then return that i. Otherwise, return -1 */
__device__ int is_example_centroid(float* V_d,float* X_d,int k,int S, int C)
{
	int  i,x;

	for (i=0; i < C; i++) {
		for (x=0; x < S; x++) {
			if ( X_d[k*S+x] != V_d[i*S+x] ) break;
		}
		if ( x == S )  /* X==V */
			return i;
	}
	return -1;
}

__device__ float distance(float *v1, float *v2,int startV1,int startV2,int S)
{
	int x,i;
	float sum=0;

	for (x=startV1,i=startV2; x < startV1+S && i<startV2+S; x++, i++)
		sum += (v1[x] - v2[i]) * (v1[x] - v2[i]);

	return sqrtf(sum);
}



/*=====================================================
  Public output utilities

  output_centroids()
  output_umatrix()
  output_members()
  =====================================================*/
int output_centroids(char *filestem)
{
	FILE *fp;
	char buf[DIM];
	int i,j;

	sprintf(buf,"%s.centroids", filestem);
	fp=FOPEN(buf,"w");
	for (i=0;i < C ;i++) {
		for (j=0; j < S; j++)
			fprintf(fp, "%f\t",V[i*S+j]);
		fprintf(fp,"\n");
	}
	fclose(fp);

	return 0;
}

int output_umatrix(char *filestem)
{
	FILE *fp;
	char buf[DIM];
	int i,j;

	sprintf(buf,"%s.umatrix", filestem);
	fp=FOPEN(buf,"w");
	for (i=0; i < N; i++) {
		for (j=0; j < C; j++)
			fprintf(fp,"%f\t", U[i*C+j]);
		fprintf(fp,"\n");
	}
	fclose(fp);

	return 0;
}

int output_members(char *filestem)
{
	FILE *fp;
	char buf[DIM];
	int i,j,max;

	sprintf(buf,"%s.members", filestem);
	fp=FOPEN(buf,"w");
	for (i=0; i < N; i++) {
		for (max=j=0; j < C; j++)
			if ( U[i*C+j] > U[i*C+max] ) max=j;
		fprintf(fp,"%d\n",max);
	}
	fclose(fp);

	return 0;
}
