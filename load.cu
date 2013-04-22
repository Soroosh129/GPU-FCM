/*=======================================================================
 *
 * load.c
 * A file of loading routines, MRI, ATR and a simple test case.
 *
 * Note: ATR support is by default included, disable by undef'ing
 *  ATR_SUPPORT (a stub will be used instead)
 *
 * $Id: load.c,v 1.3 2002/07/12 20:48:48 eschrich Exp $
 * Steven Eschrich
 *
 * Copyright (C) 2002 University of South Florida
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *=======================================================================*/

#define ATR_SUPPORT
#define DIM 256

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#ifdef ATR_SUPPORT
 #include <tiffio.h>
#endif

#include "utils.h"


/* Load the dummy test file into the dataset. The information is
   produced without using a file, so this is generated in-memory

   The variable S sets the dimension, the variable N sets the number
   of examples.
*/
int load_test_data(float **ds,float **ds_d, int *s, int *n)
{
  int i;
  float *X;
  int S=1;
  int N=500;
  fprintf(stdout,"Loading test dataset...");
  X=(float *)CALLOC(N*S,sizeof(float));
  cudaMalloc(ds_d,N*S*sizeof(float));
  for (i=0; i < N; i++) {
    if ( i < 100 ) X[i*S+0]=i;
    else if ( i < 200 ) X[i*S+0]=i-100;
    else if ( i < 300 ) X[i*S+0]=i-200;
    else if ( i < 400 ) X[i*S+0]=255-i-300;
    else if ( i < 500 ) X[i*S+0]=255-i-400;
  }
  *n=N;
  *s=S;
  *ds=X;
  fprintf(stdout,"done (%d exmaples).\n", N);
  cudaMemcpy(*ds_d,X,N*S* sizeof(float),cudaMemcpyHostToDevice);

  return 0;
}



/*

  loadMRI()

  Loads an MRI image into memory as an int ** array of feature
  values.
*/
int load_mri_data(char *filename, float **ds,float **ds_d, int *s, int *n)
{
  FILE *fp;
  int image_length=DIM;
  int image_width=DIM;
  int S=3;
  int i,j;
  unsigned short int *buf;
  float *X;
  fprintf(stderr,"Loading MRI image %s...", filename);

  fp=FOPEN(filename,"r");

  /* Allocate storage */
  X=(float *)CALLOC(image_length * image_width * S,sizeof(float));
  HANDLE_ERROR(cudaMalloc(ds_d,image_length * image_width * S*sizeof(float)));
  buf=(unsigned short int *)CALLOC(image_width*image_length, sizeof(unsigned short int));
  for (i=0; i < S; i++) {
    fread(buf,2,image_width*image_length,fp);
    for (j=0; j < image_width*image_length;j++) {
      X[j*S+i]=buf[j];
    }
  }

  fclose(fp);
  fprintf(stderr,"done (%d examples).\n", image_width * image_length);
  *ds=X;
  *s=S;
  *n=image_length*image_width;
  cudaGetErrorString(cudaMemcpy(*ds_d,X,image_length * image_width * S * sizeof(float),cudaMemcpyHostToDevice));

  return 0;

}



/*

  load_atr_data()

  Loads an ATR TIFF image into memory as an int ** array of feature
  values.
*/

#ifdef ATR_SUPPORT
int kernel[5][5]={
{1,-2,0,2,-1},
{0,0,0,0,0},
{-2,4,0,-4,1},
{0,0,0,0,0},
{1,-2,0,2,-1}
};


int load_atr_data(char *filename, float **ds,float **ds_d ,int *s, int *n)
{
  TIFF *fp;
  int image_length, image_width;
  int i,j,rc,cc;
  unsigned char *buf;
  float *X;
  float val;
  fprintf(stderr,"Loading ATR image %s...", filename);

  if ( (fp=TIFFOpen(filename,"r")) == 0 )
    die("Can't open %s", filename);
  TIFFGetField(fp, TIFFTAG_IMAGELENGTH, &image_length);
  TIFFGetField(fp, TIFFTAG_IMAGEWIDTH, &image_width);

  /* Allocate storage */
  X=(float *)CALLOC(image_length * image_width * 2, sizeof(float));
cudaMalloc(ds_d,image_length * image_width * 2* sizeof(float));
  buf=(unsigned char *)CALLOC(image_width, sizeof(unsigned char));
  for (i=0; i < image_length; i++) {
    TIFFReadScanline(fp, buf, i, 0);
    for (j=0; j < image_width;j++) {
      X[i*image_width + j * image_length +0]=buf[j];
    }
  }

  TIFFClose(fp);
  fprintf(stderr,"done (%d examples).\n", image_width * image_length);

  /* normalize(); */

  /* If more than one feature asked for, generate Laws' texture values */
  fprintf(stderr,"Generating Laws' texture feature...");
  for (i=0; i < image_length; i++) {
    for (j=0; j < image_width;j++) {
      val=0;
           for (rc=0; rc < 5; rc++ ) {
              for (cc=0; cc < 5;cc++) {
                 if (i-2+rc < 0 || i-2+rc >= image_length ||
	               j-2+cc < 0 || j-2+cc >= image_width)
                    continue;
                 else
                    val += X[(i-2+rc)*(image_width) + (j-2+cc)*image_length+0] * (float)kernel[rc][cc];
              }
	   }
	   X[i*image_width + j * image_length +1]=val;
	 }
      }

  fprintf(stderr,"done.\n");
  *s=2;
  *n=image_length*image_width;
  *ds=X;

  cudaMemcpy(*ds_d,X,image_length * image_width * 2 * sizeof(float),cudaMemcpyHostToDevice);
  return 0;

}



#else
int load_atr_data()
{
  die("ATR support (TIFF support) not compiled in.\n");
}
#endif
