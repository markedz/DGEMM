#include "bl_dgemm.h"


void microkernel( int k, double *A, int lda, double *B, int ldb, double *C, int ldc )
{
//   int p,i,j;
//    for (p = 0; p < k; p++ ) {
//     for (i = 0; i < MR; i++ )
//      for (j = 0; j < NR; j++ )
//         C(i,j) += A( i, p ) * B( p, j);
//   }
  
   register double 
   c00, c01, c02, c03,
   c10, c11, c12, c13,
   c20, c21, c22, c23,
   c30, c31, c32, c33;
    
   register double b0, b1, b2, b3;
   register double a0, a1, a2, a3;
   
   int p;

   c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0,
   c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0,
   c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0,
   c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;
   
   for (p = 0; p < k; p++ ) {
      
      //using indirect addressing (assumes col-major ordering as in BLAS)
      a0 = A[0];
      a1 = A[1];
      a2 = A[2];
      a3 = A[3];
      
      b0 = B[0];
      b1 = B[ldb];
      b2 = B[2*ldb];
      b3 = B[3*ldb]; //*(B + 3*ldb);
      
      c00 += a0*b0;
      c10 += a1*b0;
      c20 += a2*b0;
      c30 += a3*b0;
      
      c01 += a0*b1;
      c11 += a1*b1;
      c21 += a2*b1;
      c31 += a3*b1;
      
      c02 += a0*b2;
      c12 += a1*b2;
      c22 += a2*b2;
      c32 += a3*b2;
      
      c03 += a0*b3;
      c13 += a1*b3;
      c23 += a2*b3;
      c33 += a3*b3;
      
      A += lda;
      B++;
      
  }
  //store results in C (assumes col-major ordering as in BLAS)
  C[0] += c00;
  C[1] += c10;
  C[2] += c20;
  C[3] += c30;
  C += ldc;
  
  C[0] += c01;
  C[1] += c11;
  C[2] += c21;
  C[3] += c31;
  C += ldc;
  
  C[0] += c02;
  C[1] += c12;
  C[2] += c22;
  C[3] += c32;
  C += ldc;
  
  C[0] += c03;
  C[1] += c13;
  C[2] += c23;
  C[3] += c33;
}

void bl_dgemm(
    int    m,
    int    n,
    int    k,
    double *A,
    int    lda,
    double *B,
    int    ldb,
    double *C,        // must be aligned for vector store
    int    ldc        
)
{
    int    i, j, p;
    int    ir, jr;

    if ( m == 0 || n == 0 || k == 0 ) {
        fprintf(stderr, "One dimension is zero.");
        return;
    }

    for ( j = 0; j < n; j += NR ) {          // Start 2-nd loop
        for ( i = 0; i < m; i += MR ) {      // Start 1-st loop

            microkernel( k, &A( i, 0 ), lda, &B( 0, j ), ldb, &C( i, j ), ldc );

        }                                          // End   1-st loop
    }                                              // End   2-nd loop

}


