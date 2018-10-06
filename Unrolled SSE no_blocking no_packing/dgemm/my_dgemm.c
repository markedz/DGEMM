#include "bl_dgemm.h"


void microkernel( int k, double *A, int lda, double *B, int ldb, double *C, int ldc )
{
   __m128d c_01_0, c_01_1, c_01_2, c_01_3,     // C(0:1,0:3)
           c_23_0, c_23_1, c_23_2, c_23_3;    // C(2:3,0:3)

   // registers for broadcasted values of the vector br
   __m128d b_0_broad, b_1_broad, b_2_broad, b_3_broad;     

   __m128d a_01, a_23; 

   int p;

   c_01_0 = _mm_load_pd(C);
   c_23_0 = _mm_load_pd(C + 2);
   c_01_1 = _mm_load_pd(C + ldc);
   c_23_1 = _mm_load_pd(C + ldc + 2); 
   c_01_2 = _mm_load_pd(C + 2*ldc);
   c_23_2 = _mm_load_pd(C + 2*ldc + 2);
   c_01_3 = _mm_load_pd(C + 3*ldc);      
   c_23_3 = _mm_load_pd(C + 3*ldc + 2); 

   for (p = 0; p < k; p++ ) {
      // assumes col-major ordering as in BLAS 
      // mem_addr must be aligned on a 16-byte boundary or a general-protection exception may be generated.
      a_01 = _mm_load_pd(A);
      a_23 = _mm_load_pd(A + 2);

      b_0_broad = _mm_load1_pd(B);
      b_1_broad = _mm_load1_pd(B + ldb);
      b_2_broad = _mm_load1_pd(B + 2*ldb);
      b_3_broad = _mm_load1_pd(B + 3*ldb);

      //1st column
      c_01_0 = _mm_add_pd(c_01_0, _mm_mul_pd(a_01,b_0_broad));
      c_23_0 = _mm_add_pd(c_23_0, _mm_mul_pd(a_23,b_0_broad));

      //2nd column
      c_01_1 = _mm_add_pd(c_01_1, _mm_mul_pd(a_01,b_1_broad));
      c_23_1 = _mm_add_pd(c_23_1, _mm_mul_pd(a_23,b_1_broad));

      //3rd column
      c_01_2 = _mm_add_pd(c_01_2, _mm_mul_pd(a_01,b_2_broad));
      c_23_2 = _mm_add_pd(c_23_2, _mm_mul_pd(a_23,b_2_broad));

      //4th column
      c_01_3 = _mm_add_pd(c_01_3, _mm_mul_pd(a_01,b_3_broad));
      c_23_3 = _mm_add_pd(c_23_3, _mm_mul_pd(a_23,b_3_broad));

      A += lda;
      B += 1;
  }
  // store results in C (assumes col-major ordering as in BLAS)
  // mem_addr must be aligned on a 16-byte boundary or a general-protection exception may be generated.
  _mm_store_pd(C,c_01_0);
  _mm_store_pd(C+2,c_23_0);
  C += ldc;
  
  _mm_store_pd(C,c_01_1);
  _mm_store_pd(C+2,c_23_1);
  C += ldc;
  
  _mm_store_pd(C,c_01_2);
  _mm_store_pd(C+2,c_23_2);
  C += ldc;
  
  _mm_store_pd(C,c_01_3);
  _mm_store_pd(C+2,c_23_3);
  C += ldc;
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


