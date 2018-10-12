#include "bl_dgemm.h"


void microkernel(
    int    k,
    double *A,
    int    lda,
    double *B,
    int    ldb,
    double *C,        
    int    ldc        
)
{
   __m256d c_03_0, c_03_1, c_03_2, c_03_3;     // C(0:3,0:3)
 
  // registers for broadcasted values of the vector br
   __m256d b_0_broad, b_1_broad, b_2_broad, b_3_broad;     

   __m256d a_03; //a(0:3)

   int p;

   c_03_0 = _mm256_load_pd(C);
   c_03_1 = _mm256_load_pd(C + ldc);
   c_03_2 = _mm256_load_pd(C + 2*ldc);
   c_03_3 = _mm256_load_pd(C + 3*ldc);  

   for ( p = 0; p < k; p++ ) {
      // assumes col-major ordering as in BLAS 
      // mem_addr must be aligned on a 32-byte boundary or a general-protection exception may be generated.
      // -> then use _mm256_loadu_pd
      a_03 = _mm256_load_pd(A);
      A += lda;
//       _mm_prefetch( (const char*) A, _MM_HINT_T0); //lowers Flops for smaller matrices
 
      b_0_broad = _mm256_broadcast_sd(B);
      b_1_broad = _mm256_broadcast_sd(B + ldb);
      b_2_broad = _mm256_broadcast_sd(B + 2*ldb);
      b_3_broad = _mm256_broadcast_sd(B + 3*ldb);

      //1st column
//       c_03_0 = _mm256_add_pd(c_03_0, _mm256_mul_pd(a_03,b_0_broad));
      c_03_0 = _mm256_fmadd_pd(a_03, b_0_broad, c_03_0);

      //2nd column
//       c_03_1 = _mm256_add_pd(c_03_1, _mm256_mul_pd(a_03,b_1_broad));
      c_03_1 = _mm256_fmadd_pd(a_03, b_1_broad, c_03_1);

      //3rd column
//       c_03_2 = _mm256_add_pd(c_03_2, _mm256_mul_pd(a_03,b_2_broad));
      c_03_2 = _mm256_fmadd_pd(a_03, b_2_broad, c_03_2);

      //4th column
//       c_03_3 = _mm256_add_pd(c_03_3, _mm256_mul_pd(a_03,b_3_broad));
      c_03_3 = _mm256_fmadd_pd(a_03, b_3_broad, c_03_3);

      B += 1;
  }
  // store results in C (assumes col-major ordering as in BLAS)
  // mem_addr must be aligned on a 32-byte boundary or a general-protection exception may be generated.
  // -> then use _mm256_storeu_pd
  _mm256_store_pd(C,c_03_0);
  C += ldc;
  
  _mm256_store_pd(C,c_03_1);
  C += ldc;
  
  _mm256_store_pd(C,c_03_2);
  C += ldc;
  
  _mm256_store_pd(C,c_03_3);
  C += ldc;
}

void macrokernel(
    int    m,
    int    n,
    int    k,
    double *A,
    int    lda,
    double *B,
    int    ldb,
    double *C,        
    int    ldc        
)
{
    int    i, j;
    for ( j = 0; j < n; j += NR ) {         
        for ( i = 0; i < m; i += MR ) { 
            microkernel( k, &A( i, 0 ), lda, &B( 0, j ), ldb, &C( i, j ), ldc );
        }                                         
    } 
}

void bl_dgemm(
    int    m,
    int    n,
    int    k,
    double *A,
    int    lda,
    double *B,
    int    ldb,
    double *C,        
    int    ldc        
)
{
    int ic, jc, pc;
    int _m, _n, _k;

    // Early return if possible
    if ( m == 0 || n == 0 || k == 0 ) {
        fprintf(stderr, "One dimension is zero.");
        return;
    }

    for ( jc = 0; jc < n; jc += NC ) {
        _n = min(n - jc, NC);
        for ( pc = 0; pc < k; pc += KC ) {
            _k = min(k - pc, KC);
            for ( ic = 0; ic < m; ic += MC ) {
                _m = min(m - ic, MC);
                macrokernel(_m, _n, _k, &A( ic, pc ), lda, &B( pc, jc ), ldb, &C( ic, jc ), ldc );                           
            }
        }
    }
}


