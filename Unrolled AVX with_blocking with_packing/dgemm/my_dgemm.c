#include "bl_dgemm.h"

void packAr(
    int    kc,
    double *A,
    int    lda,
    double *buf // buffer
)
{
    int i, j;
    for ( j = 0; j < kc; j++ ) {
        for ( i = 0; i < MR; i++ ) {
            buf[i] = A(i,j);
        }
        buf += MR;
    }
}

void packBr(
    int     kc,
    double  *B,
    int     ldb,
    double  *buf // buffer
)
{
    int i, j;

    for ( i = 0; i < kc; i++ ) {
        for ( j = 0; j < NR; j++ ) {
            buf[j] = B(i,j);
        }
        buf += NR;
    }
}

void packAc(
    int     mc,
    int     kc,      // could be < KC
    double  *A,
    int     lda,
    double  *buf     // buffer
)
{
    int i, j, q, r, Ar_size;

    q = mc / MR;
    r = mc % MR;
    Ar_size = MR*kc; // reused q times

  for( i = 0; i < q; i++ ) {
      packAr(kc,A,lda,buf);
      A += MR;
      buf += Ar_size;
  }
  
  if ( r > 0 ) {     // padding with zeros
      for ( j = 0; j < kc; j++ ) {
          for ( i = 0; i < r; i++ ) {
              buf[i] = A[i];
          }
          A += lda;
          for ( ; i < MR; i++ )
              buf[i] = 0.0;
          buf += MR;
      }
  }
}

void packBc(
    int     nc,
    int     kc,      // could be < KC
    double  *B,
    int     ldb,
    double  *buf     // buffer
)
{
    int i, j, q, r, Br_size;

    q = nc / NR;
    r = nc % NR;
    Br_size = NR*kc; // reused q times
    
    for( i = 0; i < q; i++ ) {
        packBr(kc,B,ldb,buf);
        B += ldb*NR;
        buf += Br_size;
  }
  
    if ( r > 0 ) {     // padding with zeros
        for ( i = 0; i < kc; i++ ) {
            for ( j = 0; j < r; j++ ) {
                buf[j] = B(i,j);
            }
            B += 1;
            for ( ; j < NR; j++ )
                buf[j] = 0.0;
            buf += NR;
        }
    }
}

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
      A += MR;
 
      b_0_broad = _mm256_broadcast_sd(B);
      b_1_broad = _mm256_broadcast_sd(B + 1);
      b_2_broad = _mm256_broadcast_sd(B + 2);
      b_3_broad = _mm256_broadcast_sd(B + 3);

      //1st column
      c_03_0 = _mm256_fmadd_pd(a_03, b_0_broad, c_03_0);

      //2nd column
      c_03_1 = _mm256_fmadd_pd(a_03, b_1_broad, c_03_1);

      //3rd column
      c_03_2 = _mm256_fmadd_pd(a_03, b_2_broad, c_03_2);

      //4th column
      c_03_3 = _mm256_fmadd_pd(a_03, b_3_broad, c_03_3);

      B += NR;
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
            microkernel( k, &A[k*i], lda, &B[k*j], ldb, &C( i, j ), ldc );
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
    int ic, jc, pc, i;
    int _m, _n, _k;
    double *Abuf, *Bbuf;

    // Early return if possible
    if ( m == 0 || n == 0 || k == 0 ) {
        fprintf(stderr, "One dimension is zero.");
        return;
    }

    Abuf    = bl_malloc_aligned( MC, KC, sizeof(double) ); 
    Bbuf    = bl_malloc_aligned( KC, NC, sizeof(double) ); 

    for ( jc = 0; jc < n; jc += NC ) {
        _n = min(n - jc, NC);
        for ( pc = 0; pc < k; pc += KC ) {
            _k = min(k - pc, KC);
            packBc( _n, _k, &B( pc, jc ), ldb, Bbuf );
            for ( ic = 0; ic < m; ic += MC ) {
                _m = min(m - ic, MC);
                packAc( _m, _k, &A( ic, pc ), lda, Abuf );
                macrokernel( _m, _n, _k, Abuf, lda, Bbuf, ldb, &C( ic, jc ), ldc );                           
            }
        }
    }
}


