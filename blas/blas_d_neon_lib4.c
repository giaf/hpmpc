#include "../include/kernel_d_neon.h"



/* preforms                                          */
/* C  = A * B' (alg== 0)                             */
/* C += A * B' (alg== 1)                             */
/* C -= A * B' (alg==-1)                             */
/* where A, B and C are packed with block size 4     */
void dgemm_ppp_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, int alg)
	{

	const int bs = 4;

	int i, j, jj;
	
	i = 0;
	for(; i<m; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_pp_nt_4x4_vfpv3_lib4(k, &pA[0+i*sda], &pB[0+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs, alg);
			}
		jj = 0;
		for(; jj<n-j-1; jj+=2)
			{
			kernel_dgemm_pp_nt_4x2_c99_lib4(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], bs, alg);
			}
		for(; jj<n-j; jj++)
			{
			kernel_dgemm_pp_nt_4x1_c99_lib4(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], bs, alg);
			}
		}

	}



/* preforms                                          */
/* C  = A * B'                                       */
/* where A, B and C are packed with block size 4,    */
/* and B is upper triangular                         */
void dtrmm_ppp_lib(int m, int n, int offset, double *pA, int sda, double *pB, int sdb, double *pC, int sdc)
	{
	
	const int bs = 4;
	
	int i, j;
	
	if(offset%bs!=0)
		pB = pB+bs*sdb+bs*bs; // shift to the next block
	
	i = 0;
	for(; i<m; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_pp_nt_4x4_vfpv3_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs, 0);
			}
		if(n-j==1)
			{
			corner_dtrmm_pp_nt_4x1_c99_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
			}
		else if(n-j==2)
			{
			corner_dtrmm_pp_nt_4x2_c99_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
			}
		else if(n-j==3)
			{
			corner_dtrmm_pp_nt_4x3_c99_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
			}
		}

	// add to the last row
	for(j=0; j<n; j++)
		{
		pC[(m-1)%bs+j*bs+((m-1)/bs)*bs*sdc] += pB[j%bs+n*bs+(j/bs)*bs*sdb];
		}

	}



/* computes the mxn panel of                         */
/* C  = A * A'                                       */
/* where A, C are packed with block size 4           */
void dsyrk_ppp_lib(int m, int n, int k, double *pA, int sda, double *pC, int sdc)
	{
	
	const int bs = 4;
	
	int i, j, j_end;
	
	i = 0;
	for(; i<m; i+=4)
		{
		j = 0;
		j_end = i+4;
		if(n-3<j_end)
			j_end = n-3;
		for(; j<j_end; j+=4)
			{
			kernel_dgemm_pp_nt_4x4_vfpv3_lib4(k, &pA[0+i*sda], &pA[0+j*sda], &pC[0+(j+0)*bs+i*sdc], bs, 1);
			}
		if(n-j==1)
			{
			kernel_dgemm_pp_nt_4x1_c99_lib4(k, &pA[0+i*sda], &pA[0+j*sda], &pC[0+(j+0)*bs+i*sdc], bs, 1);
			}
		else if(n-j==2)
			{
			kernel_dgemm_pp_nt_4x2_c99_lib4(k, &pA[0+i*sda], &pA[0+j*sda], &pC[0+(j+0)*bs+i*sdc], bs, 1);
			}
		else if(n-j==3)
			{
			kernel_dgemm_pp_nt_4x3_c99_lib4(k, &pA[0+i*sda], &pA[0+j*sda], &pC[0+(j+0)*bs+i*sdc], bs, 1);
			}
		}

	}



/* computes the lower triangular Cholesky factor of pC, */
/* and copies its transposed in pL                      */
void dpotrf_p_dcopy_p_t_lib(int n, int nna, double *pC, int sdc, double *pL, int sdl)
	{

	const int bs = 4;
	
	int
		i, j, jj;

	j = 0;
	if(j<nna-3)
		{
		kernel_dpotrf_dtrsv_4x4_c99_lib4(n-j-4, &pC[0+j*bs+j*sdc], sdc);
		j += 4;
		for(; j<nna-3; j+=4)
			{
			i = j;
			for(; i<n; i+=4)
				{
				kernel_dgemm_pp_nt_4x4_vfpv3_lib4(j, &pC[0+i*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], bs, -1);
				}
			kernel_dpotrf_dtrsv_4x4_c99_lib4(n-j-4, &pC[0+j*bs+j*sdc], sdc);
			}
		}
	int j0 = j;
	if(j==0) // assume that n>0
		{
		kernel_dpotrf_dtrsv_dcopy_4x4_c99_lib4(n-j-4, &pC[0+j*bs+j*sdc], sdc, (bs-nna%bs)%bs, &pL[0+(j-j0)*bs+((j-j0)/bs)*bs*sdc], sdl);
		j += 4;
		}
	for(; j<n-3; j+=4)
		{
		i = j;
		for(; i<n; i+=4)
			{
			kernel_dgemm_pp_nt_4x4_vfpv3_lib4(j, &pC[0+i*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], bs, -1);
			}
		kernel_dpotrf_dtrsv_dcopy_4x4_c99_lib4(n-j-4, &pC[0+j*bs+j*sdc], sdc, (bs-nna%bs)%bs, &pL[0+(j-j0)*bs+((j-j0)/bs)*bs*sdc], sdl);
		}
	if(j<n)
		{
		if(n-j==1)
			{
			i = j;
			kernel_dgemm_pp_nt_4x1_c99_lib4(j, &pC[0+i*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], bs, -1);
			corner_dpotrf_dtrsv_dcopy_1x1_c99_lib4(&pC[0+j*bs+j*sdc], sdc, (bs-nna%bs)%bs, &pL[0+(j-j0)*bs+((j-j0)/bs)*bs*sdc], sdl);
			}
		else if(n-j==2)
			{
			i = j;
			kernel_dgemm_pp_nt_4x2_c99_lib4(j, &pC[0+i*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], bs, -1);
			corner_dpotrf_dtrsv_dcopy_2x2_c99_lib4(&pC[0+j*bs+j*sdc], sdc, (bs-nna%bs)%bs, &pL[0+(j-j0)*bs+((j-j0)/bs)*bs*sdc], sdl);
			}
		else if(n-j==3)
			{
			i = j;
			kernel_dgemm_pp_nt_4x3_c99_lib4(j, &pC[0+i*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], bs, -1);
			corner_dpotrf_dtrsv_dcopy_3x3_c99_lib4(&pC[0+j*bs+j*sdc], sdc, (bs-nna%bs)%bs, &pL[0+(j-j0)*bs+((j-j0)/bs)*bs*sdc], sdl);
			}
		}

	}



/* computes an mxn band of the lower triangular Cholesky factor of pC, supposed to be aligned */
void dpotrf_p_lib(int m, int n, double *pC, int sdc)
	{

	const int bs = 4;
	
	int i, j;

	j = 0;
	for(; j<n-3; j+=4)
		{
		i = j;
		for(; i<m; i+=4)
			{
			kernel_dgemm_pp_nt_4x4_vfpv3_lib4(j, &pC[0+i*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], bs, -1);
			}
		kernel_dpotrf_dtrsv_4x4_c99_lib4(m-j-4, &pC[0+j*bs+j*sdc], sdc);
		}
	if(j<n)
		{
		if(n-j==1)
			{
			i = j;
			for(; i<m; i+=4)
				{
				kernel_dgemm_pp_nt_4x1_c99_lib4(j, &pC[0+i*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], bs, -1);
				}
			kernel_dpotrf_dtrsv_1x1_c99_lib4(m-j-1, &pC[0+j*bs+j*sdc], sdc);
			}
		else if(n-j==2)
			{
			i = j;
			for(; i<m; i+=4)
				{
				kernel_dgemm_pp_nt_4x2_c99_lib4(j, &pC[0+i*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], bs, -1);
				}
			kernel_dpotrf_dtrsv_2x2_c99_lib4(m-j-2, &pC[0+j*bs+j*sdc], sdc);
			}
		else if(n-j==3)
			{
			i = j;
			for(; i<m; i+=4)
				{
				kernel_dgemm_pp_nt_4x3_c99_lib4(j, &pC[0+i*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], bs, -1);
				}
			kernel_dpotrf_dtrsv_3x3_c99_lib4(m-j-3, &pC[0+j*bs+j*sdc], sdc);
			}
		}

	}



void dgemv_p_n_lib(int n, int m, int offset, double *pA, int sda, double *x, double *y, int alg)
	{
	
	const int bs = 4;
	
	int j;

	int nna = (bs-offset%bs)%bs;

	if(nna>n) // it is always nna < bs , thus n<bs !!!!!
		{
		if(nna%2==1)
			{
			kernel_dgemv_n_1_c99_lib4(m, pA, x, y, alg);
			pA += 1;
			y  += 1;
			n  -= 1;
			}
		j = 0;
		for(; j<n-1; j+=2)
			{
			kernel_dgemv_n_2_c99_lib4(m, pA, x, y, alg);
			pA += 2;
			y  += 2;
			}
		for(; j<n; j++)
			{
			kernel_dgemv_n_1_c99_lib4(m, pA, x, y, alg);
			pA += 1;
			y  += 1;
			}
		return;
		}
	j=0;
	if(nna>0) // it can be nna = {1, 2, 3}
		{
		if(nna%2==1)
			{
			kernel_dgemv_n_1_c99_lib4(m, pA, x, y, alg);
			pA += 1;
			y  += 1;
			j++;
			}
		if(nna%4>=2)
			{
			kernel_dgemv_n_2_c99_lib4(m, pA, x, y, alg);
			pA += 2;
			y  += 2;
			j+=2;
			}
		pA += (sda-1)*bs;
		}
/*	for(; j<n-(bs-1); j+=bs)*/
	for(; j<n-7; j+=8)
		{
		kernel_dgemv_n_8_c99_lib4(m, pA, pA+sda*bs, x, y, alg);
		pA += 2*sda*bs;
		y  += 2*bs;
		}
	for(; j<n-3; j+=4)
		{
		kernel_dgemv_n_4_c99_lib4(m, pA, x, y, alg);
		pA += sda*bs;
		y  += bs;
		}
	for(; j<n-1; j+=2)
		{
		kernel_dgemv_n_2_c99_lib4(m, pA, x, y, alg);
		pA += 2;
		y  += 2;
		}
	for(; j<n; j++)
		{
		kernel_dgemv_n_1_c99_lib4(m, pA, x, y, alg);
		pA += 1;
		y  += 1;
		}

	}



void dgemv_p_t_lib(int n, int m, int offset, double *pA, int sda, double *x, double *y, int alg)
	{
	
	const int bs = 4;
	
	int nna = (bs-offset%bs)%bs;
	
	int j;
	
	j=0;
	for(; j<m-7; j+=8)
		{
		kernel_dgemv_t_8_c99_lib4(n, nna, pA+j*bs, sda, x, y+j, alg);
		}
	for(; j<m-3; j+=4)
		{
		kernel_dgemv_t_4_c99_lib4(n, nna, pA+j*bs, sda, x, y+j, alg);
		}
	for(; j<m-1; j+=2)
		{
		kernel_dgemv_t_2_c99_lib4(n, nna, pA+j*bs, sda, x, y+j, alg);
		}
	for(; j<m; j++)
		{
		kernel_dgemv_t_1_c99_lib4(n, nna, pA+j*bs, sda, x, y+j, alg);
		}

	}



void dtrmv_p_n_lib(int m, int offset, double *pA, int sda, double *x, double *y)
	{
	
	const int bs = 4;
	
	int mna = (bs-offset%bs)%bs;
	
	int j;
	
	if(mna>m) // it is always mna < bs , thus m<bs !!!!!
		{
		if(mna%2==1)
			{
			kernel_dgemv_n_1_c99_lib4(j+1, pA, x, y, 1);
			pA += 1;
			y  += 1;
			m  -= 1;
			}
		j = 0;
		for(; j<m-1; j+=2)
			{
			kernel_dgemv_n_2_c99_lib4(j+1, pA, x, y, 1);
			y[1] += pA[(j+1)*bs+1] * x[j+1];
			pA += 2;
			y  += 2;
			}
		for(; j<m; j++)
			{
			kernel_dgemv_n_1_c99_lib4(j+1, pA, x, y, 1);
			pA += 1;
			y  += 1;
			}
		return;
		}
	j=0;
	if(mna>0)
		{
		for(; j<mna%2; j++)
			{
			kernel_dgemv_n_1_c99_lib4(j+1, pA, x, y, 1);
			pA += 1;
			y  += 1;
			}
/*		for(; j<m-1; j+=2)*/
		for(; j<mna; j+=2)
			{
			kernel_dgemv_n_2_c99_lib4(j+1, pA, x, y, 1);
			y[1] += pA[(j+1)*bs+1] * x[j+1];
			pA += 2;
			y  += 2;
			}
		pA += (sda-1)*bs;
		}
	for(; j<m-7; j+=8)
		{
		kernel_dgemv_n_8_c99_lib4(j+1, pA, pA+sda*bs, x, y, 1);
		y[1] += pA[1+bs*(j+1)] * x[j+1];
		y[2] += pA[2+bs*(j+1)] * x[j+1] + pA[2+bs*(j+2)] * x[j+2];
		y[3] += pA[3+bs*(j+1)] * x[j+1] + pA[3+bs*(j+2)] * x[j+2] + pA[3+bs*(j+3)] * x[j+3];
		pA += sda*bs;
		y  += bs;

		kernel_dgemv_n_4_c99_lib4(4, pA+bs*(j+1), x+j+1, y, 1);
		y[1] += pA[1+bs*(j+5)] * x[j+5];
		y[2] += pA[2+bs*(j+5)] * x[j+5] + pA[2+bs*(j+6)] * x[j+6];
		y[3] += pA[3+bs*(j+5)] * x[j+5] + pA[3+bs*(j+6)] * x[j+6] + pA[3+bs*(j+7)] * x[j+7];
		pA += sda*bs;
		y  += bs;
		}
	for(; j<m-3; j+=4)
		{
		kernel_dgemv_n_4_c99_lib4(j+1, pA, x, y, 1);
		y[1] += pA[1+bs*(j+1)] * x[j+1];
		y[2] += pA[2+bs*(j+1)] * x[j+1] + pA[2+bs*(j+2)] * x[j+2];
		y[3] += pA[3+bs*(j+1)] * x[j+1] + pA[3+bs*(j+2)] * x[j+2] + pA[3+bs*(j+3)] * x[j+3];
		pA += sda*bs;
		y  += bs;
		}
	for(; j<m-1; j+=2)
		{
		kernel_dgemv_n_2_c99_lib4(j+1, pA, x, y, 1);
		y[1] += pA[(j+1)*bs+1] * x[j+1];
		pA += 2;
		y  += 2;
		}
	for(; j<m; j++)
		{
		kernel_dgemv_n_1_c99_lib4(j+1, pA, x, y, 1);
		pA += 1;
		y  += 1;
		}

	}



void dtrmv_p_t_lib(int m, int offset, double *pA, int sda, double *x, double *y)
	{
	
	const int bs = 4;
	
	int mna = (bs-offset%bs)%bs;
	int mmax = m;
	
	int j;
	
	j=0;
	if(mna>0)
		{
		for(; j<mna; j++)
			{
			kernel_dgemv_t_1_c99_lib4(mmax-j, mna-j, pA+j*bs+j, sda, x+j, y+j, 1);
			}
		pA += j + sda*bs;
		x  += j;
		y  += j;
		mmax -= j;
		}
	for(; j<m-7; j+=8)
		{
		y[0] += pA[0+bs*0] * x[0] + pA[1+bs*0] * x[1] + pA[2+bs*0] * x[2];
		y[1] += pA[1+bs*1] * x[1] + pA[2+bs*1] * x[2];
		y[2] += pA[2+bs*2] * x[2];

		kernel_dgemv_t_4_c99_lib4(4, 1, pA+3, sda, x+3, y, 1);

		y[4] += pA[4+bs*4] * x[4] + pA[5+bs*4] * x[5] + pA[6+bs*4] * x[6];
		y[5] += pA[5+bs*5] * x[5] + pA[6+bs*5] * x[6];
		y[6] += pA[6+bs*6] * x[6];

		kernel_dgemv_t_8_c99_lib4(mmax-7, 1, pA+7, sda, x+7, y, 1);

		pA += 2*bs*sda + 2*bs*bs;
		x  += 2*bs;
		y  += 2*bs;
		mmax -= 2*bs;
		}
	// clean-up loop
	for(; j<m-3; j+=4)
		{
		y[0] += pA[0+bs*0] * x[0] + pA[1+bs*0] * x[1] + pA[2+bs*0] * x[2];
		y[1] += pA[1+bs*1] * x[1] + pA[2+bs*1] * x[2];
		y[2] += pA[2+bs*2] * x[2];
		kernel_dgemv_t_4_c99_lib4(mmax, mmax, pA, sda, x, y, 1);
		pA += bs*sda + bs*bs;
		x  += bs;
		y  += bs;
		mmax -= bs;
		}
	for(; j<m-1; j+=2)
		{
		y[0] += pA[0+bs*0] * x[0];
		kernel_dgemv_t_2_c99_lib4(mmax, mmax, pA, sda, x, y, 1);
		pA += 2 + 2*bs;
		x  += 2;
		y  += 2;
		mmax -= 2;
		}
	for(; j<m; j++)
		{
		kernel_dgemv_t_1_c99_lib4(mmax, mmax, pA, sda, x, y, 1);
		pA += 1 + bs;
		x  += 1;
		y  += 1;
		mmax -= 1;
		}

	}



void dsymv_p_lib(int m, double *pA, int sda, double *x, double *y)
	{
	
	const int bs = 4;

	int j;
	
	double *ptrA, *ptrx;
	
	j=0;
	for(; j<m-7; j+=8)
		{
		kernel_dgemv_n_8_c99_lib4(j, pA, pA+sda*bs, x, y, 1);

		ptrA = pA + j*bs;
		ptrx =  x + j;

		y[0] += ptrA[0+bs*0]*ptrx[0] + ptrA[1+bs*0]*ptrx[1] + ptrA[2+bs*0]*ptrx[2] + ptrA[3+bs*0]*ptrx[3];
		y[1] += ptrA[1+bs*0]*ptrx[0] + ptrA[1+bs*1]*ptrx[1] + ptrA[2+bs*1]*ptrx[2] + ptrA[3+bs*1]*ptrx[3];
		y[2] += ptrA[2+bs*0]*ptrx[0] + ptrA[2+bs*1]*ptrx[1] + ptrA[2+bs*2]*ptrx[2] + ptrA[3+bs*2]*ptrx[3];
		y[3] += ptrA[3+bs*0]*ptrx[0] + ptrA[3+bs*1]*ptrx[1] + ptrA[3+bs*2]*ptrx[2] + ptrA[3+bs*3]*ptrx[3];

		kernel_dgemv_n_4_c99_lib4(4, ptrA+sda*bs, ptrx, y+4, 1);

		kernel_dgemv_t_4_c99_lib4(4, 0, ptrA+sda*bs, sda, ptrx+4, y, 1);

		y[4] += ptrA[4+bs*4]*ptrx[4] + ptrA[5+bs*4]*ptrx[5] + ptrA[6+bs*4]*ptrx[6] + ptrA[7+bs*4]*ptrx[7];
		y[5] += ptrA[5+bs*4]*ptrx[4] + ptrA[5+bs*5]*ptrx[5] + ptrA[6+bs*5]*ptrx[6] + ptrA[7+bs*5]*ptrx[7];
		y[6] += ptrA[6+bs*4]*ptrx[4] + ptrA[6+bs*5]*ptrx[5] + ptrA[6+bs*6]*ptrx[6] + ptrA[7+bs*6]*ptrx[7];
		y[7] += ptrA[7+bs*4]*ptrx[4] + ptrA[7+bs*5]*ptrx[5] + ptrA[7+bs*6]*ptrx[6] + ptrA[7+bs*7]*ptrx[7];
		
		kernel_dgemv_t_8_c99_lib4(m-j-8, 0, ptrA+2*sda*bs, sda, ptrx+8, y, 1);
		
		pA += 2*sda*bs;
		y  += 8;
		}
	for(; j<m-3; j+=4)
		{
		kernel_dgemv_n_4_c99_lib4(j, pA, x, y, 1);
		ptrA = pA + j*bs;
		ptrx =  x + j;
		y[0] += ptrA[0+bs*0]*ptrx[0] + ptrA[1+bs*0]*ptrx[1] + ptrA[2+bs*0]*ptrx[2] + ptrA[3+bs*0]*ptrx[3];
		y[1] += ptrA[1+bs*0]*ptrx[0] + ptrA[1+bs*1]*ptrx[1] + ptrA[2+bs*1]*ptrx[2] + ptrA[3+bs*1]*ptrx[3];
		y[2] += ptrA[2+bs*0]*ptrx[0] + ptrA[2+bs*1]*ptrx[1] + ptrA[2+bs*2]*ptrx[2] + ptrA[3+bs*2]*ptrx[3];
		y[3] += ptrA[3+bs*0]*ptrx[0] + ptrA[3+bs*1]*ptrx[1] + ptrA[3+bs*2]*ptrx[2] + ptrA[3+bs*3]*ptrx[3];
		kernel_dgemv_t_4_c99_lib4(m-j-4, 0, ptrA+4, sda, ptrx+4, y, 1);
		pA += 4 + (sda-1)*bs;
		y  += 4;
		}
	for(; j<m-1; j+=2)
		{
		kernel_dgemv_n_2_c99_lib4(j, pA, x, y, 1);
		ptrA = pA + j*bs;
		ptrx =  x + j;
		y[0] += ptrA[0+bs*0]*ptrx[0] + ptrA[1+bs*0]*ptrx[1];
		y[1] += ptrA[1+bs*0]*ptrx[0] + ptrA[1+bs*1]*ptrx[1];
		kernel_dgemv_t_2_c99_lib4(m-j-2, 0, ptrA+2, sda, ptrx+2, y, 1);
		pA += 2;
		y  += 2;
		}
	for(; j<m; j++)
		{
		kernel_dgemv_n_1_c99_lib4(j, pA, x, y, 1);
		ptrA = pA + j*bs;
		ptrx =  x + j;
		y[0] += ptrA[0+bs*0]*ptrx[0];
		kernel_dgemv_t_1_c99_lib4(m-j-1, 0, ptrA+1, sda, ptrx+1, y, 1);
		pA += 1;
		y  += 1;
		}

	}



void dtrsv_p_n_lib(int n, double *pA, int sda, double *x)
	{
	
	const int bs = 4;
	
	int j;
	
	double *ptrA, *ptrAd, *ptrx;

	// blocks of 4 (pA is supposed to be properly aligned)
	ptrA  = pA;
	ptrAd = pA;
	ptrx  = x;

	j = 0;
	for(; j<n-7; j+=8)
		{
		// correct
		kernel_dgemv_n_8_c99_lib4(j, ptrA, ptrA+bs*sda, x, ptrx, -1);

		// solve
		ptrx[0] = (ptrx[0]) / ptrAd[0+bs*0];
		ptrx[1] = (ptrx[1] - ptrx[0] * ptrAd[1+bs*0]) / ptrAd[1+bs*1];
		ptrx[2] = (ptrx[2] - ptrx[0] * ptrAd[2+bs*0] - ptrx[1] * ptrAd[2+bs*1]) / ptrAd[2+bs*2];
		ptrx[3] = (ptrx[3] - ptrx[0] * ptrAd[3+bs*0] - ptrx[1] * ptrAd[3+bs*1] - ptrx[2] * ptrAd[3+bs*2]) / ptrAd[3+bs*3];

		// correct
		kernel_dgemv_n_4_c99_lib4(4, ptrAd+bs*sda, ptrx, ptrx+4, -1);

		ptrA  += bs*sda;
		ptrAd += bs*(sda+bs);
		ptrx  += bs;

		// solve
		ptrx[0] = (ptrx[0]) / ptrAd[0+bs*0];
		ptrx[1] = (ptrx[1] - ptrx[0] * ptrAd[1+bs*0]) / ptrAd[1+bs*1];
		ptrx[2] = (ptrx[2] - ptrx[0] * ptrAd[2+bs*0] - ptrx[1] * ptrAd[2+bs*1]) / ptrAd[2+bs*2];
		ptrx[3] = (ptrx[3] - ptrx[0] * ptrAd[3+bs*0] - ptrx[1] * ptrAd[3+bs*1] - ptrx[2] * ptrAd[3+bs*2]) / ptrAd[3+bs*3];

		ptrA  += bs*sda;
		ptrAd += bs*(sda+bs);
		ptrx  += bs;

		}

	// clean up stuff at the end
	for(; j<n-3; j+=4)
		{
		// correct
		kernel_dgemv_n_4_c99_lib4(j, ptrA, x, ptrx, -1);

		// solve
		ptrx[0] = (ptrx[0]) / ptrAd[0+bs*0];
		ptrx[1] = (ptrx[1] - ptrx[0] * ptrAd[1+bs*0]) / ptrAd[1+bs*1];
		ptrx[2] = (ptrx[2] - ptrx[0] * ptrAd[2+bs*0] - ptrx[1] * ptrAd[2+bs*1]) / ptrAd[2+bs*2];
		ptrx[3] = (ptrx[3] - ptrx[0] * ptrAd[3+bs*0] - ptrx[1] * ptrAd[3+bs*1] - ptrx[2] * ptrAd[3+bs*2]) / ptrAd[3+bs*3];

		ptrA  += bs*sda;
		ptrAd += bs*(sda+bs);
		ptrx  += bs;

		}
	for(; j<n-1; j+=2)
		{
		// correct
		kernel_dgemv_n_2_c99_lib4(j, ptrA, x, ptrx, -1);

		// solve
		ptrx[0] = (ptrx[0]) / ptrAd[0+bs*0];
		ptrx[1] = (ptrx[1] - ptrx[0] * ptrAd[1+bs*0]) / ptrAd[1+bs*1];
		
		ptrA  += 2;
		ptrAd += 2*bs+2;
		ptrx  += 2;
		}
	for(; j<n; j++)
		{
		// correct
		kernel_dgemv_n_1_c99_lib4(j, ptrA, x, ptrx, -1);

		// solve
		ptrx[0] = (ptrx[0]) / ptrAd[0+bs*0];
		
		ptrA  += 1;
		ptrAd += bs+1;
		ptrx  += 1;
		}

	}



void dtrsv_p_t_lib(int n, double *pA, int sda, double *x)
	{
	
	const int bs = 4;
	
	int i, j;
	
	int rn = n%bs;
	int qn = n/bs;
	int ri, qi;
	
	double *ptrA, *ptrx;
	
	// clean up stuff at the end
	j = 0;
	ptrA = pA + qn*bs*(sda+bs);
	ptrx = x + qn*bs;

	for(; j<rn%2; j++)
		{
		i = rn-1-j;
		kernel_dgemv_t_1_c99_lib4(j, j, &ptrA[i+1+bs*(i+0)], sda, &ptrx[i+1], &ptrx[i], -1);
		ptrx[i+0] = (ptrx[i+0]) / ptrA[i+0+bs*(i+0)];
		}
	for(; j<rn; j+=2)
		{
		i = rn-2-j;
		kernel_dgemv_t_2_c99_lib4(j, j, &ptrA[i+2+bs*(i+0)], sda, &ptrx[i+2], &ptrx[i], -1);
		ptrx[i+1] = (ptrx[i+1]) / ptrA[(i+1)+bs*(i+1)];
		ptrx[i+0] = (ptrx[i+0] - ptrA[(i+1)+bs*(i+0)]*ptrx[i+1]) / ptrA[(i+0)+bs*(i+0)];
		}

	// blocks of 8
	j = 0;
	for(; j<qn-1; j+=2)
		{
		
		// all 4 rows
		ptrA = pA + (qn-j-2)*bs*(sda+bs) ;
		ptrx = x  + (qn-j-2)*bs          ;

		// correct
		kernel_dgemv_t_8_c99_lib4(rn+j*bs, 0, ptrA+2*bs*sda, sda, ptrx+8, ptrx, -1);

		// last 4 rows
		ptrA = pA + (qn-j-1)*bs*(sda+bs) ;
		ptrx = x  + (qn-j-1)*bs          ;

		// solve
		ptrx[3] = (ptrx[3]) / ptrA[3+bs*3];
		ptrx[2] = (ptrx[2] - ptrA[3+bs*2]*ptrx[3]) / ptrA[2+bs*2];
		ptrx[1] = (ptrx[1] - ptrA[3+bs*1]*ptrx[3] - ptrA[2+bs*1]*ptrx[2]) / ptrA[1+bs*1];
		ptrx[0] = (ptrx[0] - ptrA[3+bs*0]*ptrx[3] - ptrA[2+bs*0]*ptrx[2] - ptrA[1+bs*0]*ptrx[1]) / ptrA[0+bs*0];

		// first 4 rows
		ptrA = pA + (qn-j-2)*bs*(sda+bs) ;
		ptrx = x  + (qn-j-2)*bs          ;

		// correct
		kernel_dgemv_t_4_c99_lib4(4, 0, ptrA+bs*sda, sda, ptrx+4, ptrx, -1);

		// solve
		ptrx[3] = (ptrx[3]) / ptrA[3+bs*3];
		ptrx[2] = (ptrx[2] - ptrA[3+bs*2]*ptrx[3]) / ptrA[2+bs*2];
		ptrx[1] = (ptrx[1] - ptrA[3+bs*1]*ptrx[3] - ptrA[2+bs*1]*ptrx[2]) / ptrA[1+bs*1];
		ptrx[0] = (ptrx[0] - ptrA[3+bs*0]*ptrx[3] - ptrA[2+bs*0]*ptrx[2] - ptrA[1+bs*0]*ptrx[1]) / ptrA[0+bs*0];

		}
	
	// blocks of 4
	for(; j<qn; j++)
		{
		
		// first 4 rows
		ptrA = pA + (qn-j-1)*bs*(sda+bs) ;
		ptrx = x  + (qn-j-1)*bs          ;
		
		kernel_dgemv_t_4_c99_lib4(rn+j*bs, 0, ptrA+bs*sda, sda, ptrx+4, ptrx, -1);
		ptrx[3] = (ptrx[3]) / ptrA[3+bs*3];
		ptrx[2] = (ptrx[2] - ptrA[3+bs*2]*ptrx[3]) / ptrA[2+bs*2];
		ptrx[1] = (ptrx[1] - ptrA[3+bs*1]*ptrx[3] - ptrA[2+bs*1]*ptrx[2]) / ptrA[1+bs*1];
		ptrx[0] = (ptrx[0] - ptrA[3+bs*0]*ptrx[3] - ptrA[2+bs*0]*ptrx[2] - ptrA[1+bs*0]*ptrx[1]) / ptrA[0+bs*0];

		}

	}
