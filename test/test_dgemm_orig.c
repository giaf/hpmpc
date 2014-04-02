#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

#include "../include/aux_d.h"
/*#include "../include/aux_s.h"*/
#include "../include/blas_d.h"
/*#include "../include/blas_lib_s.h"*/
#include "../include/block_size.h"



int main()
	{
	
	int i, j, rep;
	
	const int bs = D_MR; //d_get_mr();
/*	const int bss = S_MR; //s_get_mr();*/
	
	printf("\nbs = %d\n\n", bs);
	
	int n = 11;
	int nrep = 1;
	
	double *A; d_zeros(&A, n, n);
	double *B; d_zeros(&B, n, n);
	double *C; d_zeros(&C, n, n);
	double *L; d_zeros(&L, n, n);
/*	float *sA; s_zeros(&sA, n, n);*/
/*	float *sB; s_zeros(&sB, n, n);*/
	
	for(i=0; i<n*n; i++)
		{
		A[i] = i;
/*		sA[i] = i;*/
		}
	
	B[0] = 2;
/*	B[1] = 1;*/
/*	sB[0] = 2;*/
/*	sB[1] = 1;*/
	for(i=1; i<n-1; i++)
		{
/*		B[i*(n+1)-1] = 1;*/
		B[i*(n+1)+0] = 2;
/*		B[i*(n+1)+1] = 1;*/
/*		sB[i*(n+1)-1] = 1;*/
/*		sB[i*(n+1)+0] = 2;*/
/*		sB[i*(n+1)+1] = 1;*/
		}
/*	B[n*n-2] = 1;*/
	B[n*n-1] = 2;
/*	sB[n*n-2] = 1;*/
/*	sB[n*n-1] = 2;*/
	
	for(i=0; i<n; i++)
		C[i*(n+1)] = 2;
	for(i=0; i<n-1; i++)
		C[1+i*(n+1)] = 1;

	d_print_mat(n, n, C, n);

	int pn = ((n+bs-1)/bs)*bs;//+4;	
/*	int pns = ((n+bss-1)/bss)*bss;//+4;	*/

	double *pA; d_zeros_align(&pA, pn, pn);
	double *pB; d_zeros_align(&pB, pn, pn);
	double *pC; d_zeros_align(&pC, pn, pn);
/*	float *spA; s_zeros_align(&spA, pns, pns);*/
/*	float *spB; s_zeros_align(&spB, pns, pns);*/
/*	float *spC; s_zeros_align(&spC, pns, pns);*/
	
	d_cvt_mat2pmat(n, n, 0, bs, A, n, pA, pn);
	d_cvt_mat2pmat(n, n, 0, bs, B, n, pB, pn);
/*	s_cvt_mat2pmat(n, n, 0, bss, sA, n, spA, pns);*/
/*	s_cvt_mat2pmat(n, n, 0, bss, sB, n, spB, pns);*/
	
	double *x; d_zeros_align(&x, n, 1);
	double *y; d_zeros_align(&y, n, 1);
	
	x[2] = 1;

/*	for(i=0; i<pn*pn; i++) pC[i] = -1;*/
/*	for(i=0; i<pn*pn; i++) spC[i] = -1;*/
	
//	d_print_pmat(pn, pn, bs, pA, pn);
//	d_print_pmat(pn, pn, bs, pB, pn);
//	d_print_pmat(pn, pn, bs, pC, pn);
//	d_print_mat(n, n, B, n);

//	double *x; d_zeros_align(&x, pn, 1);
//	double *y; d_zeros_align(&y, pn, 1);
//	x[3] = 1.0;

	d_cvt_mat2pmat(n, n, bs-n%bs, bs, C, n, pC+((bs-n%bs))%bs*(bs+1), pn);
	d_print_pmat(pn, pn, bs, pC, pn);

	/* timing */
	struct timeval tv0, tv1;
	gettimeofday(&tv0, NULL); // start
	
/*	d_print_pmat(n, n, bs, pC, pn);*/

	for(rep=0; rep<nrep; rep++)
		{

/*		dgemm_ppp_nt_lib(n, n, n, pA, pn, pB, pn, pC, pn, 0);*/
/*		dgemm_ppp_nt_lib(n, n, n, pB, pn, pA, pn, pC, pn, 0);*/
		dtrmm_pup_nn_lib(n, n, pA, pn, B, n, pC, pn);




/*		dgemm_pup_nn_lib(n, n, n, pA, pn, B, n, pC, pn, 0);*/
/*		dgemm_ppp_nt_lib(n, n, n, pA, pn, pA, pn, pC+(bs-n)*(bs+1), pn, 1);*/
/*		d_print_pmat(pn, pn, bs, pC, pn);*/
/*		dpotrf_p_dcopy_u_lib(n, (bs-n%bs)%bs, pC+((bs-n%bs))%bs*(bs+1), pn, L, n);*/
/*		d_print_pmat(pn, pn, bs, pC, pn);*/
/*		d_print_mat(n, n, L, n);*/
/*		exit(2);*/

//		dgemm_nt_lib(n, n, n, A, n, B, n, C, n, 0);
//		dgemm_nt_lib_asm(n, n, n, pA, pn, pB, pn, pC, pn, 0);
//		sgemm_nt_lib_neon(n, n, n, spA, pns, spB, pns, spC, pns, 0);
//		dsymm_nt_lib(n, n, A, n, B, n, C, n);
//		dpotrf_lib(n, B, n);
//		dgemm_nt_lib2(n, pB, pA, pC, pn);
//		dgemv_n_lib(n-1, n, 1, pn, pA+1, x, y);
//		dtrmv_n_lib(n-1, 1, pA+1, pn, x, y);
/*		dtrmv_t_lib(n-1, 1, pA+1, pn, x, y);*/

		}
	
	gettimeofday(&tv1, NULL); // stop

	float time = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
	float flop = 2.0*n*n*n;
//	float flop = 1.0/3.0*n*n*n;
	float Gflops = 1e-9*flop/time;
	float Gflops_max = 1*1;
	
	printf("\nn\tGflops\t\t%%\n%d\t%f\t%f\n\n", n, Gflops, 100.0*Gflops/Gflops_max);

	if(n<=20)
		{
//		d_print_pmat(pn, pn, bs, pC, pn);
//		d_print_pmat(n, n, bs, pB, pn);
		d_print_pmat(n, n, bs, pA, pn);
		d_print_mat(n, n, B, n);
/*		d_print_pmat(n, n, bs, pB, pn);*/
		d_print_pmat(n, n, bs, pC, pn);
//		s_print_pmat(n, n, bss, spC, pns);
/*		d_print_mat(n, 1, y, pn);*/
		}





	return 0;
	
	}
