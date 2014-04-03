#include "../include/blas_d.h"
#include "../include/block_size.h"



void dricposv(int nx, int nu, int N, int sda, double **hpBAbt, double **hpQ, double **hux, double *pL, double *pBAbtL)
	{
	
	const int bs = D_MR; //d_get_mr();

	int ii, jj;
	
	int nz = nx+nu+1;

	/* initial Cholesky factorization */
	dpotrf_p_dcopy_p_t_lib(nz, nu, hpQ[N], sda, pL, sda);

/*	d_print_pmat(nz, nz, bs, hpQ[N], sda);*/
/*	d_print_pmat(nz, nz, bs, pL, sda);*/

	/* factorization and backward substitution */
	for(ii=0; ii<N; ii++)
		{	
		dtrmm_ppp_lib(nz, nx, nu, hpBAbt[N-ii-1], sda, pL, sda, pBAbtL, sda);
/*	d_print_pmat(nz, nz, bs, pBAbtL, sda);*/
		dsyrk_ppp_lib(nz, nx, pBAbtL, sda, hpQ[N-ii-1], sda);
/*	d_print_pmat(nz, nz, bs, hpQ[N-ii-1], sda);*/
		dpotrf_p_dcopy_p_t_lib(nz, nu, hpQ[N-ii-1], sda, pL, sda);
/*	d_print_pmat(nz, nz, bs, hpQ[N-ii-1], sda);*/
/*	exit(2);*/
		}


	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = hpQ[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		dgemv_p_t_lib(nx, nu, nu, &hpQ[ii][(nu/bs)*bs*sda+nu%bs], sda, &hux[ii][nu], &hux[ii][0], 1);
		dtrsv_p_t_lib(nu, hpQ[ii], sda, &hux[ii][0]);
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hux[ii][jj];
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		dgemv_p_t_lib(nx+nu, nx, 0, hpBAbt[ii], sda, &hux[ii][0], &hux[ii+1][nu], 1);
		}
	
	}



void dricpotrs(int nx, int nu, int N, int sda, double **hpBAbt, double **hpQ, double **hq, double **hux, double *pBAbtL)
	{
	
	const int bs = D_MR; //d_get_mr();

	int ii, jj;
	
	int nz = nx+nu+1;

	/* backward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nx; jj++) pBAbtL[nu+jj] = hpBAbt[N-ii-1][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj]; // copy b
		for(jj=0; jj<nx; jj++) pBAbtL[sda+nu+jj] = 0; // clean
		dtrmv_p_t_lib(nx, nu, hpQ[N-ii]+(nu/bs)*bs*sda+nu%bs+nu*bs, sda, pBAbtL+nu, pBAbtL+sda+nu); // L'*b
		for(jj=0; jj<nx; jj++) pBAbtL[jj] = hq[N-ii][jj]; // copy p
		dtrmv_p_n_lib(nx, nu, hpQ[N-ii]+(nu/bs)*bs*sda+nu%bs+nu*bs, sda, pBAbtL+sda+nu, pBAbtL); // L*(L'*b) + p
		dgemv_p_n_lib(nx+nu, nx, 0, hpBAbt[N-ii-1], sda, pBAbtL, hq[N-ii-1], 1);
		dtrsv_p_n_lib(nu, hpQ[N-ii-1], sda, hq[N-ii-1]);
		dgemv_p_n_lib(nx, nu, nu, hpQ[N-ii-1]+(nu/bs)*bs*sda+nu%bs, sda, hq[N-ii-1], hq[N-ii-1]+nu, -1);
		}

	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = hpQ[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		dgemv_p_t_lib(nx, nu, nu, &hpQ[ii][(nu/bs)*bs*sda+nu%bs], sda, &hux[ii][nu], &hux[ii][0], 1);
		dtrsv_p_t_lib(nu, hpQ[ii], sda, &hux[ii][0]);
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hux[ii][jj];
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		dgemv_p_t_lib(nx+nu, nx, 0, hpBAbt[ii], sda, &hux[ii][0], &hux[ii+1][nu], 1);
		}
	
	}

