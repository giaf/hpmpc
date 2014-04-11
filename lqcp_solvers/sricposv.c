#include "../include/blas_s.h"
#include "../include/block_size.h"



void sricposv_orig(int nx, int nu, int N, int sda, float **hpBAbt, float **hpQ, float **hux, float *pL, float *pBAbtL)
	{
	
	const int bs = S_MR; //d_get_mr();

	int ii, jj;
	
	int nz = nx+nu+1;

	/* initial Cholesky factorization */
	spotrf_p_scopy_p_t_lib(nz, nu, hpQ[N], sda, pL, sda);

/*	d_print_pmat(nz, nz, bs, hpQ[N], sda);*/
/*	d_print_pmat(nz, nz, bs, pL, sda);*/

	/* factorization and backward substitution */
	for(ii=0; ii<N; ii++)
		{	
		strmm_ppp_lib(nz, nx, nu, hpBAbt[N-ii-1], sda, pL, sda, pBAbtL, sda);
/*	d_print_pmat(nz, nz, bs, pBAbtL, sda);*/
		ssyrk_ppp_lib(nz, nz, nx, pBAbtL, sda, hpQ[N-ii-1], sda);
/*	d_print_pmat(nz, nz, bs, hpQ[N-ii-1], sda);*/
		spotrf_p_scopy_p_t_lib(nz, nu, hpQ[N-ii-1], sda, pL, sda);
/*	d_print_pmat(nz, nz, bs, hpQ[N-ii-1], sda);*/
/*	exit(2);*/
		}


	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = hpQ[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		sgemv_p_t_lib(nx, nu, nu, &hpQ[ii][(nu/bs)*bs*sda+nu%bs], sda, &hux[ii][nu], &hux[ii][0], 1);
		strsv_p_t_lib(nu, hpQ[ii], sda, &hux[ii][0]);
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hux[ii][jj];
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		sgemv_p_t_lib(nx+nu, nx, 0, hpBAbt[ii], sda, &hux[ii][0], &hux[ii+1][nu], 1);
/*exit(2);	*/
		}
	}



/* version tailoerd for mpc (x0 fixed) */
void sricposv_mpc(int nx, int nu, int N, int sda, float **hpBAbt, float **hpQ, float **hux, float *pL, float *pBAbtL)
	{
	
	const int bs = S_MR; //d_get_mr();

	int ii, jj;
	
	int nz = nx+nu+1;

	/* factorization and backward substitution */

	/* final stage */
/*	dpotrf_p_dcopy_p_t_lib(nz, nu, hpQ[N], sda, pL, sda);*/
	int nu4 = (nu/bs)*bs;
	spotrf_p_scopy_p_t_lib(nz-nu4, nu%bs, hpQ[N]+nu4*(sda+bs), sda, pL, sda);

/*d_print_pmat(nz, nz, bs, hpQ[N], sda);*/

	/* middle stages */
	for(ii=0; ii<N-1; ii++)
		{	
/*d_print_pmat(nz, nz, bs, hpBAbt[N-ii-1], sda);*/
		strmm_ppp_lib(nz, nx, nu, hpBAbt[N-ii-1], sda, pL, sda, pBAbtL, sda);
/*d_print_pmat(nz, nz, bs, pBAbtL, sda);*/
		ssyrk_ppp_lib(nz, nz, nx, pBAbtL, sda, hpQ[N-ii-1], sda);
/*d_print_pmat(nz, nz, bs, hpQ[N-ii-1], sda);*/
		spotrf_p_scopy_p_t_lib(nz, nu, hpQ[N-ii-1], sda, pL, sda);
/*d_print_pmat(nz, nz, bs, hpQ[N-ii-1], sda);*/
/*exit(3);*/
		}

	/* initial stage */
	strmm_ppp_lib(nz, nx, nu, hpBAbt[0], sda, pL, sda, pBAbtL, sda);
/*d_print_pmat(nz, nx, bs, pBAbtL, sda);*/
	ssyrk_ppp_lib(nz, nu, nx, pBAbtL, sda, hpQ[0], sda);
/*d_print_pmat(nz, nu, bs, hpQ[0], sda);*/
	spotrf_p_lib(nz, nu, hpQ[0], sda);
/*d_print_pmat(nz, nu, bs, hpQ[0], sda);*/

/*exit(3);*/

	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = hpQ[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		sgemv_p_t_lib(nx, nu, nu, &hpQ[ii][(nu/bs)*bs*sda+nu%bs], sda, &hux[ii][nu], &hux[ii][0], 1);
/*s_print_mat(nu, 1, &hux[ii][0],nu);*/
		strsv_p_t_lib(nu, hpQ[ii], sda, &hux[ii][0]);
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hux[ii][jj];
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		sgemv_p_t_lib(nx+nu, nx, 0, hpBAbt[ii], sda, &hux[ii][0], &hux[ii+1][nu], 1);
		}
	
/*exit(3);*/

	}



void sricpotrs(int nx, int nu, int N, int sda, float **hpBAbt, float **hpQ, float **hq, float **hux, float *pBAbtL)
	{
	
	const int bs = S_MR; //d_get_mr();

	int ii, jj;
	
	int nz = nx+nu+1;

	/* backward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nx; jj++) pBAbtL[nu+jj] = hpBAbt[N-ii-1][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj]; // copy b
		for(jj=0; jj<nx; jj++) pBAbtL[sda+nu+jj] = 0; // clean
		strmv_p_t_lib(nx, nu, hpQ[N-ii]+(nu/bs)*bs*sda+nu%bs+nu*bs, sda, pBAbtL+nu, pBAbtL+sda+nu); // L'*b
		for(jj=0; jj<nx; jj++) pBAbtL[jj] = hq[N-ii][jj]; // copy p
		strmv_p_n_lib(nx, nu, hpQ[N-ii]+(nu/bs)*bs*sda+nu%bs+nu*bs, sda, pBAbtL+sda+nu, pBAbtL); // L*(L'*b) + p
		sgemv_p_n_lib(nx+nu, nx, 0, hpBAbt[N-ii-1], sda, pBAbtL, hq[N-ii-1], 1);
		strsv_p_n_lib(nu, hpQ[N-ii-1], sda, hq[N-ii-1]);
		sgemv_p_n_lib(nx, nu, nu, hpQ[N-ii-1]+(nu/bs)*bs*sda+nu%bs, sda, hq[N-ii-1], hq[N-ii-1]+nu, -1);
		}

	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = hpQ[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		sgemv_p_t_lib(nx, nu, nu, &hpQ[ii][(nu/bs)*bs*sda+nu%bs], sda, &hux[ii][nu], &hux[ii][0], 1);
		strsv_p_t_lib(nu, hpQ[ii], sda, &hux[ii][0]);
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hux[ii][jj];
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		sgemv_p_t_lib(nx+nu, nx, 0, hpBAbt[ii], sda, &hux[ii][0], &hux[ii+1][nu], 1);
		}
	
	}

