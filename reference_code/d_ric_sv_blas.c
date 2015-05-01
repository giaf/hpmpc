/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014-2015 by Technical University of Denmark. All rights reserved.                *
*                                                                                                 *
* HPMPC is free software; you can redistribute it and/or                                          *
* modify it under the terms of the GNU Lesser General Public                                      *
* License as published by the Free Software Foundation; either                                    *
* version 2.1 of the License, or (at your option) any later version.                              *
*                                                                                                 *
* HPMPC is distributed in the hope that it will be useful,                                        *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *
* See the GNU Lesser General Public License for more details.                                     *
*                                                                                                 *
* You should have received a copy of the GNU Lesser General Public                                *
* License along with HPMPC; if not, write to the Free Software                                    *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *
*                                                                                                 *
* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *
*                                                                                                 *
**************************************************************************************************/

// BLAS
//void dgemm_( char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc );
//void dtrmm_( char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb ); // side='L' if B := alpha*op(A)*B
//void dtrsm_( char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb ); // side='L' if op(A)*X = alpha*B
//void dsyrk_( char *uplo, char *trans, int *n, int *k, double *alpha, double *A, int *lda, double *beta, double *C, int *ldc ); // trans='N' if C := alpha*A*A' + beta*C
//void dgemv_( char *trans, int *m, int *n, double *alpha, double *A, int *lda, double *x, int *incx, double *beta, double *y, int *incy );
//void dtrmv_( char *uplo, char *trans, char *diag, int *n, double *A, int *lda, double *x, int *incx );
//void dtrsv_( char *uplo, char *trans, char *diag, int *n, double *A, int *lda, double *x, int *incx );

// LAPACK
//void dlauum_( char *uplo, int *n, double *A, int *lda, int *info ); // info==0 success
//void dpotrf_( char *uplo, int *n, double *A, int *lda, int *info );
//void dtrtri_( char *uplo, char *diag, int *n, double *A, int *lda, int *info ); // diag='N' if not_unit_triangular

#include <blas.h>
#include <lapack.h>



// TODO return error code
void d_ric_trf_mhe_if_blas(int nx, int nw, int ndN, int N, double **A, double **G, double **Q, double **R, double **AGU, double **Up, double **Ue, double **Ur, double *Ud)
	{

	int ii, jj, nn;

	int nxw = nx+nw;

	int info;
	char c_l = 'L';
	char c_n = 'N';
	char c_r = 'R';
	char c_u = 'U';
	double d_0 = 0.0;
	double d_1 = 1.0;

	for(nn=0; nn<N; nn++)
		{

		//printf("\nn = %d\n\n", nn);

		// copy U to E
		for(jj=0; jj<nx; jj++)
			for(ii=0; ii<=jj; ii++)
				Ue[nn][ii+nx*jj] = Up[nn][ii+nx*jj];
		//d_print_mat(nx, nx, Ue[nn], nx);
		
		// E = U*U'
		dlauum_( &c_u, &nx, Ue[nn], &nx, &info );
		//d_print_mat(nx, nx, Ue[nn], nx);
		
		// E = Q + U*U'
		for(jj=0; jj<nx; jj++)
			for(ii=0; ii<=jj; ii++)
				Ue[nn][ii+nx*jj] += Q[nn][ii+nx*jj];
		//d_print_mat(nx, nx, Ue[nn], nx);
		
		// Ue = chol(E)
		dpotrf_( &c_u, &nx, Ue[nn], &nx, &info );
		//d_print_mat(nx, nx, Ue[nn], nx);
		
		// copy A to AGU
		for(jj=0; jj<nx*nx; jj++)
			AGU[nn][jj] = A[nn][jj];
		//d_print_mat(nx, nx+nw, AGU[nn], nx);
		
		// AUe = A * inv(Ue)
		dtrsm_( &c_r, &c_u, &c_n, &c_n, &nx, &nx, &d_1, Ue[nn], &nx, AGU[nn], &nx );
		//d_print_mat(nx, nx+nw, AGU[nn], nx);

		// copy R to Ur
		for(jj=0; jj<nw; jj++)
			for(ii=0; ii<=jj; ii++)
				Ur[nn][ii+nw*jj] = R[nn][ii+nw*jj];
		//d_print_mat(nw, nw, Ur[nn], nw);

		// Ur = chol(R)
		dpotrf_( &c_u, &nw, Ur[nn], &nw, &info );
		//d_print_mat(nw, nw, Ur[nn], nw);

		// copy G to AGU
		for(jj=0; jj<nx*nw; jj++)
			AGU[nn][nx*nx+jj] = G[nn][jj];
		//d_print_mat(nx, nx+nw, AGU[nn], nx);

		// GUr = G * inv(Ur)
		dtrsm_( &c_r, &c_u, &c_n, &c_n, &nx, &nw, &d_1, Ur[nn], &nw, AGU[nn]+nx*nx, &nx );
		//d_print_mat(nx, nx+nw, AGU[nn], nx);

		// U = AGu * AGu'
		dsyrk_( &c_u, &c_n, &nx, &nxw, &d_1, AGU[nn], &nx, &d_0, Up[nn+1], &nx );
		//d_print_mat(nx, nx, Up[nn+1], nx);
		
		// U = chol(U)
		dpotrf_( &c_u, &nx, Up[nn+1], &nx, &info );
		//d_print_mat(nx, nx, Up[nn+1], nx);
		
		// U = inv(U)
		dtrtri_( &c_u, &c_n, &nx, Up[nn+1], &nx, &info );
		//d_print_mat(nx, nx, Up[nn+1], nx);

		//exit(1);
		
		}

	// copy U to E
	for(jj=0; jj<nx; jj++)
		for(ii=0; ii<=jj; ii++)
			Ue[N][ii+nx*jj] = Up[N][ii+nx*jj];
	
	// E = U*U'
	dlauum_( &c_u, &nx, Ue[N], &nx, &info );

	// E = Q + U*U'
	for(jj=0; jj<nx; jj++)
		for(ii=0; ii<=jj; ii++)
			Ue[N][ii+nx*jj] += Q[N][ii+nx*jj];

	// Ue = chol(E)
	dpotrf_( &c_u, &nx, Ue[N], &nx, &info );

	if(ndN>0)
		{

		// copy D to AGU
		for(jj=0; jj<ndN*nx; jj++)
			AGU[N][jj] = A[N][jj];
		//d_print_mat(nx, nx+nw, AGU[nn], nx);
		
		// AUe = D * inv(Ue)
		dtrsm_( &c_r, &c_u, &c_n, &c_n, &ndN, &nx, &d_1, Ue[N], &nx, AGU[N], &ndN );
		//d_print_mat(nx, nx+nw, AGU[nn], nx);

		// U = AGU * AGU'
		dsyrk_( &c_u, &c_n, &ndN, &nx, &d_1, AGU[N], &ndN, &d_0, Ud, &ndN );
		//d_print_mat(ndN, ndN, Ud, ndN);

		// Ud = chol(Ud)
		dpotrf_( &c_u, &ndN, Ud, &ndN, &info );
		//d_print_mat(ndN, ndN, Ud, ndN);

		}

	//exit(1);

	}



void d_ric_trs_mhe_if_blas(int nx, int nw, int ndN, int N, double **AGU, double **Up, double **Ue, double **Ur, double *Ud, double **q, double **r, double **f, double **xp, double **x, double **w, double **lam, double *work)
	{

	int ii, jj, nn;

	int nxw = nx+nw;

	int info;
	int i_1 = 1;
	char c_l = 'L';
	char c_n = 'N';
	char c_r = 'R';
	char c_t = 'T';
	char c_u = 'U';
	double d_0 = 0.0;
	double d_1 = 1.0;
	double d_m1 = -1.0;

	double *x_temp, *w_temp;
	x_temp = work;
	work += nx;
	w_temp = work;
	work += nw;

	// forward substitution
	for(nn=0; nn<N; nn++)
		{

		for(ii=0; ii<nx; ii++) x_temp[ii] = xp[nn][ii];
		//d_print_mat(1, nx, x_temp, 1);
		//d_print_mat(nx, nx, Up[nn], nx);
		dtrmv_( &c_u, &c_t, &c_n, &nx, Up[nn], &nx, x_temp, &i_1 );
		//d_print_mat(1, nx, x_temp, 1);
		dtrmv_( &c_u, &c_n, &c_n, &nx, Up[nn], &nx, x_temp, &i_1 );
		//d_print_mat(1, nx, x_temp, 1);
		for(ii=0; ii<nx; ii++) x[nn][ii] = - q[nn][ii] + x_temp[ii];
		//d_print_mat(1, nx, x[nn], 1);

		dtrsv_( &c_u, &c_t, &c_n, &nx, Ue[nn], &nx, x[nn], &i_1 );
		//d_print_mat(1, nx, x[nn], 1);
		//for(ii=0; ii<nx; ii++) x[nn][ii] = - x[nn][ii];
		//d_print_mat(1, nx, x[nn], 1);
		for(ii=0; ii<nx; ii++) xp[nn+1][ii] = f[nn][ii];
		//d_print_mat(1, nx, xp[nn+1], 1);
		dgemv_( &c_n, &nx, &nx, &d_1, AGU[nn], &nx, x[nn], &i_1, &d_1, xp[nn+1], &i_1 );
		//d_print_mat(1, nx, xp[nn+1], 1);

		for(ii=0; ii<nw; ii++) w[nn][ii] = r[nn][ii];
		dtrsv_( &c_u, &c_t, &c_n, &nw, Ur[nn], &nw, w[nn], &i_1 );
		//d_print_mat(1, nw, w[nn], 1);
		dgemv_( &c_n, &nx, &nw, &d_m1, AGU[nn]+nx*nx, &nx, w[nn], &i_1, &d_1, xp[nn+1], &i_1 );
		//d_print_mat(1, nx, xp[nn+1], 1);

		//if(nn==1)
		//return;
		//exit(1);

		}

	//d_print_mat(1, nx, xp[N], 1);
	for(ii=0; ii<nx; ii++) x_temp[ii] = xp[N][ii];
	//d_print_mat(1, nx, x_temp, 1);
	//d_print_mat(nx, nx, Up[nn], nx);
	dtrmv_( &c_u, &c_t, &c_n, &nx, Up[N], &nx, x_temp, &i_1 );
	//d_print_mat(1, nx, x_temp, 1);
	dtrmv_( &c_u, &c_n, &c_n, &nx, Up[N], &nx, x_temp, &i_1 );
	//d_print_mat(1, nx, x_temp, 1);
	for(ii=0; ii<nx; ii++) x[N][ii] = - q[N][ii] + x_temp[ii];
	//d_print_mat(1, nx, x[N], 1);


	// backwars substitution
	if(ndN<=0)
		{
		dtrsv_( &c_u, &c_t, &c_n, &nx, Ue[N], &nx, x[N], &i_1 );
		//d_print_mat(1, nx, x[N], 1);
		dtrsv_( &c_u, &c_n, &c_n, &nx, Ue[N], &nx, x[N], &i_1 );
		//d_print_mat(1, nx, x[N], 1);
		}
	else
		{
		//d_print_mat(nx, nx, Ue[N], nx);
		//d_print_mat(ndN, nx, AGU[N], ndN);
		dtrsv_( &c_u, &c_t, &c_n, &nx, Ue[N], &nx, x[N], &i_1 );
		//d_print_mat(1, nx, x[N], 1);
		for(ii=0; ii<ndN; ii++) lam[N][ii] = f[N][ii];
		//d_print_mat(1, ndN, lam[N], 1);
		dgemv_( &c_n, &ndN, &nx, &d_1, AGU[N], &ndN, x[N], &i_1, &d_m1, lam[N], &i_1 );
		//d_print_mat(1, ndN, lam[N], 1);
		dtrsv_( &c_u, &c_t, &c_n, &ndN, Ud, &ndN, lam[N], &i_1 );
		dtrsv_( &c_u, &c_n, &c_n, &ndN, Ud, &ndN, lam[N], &i_1 );
		//d_print_mat(1, ndN, lam[N], 1);

		//d_print_mat(1, nx, x[N], 1);
		dgemv_( &c_t, &ndN, &nx, &d_m1, AGU[N], &ndN, lam[N], &i_1, &d_1, x[N], &i_1 );
		dtrsv_( &c_u, &c_n, &c_n, &nx, Ue[N], &nx, x[N], &i_1 );
		//d_print_mat(1, nx, x[N], 1);

		}
	
	for(nn=0; nn<N; nn++)
		{

		for(ii=0; ii<nx; ii++) lam[N-nn-1][ii] = xp[N-nn][ii] - x[N-nn][ii];
		//d_print_mat(1, nx, lam[N-nn-1], 1);
		dtrmv_( &c_u, &c_t, &c_n, &nx, Up[N-nn], &nx, lam[N-nn-1], &i_1 );
		//d_print_mat(1, nx, lam[N-nn-1], 1);
		dtrmv_( &c_u, &c_n, &c_n, &nx, Up[N-nn], &nx, lam[N-nn-1], &i_1 );
		//d_print_mat(1, nx, lam[N-nn-1], 1);

		dgemv_( &c_t, &nx, &nx, &d_m1, AGU[N-nn-1], &nx, lam[N-nn-1], &i_1, &d_1, x[N-nn-1], &i_1 );
		dtrsv_( &c_u, &c_n, &c_n, &nx, Ue[N-nn-1], &nx, x[N-nn-1], &i_1 );
		//d_print_mat(1, nx, x[N-nn-1], 1);

		dgemv_( &c_t, &nx, &nw, &d_m1, AGU[N-nn-1]+nx*nx, &nx, lam[N-nn-1], &i_1, &d_m1, w[N-nn-1], &i_1 );
		dtrsv_( &c_u, &c_n, &c_n, &nw, Ur[N-nn-1], &nw, w[N-nn-1], &i_1 );
		//d_print_mat(1, nw, w[N-nn-1], 1);

		}

	//d_print_mat(1, nw, w[0], 1);
	//exit(1);

	return;

	}



void d_ric_sv_mpc_blas(int nx, int nu, int N, double **BAbt, double **Q, double **Lp, double *BAbtL)
	{

	int ii, jj, nn;
	
	int nz  = nx+nu+1;
	int nx1 = nx+1;
	
	int info;
	char c_l = 'L';
	char c_n = 'N';
	char c_r = 'R';
	char c_t = 'T';
	char c_u = 'U';
	double d_0 = 0.0;
	double d_1 = 1.0;

	// copy P_N
	for(jj=0; jj<nx; jj++)
		for(ii=jj; ii<nx; ii++)
			Lp[N][nu+ii+nz*(nu+jj)] = Q[N][nu+ii+nz*(nu+jj)];

	// L_N = chol(P_N)
	dpotrf_( &c_l, &nx, Lp[N], &nx, &info );

	for(nn=0; nn<N-1; nn++)
		{

		// copy BAbt
		for(ii=0; ii<nx*nz; ii++)
			BAbtL[ii] = BAbt[N-nn-1][ii];

		// BAbtL = BAbt * L
		dtrmm_( &c_r, &c_l, &c_n, &c_n, &nz, &nx, &d_1, Lp[N-nn]+nu*(nz+1), &nz, BAbtL, &nz );
		for(ii=0; ii<nx; ii++) BAbtL[nx+nu+ii*nz] += Lp[N-nn][nx+nu+(nu+ii)*nz];

		// copy P_N
		for(jj=0; jj<nz; jj++)
			for(ii=jj; ii<nz; ii++)
				Lp[N-nn-1][ii+nz*jj] = Q[N-nn-1][ii+nz*jj];

		// Q_n += BAbtL * BAbtL'
		dsyrk_( &c_l, &c_n, &nz, &nx, &d_1, BAbtL, &nz, &d_1, Lp[N-nn-1], &nz );

		// L_n = chol(Q_n)
		dpotrf_( &c_l, &nz, Lp[N-nn-1], &nz, &info );

		}
	
	// copy BAbt
	for(ii=0; ii<nx*nz; ii++)
		BAbtL[ii] = BAbt[0][ii];

	// BAbtL = BAbt * L
	dtrmm_( &c_r, &c_l, &c_n, &c_n, &nz, &nx, &d_1, Lp[1]+nu*(nz+1), &nz, BAbtL, &nz );
	for(ii=0; ii<nx; ii++) BAbtL[nx+nu+ii*nz] += Lp[1][nx+nu+(nu+ii)*nz];

	// copy P_N
	for(jj=0; jj<nu; jj++)
		for(ii=jj; ii<nz; ii++)
			Lp[0][ii+nz*jj] = Q[0][ii+nz*jj];

	// Q_n += BAbtL * BAbtL'
	dsyrk_( &c_l, &c_n, &nu, &nx, &d_1, BAbtL, &nz, &d_1, Lp[0], &nz );
	dgemm_( &c_n, &c_t, &nx1, &nu, &nx, &d_1, BAbtL+nu, &nz, BAbtL, &nz, &d_1, Lp[0]+nu, &nz );

	// L_n = chol(Q_n)
	dpotrf_( &c_l, &nu, Lp[0], &nz, &info );
	dtrsm_( &c_r, &c_l, &c_t, &c_n, &nx1, &nu, &d_1, Lp[0], &nz, Lp[0]+nu, &nz );

	// TODO forward loop

	}


