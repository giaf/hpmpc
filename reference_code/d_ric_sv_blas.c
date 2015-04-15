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
void dtrsm_ ( char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb ); // side='L' if op(A)*X = alpha*B
void dsyrk_ ( char *uplo, char *trans, int *n, int *k, double *alpha, double *A, int *lda, double *beta, double *C, int *ldc ); // trans='N' if C := alpha*A*A' + beta*C

// LAPACK
void dlauum_( char *uplo, int *n, double *A, int *lda, int *info ); // info==0 success
void dpotrf_( char *uplo, int *n, double *A, int *lda, int *info );
void dtrtri_( char *uplo, char *diag, int *n, double *A, int *lda, int *info ); // diag='N' if not_unit_triangular



void d_ric_trf_mhe_if_blas(int nx, int nw, int ndN, int N, double **A, double **G, double **Q, double **R, double **AGU, double **Up, double **Ue, double **Ur, double *Ud)
	{

	int ii, jj, nn;

	int nxw = nx+nw;

	int info;
	char c_u = 'U';
	char c_l = 'L';
	char c_r = 'R';
	char c_n = 'N';
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
		//d_print_mat(nx, nx, Q[nn], nx);
		
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

		//if(nn==1)
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
		//exit(1);

		}


	}
