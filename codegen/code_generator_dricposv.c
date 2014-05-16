/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014 by Technical University of Denmark. All rights reserved.                     *
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

#include <stdio.h>

#include "../problem_size.h"
#include "blas_d_codegen.h"
#include "../include/block_size.h"

#define NZ NX+NU+1
#define PNZ D_MR*((NZ+D_MR-NU%D_MR+D_MR-1)/D_MR);



void main()
	{
	
	const int bs = D_MR;
	
	int jj;
	
	int nx = NX;
	int nu = NU;
	int N = NN;
	
	int nz  = nx+nu+1;
	int nxu  = nx+nu;
	int sda = PNZ;
	
	FILE *f;
    f = fopen("dricposv_codegen.c", "w"); // a

	fprintf(f, "/**************************************************************************************************\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* This file is part of HPMPC.                                                                     *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *\n");
	fprintf(f, "* Copyright (C) 2014 by Technical University of Denmark. All rights reserved.                     *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* HPMPC is free software; you can redistribute it and/or                                          *\n");
	fprintf(f, "* modify it under the terms of the GNU Lesser General Public                                      *\n");
	fprintf(f, "* License as published by the Free Software Foundation; either                                    *\n");
	fprintf(f, "* version 2.1 of the License, or (at your option) any later version.                              *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* HPMPC is distributed in the hope that it will be useful,                                        *\n");
	fprintf(f, "* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *\n");
	fprintf(f, "* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *\n");
	fprintf(f, "* See the GNU Lesser General Public License for more details.                                     *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* You should have received a copy of the GNU Lesser General Public                                *\n");
	fprintf(f, "* License along with HPMPC; if not, write to the Free Software                                    *\n");
	fprintf(f, "* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "**************************************************************************************************/\n");
	fprintf(f, "\n");
	fprintf(f, "#include <stdlib.h>\n");
	fprintf(f, "#include <stdio.h>\n");
	fprintf(f, "\n");
	fprintf(f, "#include \"../include/aux_d.h\"\n");
	fprintf(f, "#include \"../include/kernel_d_lib2.h\"\n");
	fprintf(f, "#include \"../include/kernel_d_lib4.h\"\n");
	fprintf(f, "\n");
	fprintf(f, "void dricposv_mpc(int nx, int nu, int N, int sda, double **hpBAbt, double **hpQ, double **hux, double *pL, double *pBAbtL, int compute_pi, double **hpi, int *info)\n");
	fprintf(f, "	{\n");
	fprintf(f, "	if(!(nx==%d && nu==%d && N==%d))\n", nx, nu, N);
	fprintf(f, "		{\n");
	fprintf(f, "		printf(\"\\nError: solver not generated for that problem size\\n\\n\");\n");
	fprintf(f, "		exit(1);\n");
	fprintf(f, "		}\n");
	fprintf(f, "	\n");
	fprintf(f, "	double *pA, *pB, *pC, *x, *y;\n");
	fprintf(f, "	\n");
	fprintf(f, "	int i, j, k, ii, jj, kk;\n");
	fprintf(f, "	\n");
	fprintf(f, "	/* factorization and backward substitution */\n");
	fprintf(f, "	\n");
	fprintf(f, "	/* final stage */\n");
	fprintf(f, "	\n");
	fprintf(f, "	/* dpotrf */\n");
	fprintf(f, "	pC = hpQ[%d];\n", N);

	dpotrf_p_code_generator(f, nz, nu);

	fprintf(f, "	\n");
	fprintf(f, "	d_transpose_pmat_lo(%d, %d, hpQ[%d]+%d, %d, pL, %d);\n", nx, nu, N, (nu/bs)*bs*sda+nu%bs+nu*bs, sda, sda);
	fprintf(f, "	\n");
	fprintf(f, "	/* middle stages */\n");
	fprintf(f, "	for(ii=0; ii<N-1; ii++)\n");
	fprintf(f, "		{\n");
	fprintf(f, "		\n");
	fprintf(f, "		/* dtrmm */\n");
	fprintf(f, "		pA = hpBAbt[%d-ii];\n", N-1);
	fprintf(f, "		pB = pL;\n");
	fprintf(f, "		pC = pBAbtL;\n");

	dtrmm_ppp_code_generator(f, nz, nx, nu);

	fprintf(f, "		\n");
	fprintf(f, "		/* */\n");
	fprintf(f, "		for(jj=0; jj<%d; jj+=4)\n", nx-3);
	fprintf(f, "			{\n");
	fprintf(f, "			pBAbtL[%d+%d*(jj+0)] += hpQ[%d-ii][%d+%d*(jj+%d)];\n", ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, N, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, nu+0);
	fprintf(f, "			pBAbtL[%d+%d*(jj+1)] += hpQ[%d-ii][%d+%d*(jj+%d)];\n", ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, N, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, nu+1);
	fprintf(f, "			pBAbtL[%d+%d*(jj+2)] += hpQ[%d-ii][%d+%d*(jj+%d)];\n", ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, N, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, nu+2);
	fprintf(f, "			pBAbtL[%d+%d*(jj+3)] += hpQ[%d-ii][%d+%d*(jj+%d)];\n", ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, N, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, nu+3);
	fprintf(f, "			}\n");
	for(jj=0; jj<nx%4; jj++)
		{
	fprintf(f, "		pBAbtL[%d] += hpQ[%d-ii][%d];\n", ((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*((nx/4)*4+jj), N, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*((nx/4)*4+nu+jj));
		}
	fprintf(f, "		\n");
	fprintf(f, "		/* dsyrk */\n");
	fprintf(f, "		pA = pBAbtL;\n");
	fprintf(f, "		pC = hpQ[%d-ii];\n", N-1);

	dsyrk_ppp_code_generator(f, nz, nz, nx);

	fprintf(f, "		\n");
	fprintf(f, "		/* dpotrf */\n");
	fprintf(f, "		pC = hpQ[%d-ii];\n", N-1);

	dpotrf_p_code_generator(f, nz, nu);

	fprintf(f, "		\n");
	fprintf(f, "		d_transpose_pmat_lo(%d, %d, hpQ[%d-ii]+%d, %d, pL, %d);\n", nx, nu, N-1, (nu/bs)*bs*sda+nu%bs+nu*bs, sda, sda);
	fprintf(f, "		\n");
	fprintf(f, "		}\n");
	fprintf(f, "	\n");
	fprintf(f, "	/* initial stage */\n");
	fprintf(f, "	\n");
	fprintf(f, "		/* dtrmm */\n");
	fprintf(f, "		pA = hpBAbt[0];\n");
	fprintf(f, "		pB = pL;\n");
	fprintf(f, "		pC = pBAbtL;\n");

	dtrmm_ppp_code_generator(f, nz, nx, nu);

	fprintf(f, "		\n");
	fprintf(f, "		/* */\n");
	fprintf(f, "		for(jj=0; jj<%d; jj+=4)\n", nx-3);
	fprintf(f, "			{\n");
	fprintf(f, "			pBAbtL[%d+%d*(jj+0)] += hpQ[1][%d+%d*(jj+%d)];\n", ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, nu+0);
	fprintf(f, "			pBAbtL[%d+%d*(jj+1)] += hpQ[1][%d+%d*(jj+%d)];\n", ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, nu+1);
	fprintf(f, "			pBAbtL[%d+%d*(jj+2)] += hpQ[1][%d+%d*(jj+%d)];\n", ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, nu+2);
	fprintf(f, "			pBAbtL[%d+%d*(jj+3)] += hpQ[1][%d+%d*(jj+%d)];\n", ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, nu+3);
	fprintf(f, "			}\n");
	for(jj=0; jj<nx%4; jj++)
		{
	fprintf(f, "		pBAbtL[%d] += hpQ[1][%d];\n", ((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*((nx/4)*4+jj), ((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*((nx/4)*4+nu+jj));
		}
	fprintf(f, "		\n");
	fprintf(f, "		/* dsyrk */\n");
	fprintf(f, "		pA = pBAbtL;\n");
	fprintf(f, "		pC = hpQ[0];\n");

	dsyrk_ppp_code_generator(f, nz, nu, nx);

	fprintf(f, "		\n");
	fprintf(f, "		/* dpotrf */\n");
	fprintf(f, "		pC = hpQ[0];\n");

	dpotrf_rec_p_code_generator(f, nz, nu);

	fprintf(f, "	\n");
	fprintf(f, "	\n");
	fprintf(f, "	\n");
/*	fprintf(f, "	d_print_pmat(%d, %d, %d, hpQ[0], %d);\n", nz, nz, bs, sda);*/
/*	fprintf(f, "	d_print_pmat(%d, %d, %d, hpQ[1], %d);\n", nz, nz, bs, sda);*/
/*	fprintf(f, "	d_print_pmat(%d, %d, %d, hpQ[%d], %d);\n", nz, nz, bs, N-1, sda);*/
/*	fprintf(f, "	d_print_pmat(%d, %d, %d, hpQ[%d], %d);\n", nz, nz, bs, N, sda);*/
/*	fprintf(f, "	\n");*/
	fprintf(f, "	/* forward substitution */\n");
	fprintf(f, "	for(ii=0; ii<N; ii++)\n");
	fprintf(f, "		{\n");
	fprintf(f, "		\n");
	fprintf(f, "		/* */\n");
	fprintf(f, "		for(jj=0; jj<%d; jj+=4)\n", nu-3);
	fprintf(f, "			{\n");
	fprintf(f, "			hux[ii][jj+0] = - hpQ[ii][%d+%d*(jj+0)];\n", ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs);
	fprintf(f, "			hux[ii][jj+1] = - hpQ[ii][%d+%d*(jj+1)];\n", ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs);
	fprintf(f, "			hux[ii][jj+2] = - hpQ[ii][%d+%d*(jj+2)];\n", ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs);
	fprintf(f, "			hux[ii][jj+3] = - hpQ[ii][%d+%d*(jj+3)];\n", ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs);
	fprintf(f, "			}\n");
	for(jj=0; jj<nu%4; jj++)
		{
	fprintf(f, "		hux[ii][%d] = - hpQ[ii][%d];\n", (nu/4)*4+jj, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*((nu/4)*4+jj));
		}
	fprintf(f, "		\n");
	fprintf(f, "		/* dgemv_t */\n");
	fprintf(f, "		pA = &hpQ[ii][%d];\n", (nu/bs)*bs*sda+nu%bs);
	fprintf(f, "		x = &hux[ii][%d];\n", nu);
	fprintf(f, "		y = &hux[ii][0];\n");

	dgemv_p_t_code_generator(f, nx, nu, nu, -1);

	fprintf(f, "		\n");
	fprintf(f, "		/* dtrsv_t */\n");
	fprintf(f, "		pA = hpQ[ii];\n");
	fprintf(f, "		x = &hux[ii][0];\n");

	dtrsv_p_t_code_generator(f, nu);

/*	fprintf(f, "		d_print_mat(%d, 1, hux[ii], %d);\n", nx+nu, nx+nu);*/
	fprintf(f, "		\n");
	fprintf(f, "		/* */\n");
	fprintf(f, "		for(jj=0; jj<%d; jj+=4)\n", nx-3);
	fprintf(f, "			{\n");
	fprintf(f, "			hux[ii+1][jj+%d] = hpBAbt[ii][%d+%d*(jj+0)];\n", nu+0, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs);
	fprintf(f, "			hux[ii+1][jj+%d] = hpBAbt[ii][%d+%d*(jj+1)];\n", nu+1, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs);
	fprintf(f, "			hux[ii+1][jj+%d] = hpBAbt[ii][%d+%d*(jj+2)];\n", nu+2, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs);
	fprintf(f, "			hux[ii+1][jj+%d] = hpBAbt[ii][%d+%d*(jj+3)];\n", nu+3, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs);
	fprintf(f, "			}\n");
	for(jj=0; jj<nx%4; jj++)
		{
	fprintf(f, "		hux[ii+1][%d] = hpBAbt[ii][%d];\n", nu+(nx/4)*4+jj, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*((nx/4)*4+jj) );
		}
	fprintf(f, "		\n");
/*	fprintf(f, "		d_print_mat(%d, 1, hux[ii+1], %d);\n", nx+nu, nx+nu);*/
	fprintf(f, "		/* dgemv_t */\n");
	fprintf(f, "		pA = hpBAbt[ii];\n");
	fprintf(f, "		x = &hux[ii][0];\n");
	fprintf(f, "		y = &hux[ii+1][%d];\n", nu);

	dgemv_p_t_code_generator(f, nx+nu, nx, 0, 1);

/*	fprintf(f, "		d_print_mat(%d, 1, hux[ii+1], %d);\n", nx+nu, nx+nu);*/
	fprintf(f, "		\n");
	fprintf(f, "		if(compute_pi)\n");
	fprintf(f, "			{\n");
	fprintf(f, "		\n");
	fprintf(f, "			/* */\n");
	fprintf(f, "			for(jj=0; jj<%d; jj+=4)\n", nx-3);
	fprintf(f, "				{\n");
	fprintf(f, "				pBAbtL[jj+%d] = hpQ[ii+1][%d+%d*(jj+%d)];\n", nu+0, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, nu+0);
	fprintf(f, "				pBAbtL[jj+%d] = hpQ[ii+1][%d+%d*(jj+%d)];\n", nu+1, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, nu+1);
	fprintf(f, "				pBAbtL[jj+%d] = hpQ[ii+1][%d+%d*(jj+%d)];\n", nu+2, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, nu+2);
	fprintf(f, "				pBAbtL[jj+%d] = hpQ[ii+1][%d+%d*(jj+%d)];\n", nu+3, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, nu+3);
	fprintf(f, "				}\n");
	for(jj=0; jj<nx%4; jj++)
		{
	fprintf(f, "			pBAbtL[%d] = hpQ[ii+1][%d];\n", nu+(nx/4)*4+jj, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*(nu+(nx/4)*4+jj) );
		}
	fprintf(f, "		\n");
	fprintf(f, "		/* dtrmv_t */\n");
	fprintf(f, "		pA = hpQ[ii+1]+%d;\n", (nu/bs)*bs*sda+nu%bs+nu*bs);
	fprintf(f, "		x = &hux[ii+1][%d];\n", nu);
	fprintf(f, "		y = &pBAbtL[%d];\n", nu);

	dtrmv_p_t_code_generator(f, nx, nu, 1);

	fprintf(f, "		\n");
	fprintf(f, "		/* dtrmv_n */\n");
	fprintf(f, "		pA = hpQ[ii+1]+%d;\n", (nu/bs)*bs*sda+nu%bs+nu*bs);
	fprintf(f, "		x = &pBAbtL[%d];\n", nu);
	fprintf(f, "		y = &hpi[ii+1][0];\n");

	dtrmv_p_n_code_generator(f, nx, nu, 0);

	fprintf(f, "		\n");
	fprintf(f, "			}\n");
	fprintf(f, "		\n");
/*	fprintf(f, "		exit(2);\n");*/
	fprintf(f, "		}\n");
	fprintf(f, "	\n");
	fprintf(f, "	}\n");
	fprintf(f, "	\n");
	fprintf(f, "	\n");
	fprintf(f, "	\n");
	fprintf(f, "void dricpotrs_mpc(int nx, int nu, int N, int sda, double **hpBAbt, double **hpQ, double **hq, double **hux, double *pBAbtL, int compute_pi, double **hpi)\n");
	fprintf(f, "	{\n");
	fprintf(f, "	if(!(nx==%d && nu==%d && N==%d))\n", nx, nu, N);
	fprintf(f, "		{\n");
	fprintf(f, "		printf(\"\\nError: solver not generated for that problem size\\n\\n\");\n");
	fprintf(f, "		exit(1);\n");
	fprintf(f, "		}\n");
	fprintf(f, "	\n");
	fprintf(f, "	double *pA, *pB, *pC, *x, *y;\n");
	fprintf(f, "	\n");
	fprintf(f, "	int i, j, k, ii, jj, kk;\n");
	fprintf(f, "	\n");
/*	fprintf(f, "	const int bs = D_MR;\n");*/
/*	fprintf(f, "	\n");*/
	fprintf(f, "	/* backward substitution */\n");
	fprintf(f, "	\n");
	fprintf(f, "	for(ii=0; ii<N; ii++)\n");
	fprintf(f, "		{\n");
	fprintf(f, "		\n");
	fprintf(f, "		/* copy b */\n");
	fprintf(f, "		for(jj=0; jj<%d; jj+=4)\n", nx-3);
	fprintf(f, "			{\n");
	fprintf(f, "			pBAbtL[jj+%d] = hpBAbt[%d-ii][%d+%d*(jj+%d)];\n", nu+0, N-1, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, 0);
	fprintf(f, "			pBAbtL[jj+%d] = hpBAbt[%d-ii][%d+%d*(jj+%d)];\n", nu+1, N-1, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, 1);
	fprintf(f, "			pBAbtL[jj+%d] = hpBAbt[%d-ii][%d+%d*(jj+%d)];\n", nu+2, N-1, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, 2);
	fprintf(f, "			pBAbtL[jj+%d] = hpBAbt[%d-ii][%d+%d*(jj+%d)];\n", nu+3, N-1, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs, 3);
	fprintf(f, "			}\n");
	for(jj=0; jj<nx%4; jj++)
		{
	fprintf(f, "		pBAbtL[%d] = hpBAbt[%d-ii][%d];\n", nu+(nx/4)*4+jj, N-1, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*((nx/4)*4+jj) );
		}
	fprintf(f, "		\n");
	fprintf(f, "		/* dtrmv_t */\n");
	fprintf(f, "		pA = hpQ[N-ii]+%d;\n", (nu/bs)*bs*sda+nu%bs+nu*bs);
	fprintf(f, "		x = pBAbtL+%d;\n", nu);
	fprintf(f, "		y = pBAbtL+%d;\n", sda+nu);

	dtrmv_p_t_code_generator(f, nx, nu, 0);

	fprintf(f, "		\n");
	fprintf(f, "		/* copy p */\n");
	fprintf(f, "		for(jj=0; jj<%d; jj+=4)\n", nx-3);
	fprintf(f, "			{\n");
	fprintf(f, "			pBAbtL[jj+%d] = hq[N-ii][jj+%d];\n", nu+0, nu+0);
	fprintf(f, "			pBAbtL[jj+%d] = hq[N-ii][jj+%d];\n", nu+1, nu+1);
	fprintf(f, "			pBAbtL[jj+%d] = hq[N-ii][jj+%d];\n", nu+2, nu+2);
	fprintf(f, "			pBAbtL[jj+%d] = hq[N-ii][jj+%d];\n", nu+3, nu+3);
	fprintf(f, "			}\n");
	for(jj=0; jj<nx%4; jj++)
		{
	fprintf(f, "		pBAbtL[%d] = hq[N-ii][%d];\n", nu+(nx/4)*4+jj, nu+(nx/4)*4+jj );
		}
	fprintf(f, "		\n");
	fprintf(f, "		/* dtrmv_n */\n");
	fprintf(f, "		pA = hpQ[N-ii]+%d;\n", (nu/bs)*bs*sda+nu%bs+nu*bs);
	fprintf(f, "		x = pBAbtL+%d;\n", sda+nu);
	fprintf(f, "		y = pBAbtL+%d;\n", nu);

	dtrmv_p_n_code_generator(f, nx, nu, 1);

	fprintf(f, "		\n");
	fprintf(f, "		/* dgemv_n */\n");
	fprintf(f, "		pA = hpBAbt[%d-ii];\n", N-1);
	fprintf(f, "		x = pBAbtL+%d;\n", nu);
	fprintf(f, "		y = hq[%d-ii];\n", N-1);

	dgemv_p_n_code_generator(f, nx+nu, nx, 0, 1);

	fprintf(f, "		\n");
	fprintf(f, "		/* dtrsv_n */\n");
	fprintf(f, "		pA = hpQ[%d-ii];\n", N-1);
	fprintf(f, "		x = hq[%d-ii];\n", N-1);

	dtrsv_p_n_code_generator(f, nu);

	fprintf(f, "		\n");
	fprintf(f, "		/* dgemv_n */\n");
	fprintf(f, "		pA = hpQ[%d-ii]+%d;\n", N-1, (nu/bs)*bs*sda+nu%bs);
	fprintf(f, "		x = hq[%d-ii];\n", N-1);
	fprintf(f, "		y = hq[%d-ii]+%d;\n", N-1, nu);

	dgemv_p_n_code_generator(f, nx, nu, nu, -1);
	fprintf(f, "		\n");
	fprintf(f, "		}\n");
	fprintf(f, "	\n");
	fprintf(f, "	\n");
	fprintf(f, "	\n");
	fprintf(f, "	/* forward substitution */\n");
	fprintf(f, "	for(ii=0; ii<N; ii++)\n");
	fprintf(f, "		{\n");
	fprintf(f, "		\n");
	fprintf(f, "		/* */\n");
	fprintf(f, "		for(jj=0; jj<%d; jj+=4)\n", nu-3);
	fprintf(f, "			{\n");
	fprintf(f, "			hux[ii][jj+0] = - hq[ii][jj+0];\n");
	fprintf(f, "			hux[ii][jj+1] = - hq[ii][jj+1];\n");
	fprintf(f, "			hux[ii][jj+2] = - hq[ii][jj+2];\n");
	fprintf(f, "			hux[ii][jj+3] = - hq[ii][jj+3];\n");
	fprintf(f, "			}\n");
	for(jj=0; jj<nu%4; jj++)
		{
	fprintf(f, "		hux[ii][%d] = - hq[ii][%d];\n", (nu/4)*4+jj, (nu/4)*4+jj );
		}
	fprintf(f, "		\n");
	fprintf(f, "		/* dgemv_t */\n");
	fprintf(f, "		pA = &hpQ[ii][%d];\n", (nu/bs)*bs*sda+nu%bs);
	fprintf(f, "		x = &hux[ii][%d];\n", nu);
	fprintf(f, "		y = &hux[ii][0];\n");

	dgemv_p_t_code_generator(f, nx, nu, nu, -1);

	fprintf(f, "		\n");
	fprintf(f, "		/* dtrsv_t */\n");
	fprintf(f, "		pA = hpQ[ii];\n");
	fprintf(f, "		x = &hux[ii][0];\n");

	dtrsv_p_t_code_generator(f, nu);

/*	fprintf(f, "		d_print_mat(%d, 1, hux[ii], %d);\n", nx+nu, nx+nu);*/
	fprintf(f, "		\n");
	fprintf(f, "		/* */\n");
	fprintf(f, "		for(jj=0; jj<%d; jj+=4)\n", nx-3);
	fprintf(f, "			{\n");
	fprintf(f, "			hux[ii+1][jj+%d] = hpBAbt[ii][%d+%d*(jj+0)];\n", nu+0, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs);
	fprintf(f, "			hux[ii+1][jj+%d] = hpBAbt[ii][%d+%d*(jj+1)];\n", nu+1, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs);
	fprintf(f, "			hux[ii+1][jj+%d] = hpBAbt[ii][%d+%d*(jj+2)];\n", nu+2, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs);
	fprintf(f, "			hux[ii+1][jj+%d] = hpBAbt[ii][%d+%d*(jj+3)];\n", nu+3, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs, bs);
	fprintf(f, "			}\n");
	for(jj=0; jj<nx%4; jj++)
		{
	fprintf(f, "		hux[ii+1][%d] = hpBAbt[ii][%d];\n", nu+(nx/4)*4+jj, ((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*((nx/4)*4+jj) );
		}
	fprintf(f, "		\n");
/*	fprintf(f, "		d_print_mat(%d, 1, hux[ii+1], %d);\n", nx+nu, nx+nu);*/
	fprintf(f, "		/* dgemv_t */\n");
	fprintf(f, "		pA = hpBAbt[ii];\n");
	fprintf(f, "		x = &hux[ii][0];\n");
	fprintf(f, "		y = &hux[ii+1][%d];\n", nu);

	dgemv_p_t_code_generator(f, nx+nu, nx, 0, 1);

/*	fprintf(f, "		d_print_mat(%d, 1, hux[ii+1], %d);\n", nx+nu, nx+nu);*/
	fprintf(f, "		\n");
	fprintf(f, "		if(compute_pi)\n");
	fprintf(f, "			{\n");
	fprintf(f, "		\n");
	fprintf(f, "		/* dtrmv_t */\n");
	fprintf(f, "		pA = hpQ[ii+1]+%d;\n", (nu/bs)*bs*sda+nu%bs+nu*bs);
	fprintf(f, "		x = &hux[ii+1][%d];\n", nu);
	fprintf(f, "		y = &pBAbtL[%d];\n", nu);

	dtrmv_p_t_code_generator(f, nx, nu, 0);

	fprintf(f, "		\n");
	fprintf(f, "		/* dtrmv_n */\n");
	fprintf(f, "		pA = hpQ[ii+1]+%d;\n", (nu/bs)*bs*sda+nu%bs+nu*bs);
	fprintf(f, "		x = &pBAbtL[%d];\n", nu);
	fprintf(f, "		y = &hpi[ii+1][0];\n");

	dtrmv_p_n_code_generator(f, nx, nu, 0);

	fprintf(f, "		\n");
	fprintf(f, "			/* */\n");
	fprintf(f, "			for(jj=0; jj<%d; jj+=4)\n", nx-3);
	fprintf(f, "				{\n");
	fprintf(f, "				hpi[ii+1][jj+%d] += hq[ii+1][jj+%d];\n", 0, nu+0);
	fprintf(f, "				hpi[ii+1][jj+%d] += hq[ii+1][jj+%d];\n", 1, nu+1);
	fprintf(f, "				hpi[ii+1][jj+%d] += hq[ii+1][jj+%d];\n", 2, nu+2);
	fprintf(f, "				hpi[ii+1][jj+%d] += hq[ii+1][jj+%d];\n", 3, nu+3);
	fprintf(f, "				}\n");
	for(jj=0; jj<nx%4; jj++)
		{
	fprintf(f, "			hpi[ii+1][%d] += hq[ii+1][%d];\n", (nx/4)*4+jj, nu+(nx/4)*4+jj );
		}
	fprintf(f, "		\n");
	fprintf(f, "			}\n");
	fprintf(f, "		\n");
/*	fprintf(f, "		exit(2);\n");*/
	fprintf(f, "		}\n");
	fprintf(f, "	\n");
	fprintf(f, "	}\n");
	fprintf(f, "	\n");

    fclose(f);
	



    f = fopen("dres_codegen.c", "w"); // a

	fprintf(f, "/**************************************************************************************************\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* This file is part of HPMPC.                                                                     *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *\n");
	fprintf(f, "* Copyright (C) 2014 by Technical University of Denmark. All rights reserved.                     *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* HPMPC is free software; you can redistribute it and/or                                          *\n");
	fprintf(f, "* modify it under the terms of the GNU Lesser General Public                                      *\n");
	fprintf(f, "* License as published by the Free Software Foundation; either                                    *\n");
	fprintf(f, "* version 2.1 of the License, or (at your option) any later version.                              *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* HPMPC is distributed in the hope that it will be useful,                                        *\n");
	fprintf(f, "* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *\n");
	fprintf(f, "* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *\n");
	fprintf(f, "* See the GNU Lesser General Public License for more details.                                     *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* You should have received a copy of the GNU Lesser General Public                                *\n");
	fprintf(f, "* License along with HPMPC; if not, write to the Free Software                                    *\n");
	fprintf(f, "* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *\n");
	fprintf(f, "*                                                                                                 *\n");
	fprintf(f, "**************************************************************************************************/\n");
	fprintf(f, "\n");
	fprintf(f, "#include <stdlib.h>\n");
	fprintf(f, "#include <stdio.h>\n");
	fprintf(f, "\n");
	fprintf(f, "#include \"../include/kernel_d_lib2.h\"\n");
	fprintf(f, "#include \"../include/kernel_d_lib4.h\"\n");
	fprintf(f, "\n");
	fprintf(f, "void dres(int nx, int nu, int N, int sda, double **hpBAbt, double **hpQ, double **hq, double **hux, double **hpi, double **hrq, double **hrb)\n");
	fprintf(f, "	{\n");
	fprintf(f, "	if(!(nx==%d && nu==%d && N==%d))\n", nx, nu, N);
	fprintf(f, "		{\n");
	fprintf(f, "		printf(\"\\nError: solver not generated for that problem size\\n\\n\");\n");
	fprintf(f, "		exit(1);\n");
	fprintf(f, "		}\n");
	fprintf(f, "	\n");
	fprintf(f, "	double *pA, *pB, *pC, *x, *y, *x_n, *y_n, *x_t, *y_t;\n");
	fprintf(f, "	\n");
	fprintf(f, "	int i, j, k, ii, jj, kk;\n");
	fprintf(f, "	\n");
	fprintf(f, "	/* first block */\n");
	fprintf(f, "	\n");
	fprintf(f, "	/* */\n");
	fprintf(f, "	for(jj=0; jj<%d; jj+=4)\n", nu-3);
	fprintf(f, "		{\n");
	fprintf(f, "		hrq[0][jj+%d] = - hq[0][jj+%d];\n", 0, 0);
	fprintf(f, "		hrq[0][jj+%d] = - hq[0][jj+%d];\n", 1, 1);
	fprintf(f, "		hrq[0][jj+%d] = - hq[0][jj+%d];\n", 2, 2);
	fprintf(f, "		hrq[0][jj+%d] = - hq[0][jj+%d];\n", 3, 3);
	fprintf(f, "		}\n");
	for(jj=0; jj<nu%4; jj++)
		{
	fprintf(f, "	hrq[0][%d] = - hq[0][%d];\n", (nu/4)*4+jj, (nu/4)*4+jj );
		}
	fprintf(f, "	\n");
	fprintf(f, "	/* dgemv_t */\n");
	fprintf(f, "	pA = &hpQ[0][%d];\n", (nu/bs)*bs*sda+nu%bs);
	fprintf(f, "	x = &hux[0][%d];\n", nu);
	fprintf(f, "	y = &hrq[0][0];\n");

	dgemv_p_t_code_generator(f, nx, nu, nu, -1);

	fprintf(f, "	\n");
	fprintf(f, "	/* dsymv_t */\n");
	fprintf(f, "	pA = &hpQ[0][0];\n");
	fprintf(f, "	x = &hux[0][0];\n");
	fprintf(f, "	y = &hrq[0][0];\n");
	
	dsymv_p_code_generator(f, nu, 0, -1);

	fprintf(f, "	\n");
	fprintf(f, "	/* dgemv_n */\n");
	fprintf(f, "	pA = &hpBAbt[0][0];\n");
	fprintf(f, "	x = &hpi[1][0];\n");
	fprintf(f, "	y = &hrq[0][0];\n");

	dgemv_p_n_code_generator(f, nu, nx, 0, -1);

	fprintf(f, "	\n");
	fprintf(f, "	/* */\n");
	fprintf(f, "	for(jj=0; jj<%d; jj+=4)\n", nx-3);
	fprintf(f, "		{\n");
	fprintf(f, "		hrb[0][jj+%d] = hux[1][jj+%d] - hpBAbt[0][%d+%d*(jj+%d)];\n", 0, nu+0, (nxu/bs)*bs*sda+nxu%bs, bs, 0);
	fprintf(f, "		hrb[0][jj+%d] = hux[1][jj+%d] - hpBAbt[0][%d+%d*(jj+%d)];\n", 1, nu+1, (nxu/bs)*bs*sda+nxu%bs, bs, 1);
	fprintf(f, "		hrb[0][jj+%d] = hux[1][jj+%d] - hpBAbt[0][%d+%d*(jj+%d)];\n", 2, nu+2, (nxu/bs)*bs*sda+nxu%bs, bs, 2);
	fprintf(f, "		hrb[0][jj+%d] = hux[1][jj+%d] - hpBAbt[0][%d+%d*(jj+%d)];\n", 3, nu+3, (nxu/bs)*bs*sda+nxu%bs, bs, 3);
	fprintf(f, "		}\n");
	for(jj=0; jj<nx%4; jj++)
		{
	fprintf(f, "	hrb[0][%d] = hux[1][%d] - hpBAbt[0][%d];\n", (nx/4)*4+jj, nu+(nx/4)*4+jj, (nxu/bs)*bs*sda+nxu%bs+bs*((nx/4)*4+jj) );
		}
	fprintf(f, "	\n");
	fprintf(f, "	/* dgemv_t */\n");
	fprintf(f, "	pA = &hpBAbt[0][0];\n");
	fprintf(f, "	x = &hux[0][0];\n");
	fprintf(f, "	y = &hrb[0][0];\n");

	dgemv_p_t_code_generator(f, nxu, nx, 0, -1);

	fprintf(f, "	\n");
	fprintf(f, "	/* middle blocks */\n");
	fprintf(f, "	\n");
	fprintf(f, "	for(ii=1; ii<N; ii++)\n");
	fprintf(f, "		{\n");
	fprintf(f, "		\n");
	fprintf(f, "		/* */\n");
	fprintf(f, "		for(jj=0; jj<%d; jj+=4)\n", nu-3);
	fprintf(f, "			{\n");
	fprintf(f, "			hrq[ii][jj+%d] = - hq[ii][jj+%d];\n", 0, 0);
	fprintf(f, "			hrq[ii][jj+%d] = - hq[ii][jj+%d];\n", 1, 1);
	fprintf(f, "			hrq[ii][jj+%d] = - hq[ii][jj+%d];\n", 2, 2);
	fprintf(f, "			hrq[ii][jj+%d] = - hq[ii][jj+%d];\n", 3, 3);
	fprintf(f, "			}\n");
	for(jj=0; jj<nu%4; jj++)
		{
	fprintf(f, "		hrq[ii][%d] = - hq[ii][%d];\n", (nu/4)*4+jj, (nu/4)*4+jj );
		}
	fprintf(f, "		\n");
	fprintf(f, "		/* */\n");
	fprintf(f, "		for(jj=0; jj<%d; jj+=4)\n", nx-3);
	fprintf(f, "			{\n");
	fprintf(f, "			hrq[ii][jj+%d] = hpi[ii][jj+%d] - hq[ii][jj+%d];\n", nu+0, 0, nu+0);
	fprintf(f, "			hrq[ii][jj+%d] = hpi[ii][jj+%d] - hq[ii][jj+%d];\n", nu+1, 1, nu+1);
	fprintf(f, "			hrq[ii][jj+%d] = hpi[ii][jj+%d] - hq[ii][jj+%d];\n", nu+2, 2, nu+2);
	fprintf(f, "			hrq[ii][jj+%d] = hpi[ii][jj+%d] - hq[ii][jj+%d];\n", nu+3, 3, nu+3);
	fprintf(f, "			}\n");
	for(jj=0; jj<nx%4; jj++)
		{
	fprintf(f, "		hrq[ii][%d] = hpi[ii][%d] - hq[ii][%d];\n", nu+(nx/4)*4+jj, (nx/4)*4+jj, nu+(nx/4)*4+jj );
		}
	fprintf(f, "		\n");
	fprintf(f, "		/* dsymv_t */\n");
	fprintf(f, "		pA = &hpQ[ii][0];\n");
	fprintf(f, "		x = &hux[ii][0];\n");
	fprintf(f, "		y = &hrq[ii][0];\n");
	
	dsymv_p_code_generator(f, nxu, 0, -1);

	fprintf(f, "		\n");
	fprintf(f, "		/* */\n");
	fprintf(f, "		for(jj=0; jj<%d; jj+=4)\n", nx-3);
	fprintf(f, "			{\n");
	fprintf(f, "			hrb[ii][jj+%d] = hux[ii+1][jj+%d] - hpBAbt[ii][%d+%d*(jj+%d)];\n", 0, nu+0, (nxu/bs)*bs*sda+nxu%bs, bs, 0);
	fprintf(f, "			hrb[ii][jj+%d] = hux[ii+1][jj+%d] - hpBAbt[ii][%d+%d*(jj+%d)];\n", 1, nu+1, (nxu/bs)*bs*sda+nxu%bs, bs, 1);
	fprintf(f, "			hrb[ii][jj+%d] = hux[ii+1][jj+%d] - hpBAbt[ii][%d+%d*(jj+%d)];\n", 2, nu+2, (nxu/bs)*bs*sda+nxu%bs, bs, 2);
	fprintf(f, "			hrb[ii][jj+%d] = hux[ii+1][jj+%d] - hpBAbt[ii][%d+%d*(jj+%d)];\n", 3, nu+3, (nxu/bs)*bs*sda+nxu%bs, bs, 3);
	fprintf(f, "			}\n");
	for(jj=0; jj<nx%4; jj++)
		{
	fprintf(f, "		hrb[ii][%d] = hux[ii+1][%d] - hpBAbt[ii][%d];\n", (nx/4)*4+jj, nu+(nx/4)*4+jj, (nxu/bs)*bs*sda+nxu%bs+bs*((nx/4)*4+jj) );
		}
	fprintf(f, "		\n");
	fprintf(f, "		/* dmvmv */\n");
	fprintf(f, "		pA = &hpBAbt[ii][0];\n");
	fprintf(f, "		x_n = &hpi[ii+1][0];\n");
	fprintf(f, "		y_n = &hrq[ii][0];\n");
	fprintf(f, "		x_t = &hux[ii][0];\n");
	fprintf(f, "		y_t = &hrb[ii][0];\n");

	dmvmv_p_code_generator(f, nxu, nx, 0, -1);

	fprintf(f, "		\n");
	fprintf(f, "		}\n");
	fprintf(f, "	\n");
	fprintf(f, "	/* last block */\n");
	fprintf(f, "	\n");
	fprintf(f, "	/* */\n");
	fprintf(f, "	for(jj=0; jj<%d; jj+=4)\n", nx-3);
	fprintf(f, "		{\n");
	fprintf(f, "		hrq[N][jj+%d] = hpi[N][jj+%d] - hq[N][jj+%d];\n", nu+0, 0, nu+0);
	fprintf(f, "		hrq[N][jj+%d] = hpi[N][jj+%d] - hq[N][jj+%d];\n", nu+1, 1, nu+1);
	fprintf(f, "		hrq[N][jj+%d] = hpi[N][jj+%d] - hq[N][jj+%d];\n", nu+2, 2, nu+2);
	fprintf(f, "		hrq[N][jj+%d] = hpi[N][jj+%d] - hq[N][jj+%d];\n", nu+3, 3, nu+3);
	fprintf(f, "		}\n");
	for(jj=0; jj<nx%4; jj++)
		{
	fprintf(f, "	hrq[N][%d] = hpi[N][%d] - hq[N][%d];\n", nu+(nx/4)*4+jj, (nx/4)*4+jj, nu+(nx/4)*4+jj );
		}
	fprintf(f, "	\n");
	fprintf(f, "	/* dsymv_t */\n");
	fprintf(f, "	pA = &hpQ[N][%d];\n", (nu/bs)*bs*sda+nu%bs+nu*bs);
	fprintf(f, "	x = &hux[N][%d];\n", nu);
	fprintf(f, "	y = &hrq[N][%d];\n", nu);
	
	dsymv_p_code_generator(f, nx, nu, -1);

	fprintf(f, "	\n");
	fprintf(f, "	\n");
	fprintf(f, "	\n");
	fprintf(f, "	\n");
	fprintf(f, "	}\n");
	fprintf(f, "	\n");

	
    fclose(f);



	return;

	}

