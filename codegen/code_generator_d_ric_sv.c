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



int main()
	{
	
	int jj;
	
	int nx = NX;
	int nu = NU;
	int N = NN;
	
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl;
	
	const int nz = nx+nu+1;
	const int anz = nal*((nz+nal-1)/nal);
	const int pnz = bs*((nz+bs-1)/bs);
	const int pnx = bs*((nx+bs-1)/bs);
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);

	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	
	FILE *f;
    f = fopen("d_ric_sv_codegen.c", "w"); // a

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
	fprintf(f, "#include \"../include/block_size.h\"\n");
	fprintf(f, "#include \"../include/aux_d.h\"\n");
	fprintf(f, "#include \"../include/kernel_d_lib4.h\"\n");
	fprintf(f, "\n");
	fprintf(f, "// version tailored for mpc (x0 fixed) \n");
	fprintf(f, "void d_ric_sv_mpc(int nx_dummy, int nu_dummy, int N_dummy, double **hpBAbt, double **hpQ, double **hux, double **hpL, double *work, double *diag, int compute_pi, double **hpi)\n");
	fprintf(f, "	{\n");
	fprintf(f, "	if(!(nx_dummy==%d && nu_dummy==%d && N_dummy==%d))\n", nx, nu, N);
	fprintf(f, "		{\n");
	fprintf(f, "		printf(\"\\nError: solver not generated for that problem size\\n\\n\");\n");
	fprintf(f, "		exit(1);\n");
	fprintf(f, "		}\n");
	fprintf(f, "	\n");
	fprintf(f, "	//const int bs = D_MR; //d_get_mr();\n");
	fprintf(f, "	//const int ncl = D_NCL;\n");
	fprintf(f, "	//const int nz = nx+nu+1;\n");
	fprintf(f, "	//const int pnz = bs*((nz+bs-1)/bs);\n");
	fprintf(f, "	//const int pnx = bs*((nx+bs-1)/bs);\n");
	fprintf(f, "	//const int cnz = ncl*((nz+ncl-1)/ncl);\n");
	fprintf(f, "	//const int cnx = ncl*((nx+ncl-1)/ncl);\n");
	fprintf(f, "	//const int pad = (ncl-nx%%ncl)%%ncl; // packing between BAbtL & P\n");
	fprintf(f, "	//const int cnl = nx+pad+cnz;\n");
	fprintf(f, "	\n");
	fprintf(f, "	double *pA, *pB, *pC, *x, *y;\n");
	fprintf(f, "	\n");
	fprintf(f, "	int ii, jj;\n");
	fprintf(f, "	\n");
	fprintf(f, "	double fact[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};\n");
	fprintf(f, "	\n");
	fprintf(f, "	// factorization and backward substitution\n");
	fprintf(f, "	\n");
	fprintf(f, "	// final stage\n");
	fprintf(f, "	\n");
	fprintf(f, "	// dpotrf\n");
	fprintf(f, "	//dsyrk_dpotrf_pp_lib(nx+nu%%bs+1, 0, nx+nu%%bs, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+(nu/bs)*bs*bs, cnl, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, diag);\n");
	fprintf(f, "	pA = hpL[%d]+%d;\n", N, (nx+pad)*bs+(nu/bs)*bs*cnl+(nu/bs)*bs*bs);
	fprintf(f, "	pC = hpQ[%d]+%d;\n", N, (nu/bs)*bs*cnz+(nu/bs)*bs*bs);

	dsyrk_dpotrf_code_generator(f, nx+nu%bs+1, 0, nx+nu%bs);

	fprintf(f, "	\n");
	fprintf(f, "	//d_transpose_pmat_lo(nx, nu, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%%bs+nu*bs, cnl, hpL[N]+(nx+pad+ncl)*bs, cnl);\n");
	fprintf(f, "		pA = hpL[%d]+%d;\n", N, (nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs);
	fprintf(f, "		pC = hpL[%d]+%d;\n", N, (nx+pad+ncl)*bs);
	
	dtrtr_l_code_generator(f, nx, nu);
	
	fprintf(f, "\n");
	fprintf(f, "	// middle stages\n");
	fprintf(f, "	for(ii=0; ii<%d; ii++)\n", N-1);
	fprintf(f, "		{\n");
	fprintf(f, "		//dtrmm_ppp_lib(nz, nx, hpBAbt[N-ii-1], cnx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, hpL[N-ii-1], cnl);\n");
	fprintf(f, "		pA = hpBAbt[%d-ii];\n", N-1);
	fprintf(f, "		pB = hpL[%d-ii]+%d;\n", N, (nx+pad+ncl)*bs);
	fprintf(f, "		pC = hpL[%d-ii];\n", N-1);

	dtrmm_code_generator(f, nz, nx);

	fprintf(f, "		\n");
	fprintf(f, "		for(jj=0; jj<%d; jj++) hpL[%d-ii][%d+jj*%d] += hpL[%d-ii][%d+(%d+jj)*%d];\n", nx, N-1, ((nx+nu)/bs)*bs*cnl+(nx+nu)%bs, bs, N, ((nx+nu)/bs)*bs*cnl+(nx+nu)%bs, nx+pad+nu, bs);
	fprintf(f, "		//dsyrk_dpotrf_pp_lib(nz, nx, nu+nx, hpL[N-ii-1], cnl, hpQ[N-ii-1], cnz, diag);\n");
	fprintf(f, "	pA = hpL[%d-ii];\n", N-1);
	fprintf(f, "	pC = hpQ[%d-ii];\n", N-1);

	dsyrk_dpotrf_code_generator(f, nz, nx, nu+nx);

	fprintf(f, "	\n");
	fprintf(f, "		for(jj=0; jj<%d; jj++) hpL[%d-ii][%d+(jj/%d)*%d+jj%%%d+jj*%d] = diag[jj];\n", nu, N-1, (nx+pad)*bs, bs, bs*cnl, bs, bs);
	fprintf(f, "		//d_transpose_pmat_lo(nx, nu, hpL[N-ii-1]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%%bs+nu*bs, cnl, hpL[N-ii-1]+(nx+pad+ncl)*bs, cnl);\n");
	fprintf(f, "		pA = hpL[%d-ii]+%d;\n", N-1, (nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs);
	fprintf(f, "		pC = hpL[%d-ii]+%d;\n", N-1, (nx+pad+ncl)*bs);
	
	dtrtr_l_code_generator(f, nx, nu);
	
	fprintf(f, "\n");
	fprintf(f, "		}\n");
	fprintf(f, "\n");
	fprintf(f, "	// first stage\n");
	fprintf(f, "	//dtrmm_ppp_lib(nz, nx, hpBAbt[0], cnx, hpL[1]+(nx+pad+ncl)*bs, cnl, hpL[0], cnl);\n");
	fprintf(f, "		pA = hpBAbt[0];\n");
	fprintf(f, "		pB = hpL[1]+%d;\n", (nx+pad+ncl)*bs);
	fprintf(f, "		pC = hpL[0];\n");

	dtrmm_code_generator(f, nz, nx);

	fprintf(f, "		\n");
	fprintf(f, "	for(jj=0; jj<%d; jj++) hpL[0][%d+jj*%d] += hpL[1][%d+(%d+jj)*%d];\n", nx, ((nx+nu)/bs)*bs*cnl+(nx+nu)%bs, bs, ((nx+nu)/bs)*bs*cnl+(nx+nu)%bs, nx+pad+nu, bs);
	fprintf(f, "	//dsyrk_dpotrf_pp_lib(nz, nx, ((nu+2-1)/2)*2, hpL[0], cnl, hpQ[0], cnz, diag);\n");
	fprintf(f, "	pA = hpL[0];\n");
	fprintf(f, "	pC = hpQ[0];\n");

	dsyrk_dpotrf_code_generator(f, nz, nx, ((nu+2-1)/2)*2);

	fprintf(f, "	\n");
	fprintf(f, "	for(jj=0; jj<%d; jj++) hpL[0][%d+(jj/%d)*%d+jj%%%d+jj*%d] = diag[jj];\n", nu, (nx+pad)*bs, bs, bs*cnl, bs, bs);
	fprintf(f, "\n");
	fprintf(f, "	// forward substitution \n");
	fprintf(f, "	for(ii=0; ii<%d; ii++)\n", N);
	fprintf(f, "		{\n");
	fprintf(f, "		for(jj=0; jj<%d; jj++) hux[ii][jj] = - hpL[ii][%d+%d*jj];\n", nu, (nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs, bs);
	fprintf(f, "		//dtrsv_dgemv_t_lib(nx+nu, nu, &hpL[ii][(nx+pad)*bs], cnl, &hux[ii][0]);\n");
	fprintf(f, "	pA = hpL[ii]+%d;\n", (nx+pad)*bs);
	fprintf(f, "	x = hux[ii];\n");

	dtrsv_dgemv_t_code_generator(f, nx+nu, nu);

	fprintf(f, "		\n");
	fprintf(f, "		for(jj=0; jj<%d; jj++) hux[ii+1][%d+jj] = hpBAbt[ii][%d+%d*jj];\n", nx, nu, ((nu+nx)/bs)*bs*cnx+(nu+nx)%bs, bs);
	fprintf(f, "		//dgemv_t_lib(nx+nu, nx, 0, hpBAbt[ii], cnx, &hux[ii][0], &hux[ii+1][nu], 1);\n");
	fprintf(f, "	pA = hpBAbt[ii];\n");
	fprintf(f, "	x = hux[ii];\n");
	fprintf(f, "	y = hux[ii+1]+%d;\n", nu);

	dgemv_t_code_generator(f, nx+nu, nx, 0, cnx, 1);

	fprintf(f, "	\n");
	fprintf(f, "		if(compute_pi)\n");
	fprintf(f, "			{\n");
	fprintf(f, "			for(jj=0; jj<%d; jj++) work[%d+jj] = hux[ii+1][%d+jj];\n", nx, anz, nu);
	fprintf(f, "			for(jj=0; jj<%d; jj++) work[jj] = hpL[ii+1][%d+%d*(%d+jj)];\n", nx, (nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs, bs, nu);
	fprintf(f, "			//dtrmv_p_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &hux[ii+1][nu], &work[0], 1);\n");
	fprintf(f, "	pA = hpL[ii+1]+%d;\n", (nx+pad+ncl)*bs);
	fprintf(f, "	x = work+%d;\n", anz);
	fprintf(f, "	y = work;\n");

	dtrmv_u_n_code_generator(f, nx, 1);

	fprintf(f, "	\n");
	fprintf(f, "			//dtrmv_p_u_t_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[ii+1][0], 0); // L*(L'*b) + p\n");
	fprintf(f, "	pA = hpL[ii+1]+%d;\n", (nx+pad+ncl)*bs);
	fprintf(f, "	x = work;\n");
	fprintf(f, "	y = hpi[ii+1];\n");

	dtrmv_u_t_code_generator(f, nx, 0);

	fprintf(f, "	\n");
	fprintf(f, "			}\n");
	fprintf(f, "		}\n");
	fprintf(f, "	}\n");
	fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "void d_ric_trs_mpc(int nx_dummy, int nu_dummy, int N_dummy, double **hpBAbt, double **hpL, double **hq, double **hux, double *work, int compute_Pb, double ** hPb, int compute_pi, double **hpi)\n");
	fprintf(f, "	{\n");
	fprintf(f, "	if(!(nx_dummy==%d && nu_dummy==%d && N_dummy==%d))\n", nx, nu, N);
	fprintf(f, "		{\n");
	fprintf(f, "		printf(\"\\nError: solver not generated for that problem size\\n\\n\");\n");
	fprintf(f, "		exit(1);\n");
	fprintf(f, "		}\n");
	fprintf(f, "	\n");
	fprintf(f, "	//const int bs = D_MR; //d_get_mr();\n");
	fprintf(f, "	//const int ncl = D_NCL;\n");
	fprintf(f, "	//const int nz = nx+nu+1;\n");
	fprintf(f, "	//const int pnz = bs*((nz+bs-1)/bs);\n");
	fprintf(f, "	//const int pnx = bs*((nx+bs-1)/bs);\n");
	fprintf(f, "	//const int cnz = ncl*((nz+ncl-1)/ncl);\n");
	fprintf(f, "	//const int cnx = ncl*((nx+ncl-1)/ncl);\n");
	fprintf(f, "	//const int pad = (ncl-nx%%ncl)%%ncl; // packing between BAbtL & P\n");
	fprintf(f, "	//const int cnl = nx+pad+cnz;\n");
	fprintf(f, "\n");
	fprintf(f, "	double *pA, *pB, *pC, *x, *y;\n");
	fprintf(f, "	\n");
	fprintf(f, "	int ii, jj;\n");
	fprintf(f, "	\n");
	fprintf(f, "	// backward substitution \n");
	fprintf(f, "	for(ii=0; ii<%d; ii++)\n", N);
	fprintf(f, "		{\n");
	fprintf(f, "		for(jj=0; jj<%d; jj++) work[jj] = hux[%d-ii][%d+jj];\n", nx, N, nu);
	fprintf(f, "		//dtrmv_p_u_n_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work, work+pnz, 0);\n");
	fprintf(f, "	pA = hpL[%d-ii]+%d;\n", N, (nx+pad+ncl)*bs);
	fprintf(f, "	x = work;\n");
	fprintf(f, "	y = work+%d;\n", anz);

	dtrmv_u_n_code_generator(f, nx, 0);

	fprintf(f, "	\n");
	fprintf(f, "		for(jj=0; jj<%d; jj++) work[jj] = hq[%d-ii][%d+jj];\n", nx, N, nu);
	fprintf(f, "		//dtrmv_p_u_t_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work+pnz, work, 1); // L*(L'*b) + p\n");
	fprintf(f, "	pA = hpL[%d-ii]+%d;\n", N, (nx+pad+ncl)*bs);
	fprintf(f, "	x = work+%d;\n", anz);
	fprintf(f, "	y = hPb[%d-ii];\n", N-1);

	dtrmv_u_t_code_generator(f, nx, 0);

	fprintf(f, "	\n");
	fprintf(f, "		for(jj=0; jj<%d; jj++) work[jj] = hPb[%d-ii][jj] + hq[%d-ii][%d+jj];\n", nx, N-1, N, nu);
	fprintf(f, "		//dgemv_p_n_lib(nx+nu, nx, hpBAbt[N-ii-1], cnx, work, hq[N-ii-1], 1);\n");
	fprintf(f, "	pA = hpBAbt[%d-ii];\n", N-1);
	fprintf(f, "	x = work;\n");
	fprintf(f, "	y = hq[%d-ii];\n", N-1);
	
	dgemv_n_code_generator(f, nx+nu, nx, 1);
	
	fprintf(f, "		\n");
	fprintf(f, "		//dtrsv_dgemv_p_n_lib(nu, nu+nx, hpL[N-ii-1]+(nx+pad)*bs, cnl, hq[N-ii-1]);\n");
	fprintf(f, "	pA = hpL[%d-ii]+%d;\n", N-1, (nx+pad)*bs);
	fprintf(f, "	x = hq[%d-ii];\n", N-1);

	dtrsv_dgemv_n_code_generator(f, nu, nu+nx);

	fprintf(f, "		\n");
	fprintf(f, "		}\n");
	fprintf(f, "\n");
	fprintf(f, "	// forward substitution \n");
	fprintf(f, "	for(ii=0; ii<%d; ii++)\n", N);
	fprintf(f, "		{\n");
	fprintf(f, "		for(jj=0; jj<%d; jj++) hux[ii][jj] = - hq[ii][jj];\n", nu);
	fprintf(f, "		//dtrsv_dgemv_p_t_lib(nx+nu, nu, &hpL[ii][(nx+pad)*bs], cnl, &hux[ii][0]);\n");
	fprintf(f, "	pA = hpL[ii]+%d;\n", (nx+pad)*bs);
	fprintf(f, "	x = hux[ii];\n");

	dtrsv_dgemv_t_code_generator(f, nx+nu, nu);

	fprintf(f, "		\n");
//	fprintf(f, "		for(jj=0; jj<%d; jj++) hux[ii+1][%d+jj] = hpBAbt[ii][%d+%d*jj];\n", nx, nu, ((nu+nx)/bs)*bs*cnx+(nu+nx)%bs, bs);
	fprintf(f, "		//dgemv_p_t_lib(nx+nu, nx, 0, hpBAbt[ii], cnx, &hux[ii][0], &hux[ii+1][nu], 1);\n");
	fprintf(f, "	pA = hpBAbt[ii];\n");
	fprintf(f, "	x = hux[ii];\n");
	fprintf(f, "	y = hux[ii+1]+%d;\n", nu);

	dgemv_t_code_generator(f, nx+nu, nx, 0, cnx, 1);

	fprintf(f, "	\n");
	fprintf(f, "		if(compute_pi)\n");
	fprintf(f, "			{\n");
	fprintf(f, "			for(jj=0; jj<%d; jj++) work[%d+jj] = hux[ii+1][%d+jj];\n", nx, anz, nu);
	fprintf(f, "			//dtrmv_p_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &hux[ii+1][nu], &work[0], 0);\n");
	fprintf(f, "	pA = hpL[ii+1]+%d;\n", (nx+pad+ncl)*bs);
	fprintf(f, "	x = work+%d;\n", anz);
	fprintf(f, "	y = work;\n");

	dtrmv_u_n_code_generator(f, nx, 0);

	fprintf(f, "	\n");
	fprintf(f, "			//dtrmv_p_u_t_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[ii+1][0], 0); // L*(L'*b) + p\n");
	fprintf(f, "	pA = hpL[ii+1]+%d;\n", (nx+pad+ncl)*bs);
	fprintf(f, "	x = work;\n");
	fprintf(f, "	y = hpi[ii+1];\n");

	dtrmv_u_t_code_generator(f, nx, 0);

	fprintf(f, "	\n");
	fprintf(f, "			for(jj=0; jj<%d; jj++) hpi[ii+1][jj] += hq[ii+1][%d+jj];\n", nx, nu);
	fprintf(f, "			}\n");
	fprintf(f, "		}\n");
	fprintf(f, "	}\n");
	fprintf(f, "\n");

	fprintf(f, "void d_ric_sv_mhe(int nx_dummy, int nu_dummy, int N_dummy, double **hpBAbt, double **hpQ, double **hux, double **hpL, double *work, double *diag, int compute_pi, double **hpi)\n");
	fprintf(f, "	{\n");
	fprintf(f, "	printf(\"ERROR: not implemented yet!\");\n");
	fprintf(f, "	exit(1);\n");
	fprintf(f, "	}\n");
	fprintf(f, "\n");
	
	fprintf(f, "void d_ric_trs_mhe(int nx_dummy, int nu_dummy, int N_dummy, double **hpBAbt, double **hpL, double **hq, double **hux, double *work, int compute_Pb, double ** hPb, int compute_pi, double **hpi)\n");
	fprintf(f, "	{\n");
	fprintf(f, "	printf(\"ERROR: not implemented yet!\");\n");
	fprintf(f, "	exit(1);\n");
	fprintf(f, "	}\n");
	fprintf(f, "\n");

    fclose(f);
	



    f = fopen("d_res_codegen.c", "w"); // a

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
	fprintf(f, "#include \"../include/block_size.h\"\n");
	fprintf(f, "#include \"../include/kernel_d_lib4.h\"\n");
	fprintf(f, "\n");
	fprintf(f, "void d_res_mpc(int nx_dummy, int nu_dummy, int N_dummy, double **hpBAbt, double **hpQ, double **hq, double **hux, double **hpi, double **hrq, double **hrb)\n");
	fprintf(f, "	{\n");
	fprintf(f, "	if(!(nx_dummy==%d && nu_dummy==%d && N_dummy==%d))\n", nx, nu, N);
	fprintf(f, "		{\n");
	fprintf(f, "		printf(\"\\nError: solver not generated for that problem size\\n\\n\");\n");
	fprintf(f, "		exit(1);\n");
	fprintf(f, "		}\n");
	fprintf(f, "	\n");
	fprintf(f, "	//const int bs = D_MR; //d_get_mr();\n");
	fprintf(f, "	//const int ncl = D_NCL;\n");
	fprintf(f, "	//const int nz = nx+nu+1;\n");
	fprintf(f, "	//const int pnz = bs*((nz+bs-1)/bs);\n");
	fprintf(f, "	//const int pnx = bs*((nx+bs-1)/bs);\n");
	fprintf(f, "	//const int cnz = ncl*((nz+ncl-1)/ncl);\n");
	fprintf(f, "	//const int cnx = ncl*((nx+ncl-1)/ncl);\n");
	fprintf(f, "	//const int pad = (ncl-nx%%ncl)%%ncl; // packing between BAbtL & P\n");
	fprintf(f, "	//const int cnl = nx+pad+cnz;\n");
	fprintf(f, "	//const int nxu = nx+nu;\n");
	fprintf(f, "\n");
	fprintf(f, "	double *pA, *pB, *pC, *x, *y, *x_n, *y_n, *x_t, *y_t;\n");
	fprintf(f, "	\n");
	fprintf(f, "	int ii, jj;\n");
	fprintf(f, "	\n");
	fprintf(f, "	// first block\n");
	fprintf(f, "	for(jj=0; jj<%d; jj++) hrq[0][jj] = - hq[0][jj];\n", nu);
	fprintf(f, "	//dgemv_p_t_lib(nx, nu, nu, hpQ[0]+(nu/bs)*bs*cnz+nu%%bs, cnz, hux[0]+nu, hrq[0], -1);\n");
	fprintf(f, "	pA = hpQ[0]+%d;\n", (nu/bs)*bs*cnz+nu%bs);
	fprintf(f, "	x = hux[0]+%d;\n", nu);
	fprintf(f, "	y = hrq[0];\n");

	dgemv_t_code_generator(f, nx, nu, nu, cnz, -1);

	fprintf(f, "	\n");
	fprintf(f, "	//dsymv_p_lib(nu, 0, hpQ[0], cnz, hux[0], hrq[0], -1);\n");
	fprintf(f, "	pA = hpQ[0];\n");
	fprintf(f, "	x = hux[0];\n");
	fprintf(f, "	y = hrq[0];\n");
	
	dsymv_code_generator(f, nu, 0, -1);

	fprintf(f, "	\n");
	fprintf(f, "	//dgemv_p_n_lib(nu, nx, hpBAbt[0], cnx, hpi[1], hrq[0], -1);\n");
	fprintf(f, "	pA = hpBAbt[0];\n");
	fprintf(f, "	x = hpi[1];\n");
	fprintf(f, "	y = hrq[0];\n");
	
	dgemv_n_code_generator(f, nu, nx, -1);
	
	fprintf(f, "		\n");
	fprintf(f, "	for(jj=0; jj<%d; jj++) hrb[0][jj] = hux[1][%d+jj] - hpBAbt[0][%d+%d*jj];\n", nx, nu, ((nx+nu)/bs)*bs*cnx+(nx+nu)%bs, bs);
	fprintf(f, "	//dgemv_p_t_lib(nxu, nx, 0, hpBAbt[0], cnx, hux[0], hrb[0], -1);\n");
	fprintf(f, "	pA = hpBAbt[0];\n");
	fprintf(f, "	x = hux[0];\n");
	fprintf(f, "	y = hrb[0];\n");

	dgemv_t_code_generator(f, nx+nu, nx, 0, cnx, -1);

	fprintf(f, "	\n");
	fprintf(f, "\n");
	fprintf(f, "	// middle blocks\n");
	fprintf(f, "	for(ii=1; ii<%d; ii++)\n", N);
	fprintf(f, "		{\n");
	fprintf(f, "		for(jj=0; jj<%d; jj++) hrq[ii][jj] = - hq[ii][jj];\n", nu);
	fprintf(f, "		for(jj=0; jj<%d; jj++) hrq[ii][%d+jj] = hpi[ii][jj] - hq[ii][%d+jj];\n", nx, nu, nu);
	fprintf(f, "		//dsymv_p_lib(nxu, 0, hpQ[ii], cnz, hux[ii], hrq[ii], -1);\n");
	fprintf(f, "		pA = hpQ[ii];\n");
	fprintf(f, "		x = hux[ii];\n");
	fprintf(f, "		y = hrq[ii];\n");
	
	dsymv_code_generator(f, nx+nu, 0, -1);

	fprintf(f, "		\n");
	fprintf(f, "		for(jj=0; jj<%d; jj++) hrb[ii][jj] = hux[ii+1][%d+jj] - hpBAbt[ii][%d+%d*jj];\n", nx, nu, ((nx+nu)/bs)*bs*cnx+(nx+nu)%bs, bs);
	fprintf(f, "		//dmvmv_p_lib(nxu, nx, 0, hpBAbt[ii], cnx, hpi[ii+1], hrq[ii], hux[ii], hrb[ii], -1);\n");
	fprintf(f, "		pA = &hpBAbt[ii][0];\n");
	fprintf(f, "		x_n = &hpi[ii+1][0];\n");
	fprintf(f, "		y_n = &hrq[ii][0];\n");
	fprintf(f, "		x_t = &hux[ii][0];\n");
	fprintf(f, "		y_t = &hrb[ii][0];\n");

	dmvmv_code_generator(f, nx+nu, nx, 0, -1);

	fprintf(f, "		\n");
	fprintf(f, "		}\n");
	fprintf(f, "\n");
	fprintf(f, "	// last block\n");
	fprintf(f, "	for(jj=0; jj<%d; jj++) hrq[%d][%d+jj] = hpi[%d][jj] - hq[%d][%d+jj];\n", nx, N, nu, N, N, nu);
	fprintf(f, "	//dsymv_p_lib(nx, nu, hpQ[N]+(nu/bs)*bs*cnz+nu%%bs+nu*bs, cnz, hux[N]+nu, hrq[N]+nu, -1);\n");
	fprintf(f, "	pA = &hpQ[%d][%d];\n", N, (nu/bs)*bs*cnz+nu%bs+nu*bs);
	fprintf(f, "	x = &hux[%d][%d];\n", N, nu);
	fprintf(f, "	y = &hrq[%d][%d];\n", N, nu);
	
	dsymv_code_generator(f, nx, nu, -1);

	fprintf(f, "	\n");
	fprintf(f, "\n");
	fprintf(f, "	}\n");
	fprintf(f, "\n");
	fprintf(f, "void d_res_mhe(int nx_dummy, int nu_dummy, int N_dummy, double **hpBAbt, double **hpQ, double **hq, double **hux, double **hpi, double **hrq, double **hrb)\n");
	fprintf(f, "	{\n");
	fprintf(f, "	printf(\"ERROR: not implemented yet!\");\n");
	fprintf(f, "	exit(1);\n");
	fprintf(f, "	}\n");
	fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "\n");
	fprintf(f, "\n");

	
    fclose(f);



	return 0;

	}

