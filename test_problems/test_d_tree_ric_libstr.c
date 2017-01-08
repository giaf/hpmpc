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

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "tools.h"

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_i_aux.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_blas.h>

#include "../include/tree.h" 
#include "../include/lqcp_solvers.h" 



int ipow(int base, int exp)
	{
	int result = 1;
	while(exp)
		{
		if(exp & 1)
			result *= base;
		exp >>= 1;
		base *= base;
		}
	return result;
	}



int get_number_of_nodes(int md, int Nr, int Nh)
	{
	int n_nodes;
	if(md==1) // i.e. standard block-banded structure
		n_nodes = Nh+1;
	else
		n_nodes = (Nh-Nr)*ipow(md,Nr) + (ipow(md,Nr+1)-1)/(md-1);
	return n_nodes;
	}



void print_node(struct node *tree)
	{
	int ii;
	printf("\n");
	printf("idx = \t\t%d\n", tree[0].idx);
	printf("dad = \t\t%d\n", tree[0].dad);
	printf("nkids = \t%d\n", tree[0].nkids);
	printf("kids = \t\t");
	for(ii=0; ii<tree[0].nkids; ii++)
		printf("%d\t", tree[0].kids[ii]);
	printf("\n");
	printf("stage = \t%d\n", tree[0].stage);
	printf("realization = \t%d\n", tree[0].real);
	printf("\n");
	return;
	}



void setup_tree(int md, int Nr, int Nh, int Nn, struct node *tree)
	{
	int ii;
	int idx, dad, stage, real, nkids, idxkid;
	// root
	idx = 0;
	dad = -1;
	stage = 0;
	real = -1;
	if(stage<Nr)
		nkids = md;
	else if(stage<Nh)
		nkids = 1;
	else 
		nkids = 0;
	tree[idx].idx = idx;
	tree[idx].dad = dad;
	tree[idx].stage = stage;
	tree[idx].real = real;
	tree[idx].nkids = nkids;
	if(nkids>0)
		{
		tree[idx].kids = (int *) malloc(nkids*sizeof(int));
		if(nkids>1)
			{
			for(ii=0; ii<nkids; ii++)
				{
				idxkid = ii+1;
				tree[idx].kids[ii] = idxkid;
				tree[idxkid].dad = idx;
				tree[idxkid].real = ii;
				}
			}
		else // nkids==1
			{
			idxkid = 1;
			tree[idx].kids[0] = idxkid;
			tree[idxkid].dad = idx;
			tree[idxkid].real = 0;
			}
		}
	// kids
	for(idx=1; idx<Nn; idx++)
		{
		stage = tree[tree[idx].dad].stage+1;
		if(stage<Nr)
			nkids = md;
		else if(stage<Nh)
			nkids = 1;
		else 
			nkids = 0;
		tree[idx].idx = idx;
		tree[idx].stage = stage;
		tree[idx].nkids = nkids;
		if(nkids>0)
			{
			tree[idx].kids = (int *) malloc(nkids*sizeof(int));
			if(nkids>1)
				{
				for(ii=0; ii<nkids; ii++)
					{
					idxkid = tree[idx-1].kids[tree[idx-1].nkids-1]+ii+1;
					tree[idx].kids[ii] = idxkid;
					tree[idxkid].dad = idx;
					tree[idxkid].real = ii;
					}
				}
			else // nkids==1
				{
				idxkid = tree[idx-1].kids[tree[idx-1].nkids-1]+1;
				tree[idx].kids[0] = idxkid;
				tree[idxkid].dad = idx;
				tree[idxkid].real = tree[idx].real;
				}
			}
		}
	// return
	return;
	}



void free_tree(int md, int Nr, int Nh, int Nn, struct node *tree)
	{
	int ii;
	int idx, dad, stage, real, nkids, idxkid;
	// root
	idx = 0;
	dad = -1;
	stage = 0;
	real = -1;
	if(stage<Nr)
		nkids = md;
	else if(stage<Nh)
		nkids = 1;
	else 
		nkids = 0;
	if(nkids>0)
		{
		free(tree[idx].kids);
		}
	// kids
	for(idx=1; idx<Nn; idx++)
		{
		stage = tree[tree[idx].dad].stage+1;
		if(stage<Nr)
			nkids = md;
		else if(stage<Nh)
			nkids = 1;
		else 
			nkids = 0;
		if(nkids>0)
			{
			free(tree[idx].kids);
			}
		}
	// return
	return;
	}



/************************************************ 
Mass-spring system: nx/2 masses connected each other with springs (in a row), and the first and the last one to walls. nu (<=nx) controls act on the first nu masses. The system is sampled with sampling time Ts. 
************************************************/
void mass_spring_system(double Ts, int nx, int nu, int N, double *A, double *B, double *b, double *x0)
	{

	int nx2 = nx*nx;

	int info = 0;

	int pp = nx/2; // number of masses
	
/************************************************
* build the continuous time system 
************************************************/
	
	double *T; d_zeros(&T, pp, pp);
	int ii;
	for(ii=0; ii<pp; ii++) T[ii*(pp+1)] = -2;
	for(ii=0; ii<pp-1; ii++) T[ii*(pp+1)+1] = 1;
	for(ii=1; ii<pp; ii++) T[ii*(pp+1)-1] = 1;

	double *Z; d_zeros(&Z, pp, pp);
	double *I; d_zeros(&I, pp, pp); for(ii=0; ii<pp; ii++) I[ii*(pp+1)]=1.0; // = eye(pp);
	double *Ac; d_zeros(&Ac, nx, nx);
	dmcopy(pp, pp, Z, pp, Ac, nx);
	dmcopy(pp, pp, T, pp, Ac+pp, nx);
	dmcopy(pp, pp, I, pp, Ac+pp*nx, nx);
	dmcopy(pp, pp, Z, pp, Ac+pp*(nx+1), nx); 
	free(T);
	free(Z);
	free(I);
	
	d_zeros(&I, nu, nu); for(ii=0; ii<nu; ii++) I[ii*(nu+1)]=1.0; //I = eye(nu);
	double *Bc; d_zeros(&Bc, nx, nu);
	dmcopy(nu, nu, I, nu, Bc+pp, nx);
	free(I);
	
/************************************************
* compute the discrete time system 
************************************************/

	double *bb; d_zeros(&bb, nx, 1);
	dmcopy(nx, 1, bb, nx, b, nx);
		
	dmcopy(nx, nx, Ac, nx, A, nx);
	dscal_3l(nx2, Ts, A);
	expm(nx, A);
	
	d_zeros(&T, nx, nx);
	d_zeros(&I, nx, nx); for(ii=0; ii<nx; ii++) I[ii*(nx+1)]=1.0; //I = eye(nx);
	dmcopy(nx, nx, A, nx, T, nx);
	daxpy_3l(nx2, -1.0, I, T);
	dgemm_nn_3l(nx, nu, nx, T, nx, Bc, nx, B, nx);
	free(T);
	free(I);
	
	int *ipiv = (int *) malloc(nx*sizeof(int));
	dgesv_3l(nx, nu, Ac, nx, ipiv, B, nx, &info);
	free(ipiv);

	free(Ac);
	free(Bc);
	free(bb);
	
			
/************************************************
* initial state 
************************************************/
	
	if(nx==4)
		{
		x0[0] = 5;
		x0[1] = 10;
		x0[2] = 15;
		x0[3] = 20;
		}
	else
		{
		int jj;
		for(jj=0; jj<nx; jj++)
			x0[jj] = 1;
		}

	}



int main()
	{

	printf("\nExample of LU factorization and backsolve\n\n");

#if defined(LA_BLASFEO)

	printf("\nLA provided by BLASFEO\n\n");

#elif defined(LA_BLAS)

	printf("\nLA provided by BLAS\n\n");

#else

	printf("\nLA provided by ???\n\n");
	exit(2);

#endif

	// loop index
	int ii, jj;

/************************************************
* problem size
************************************************/	

	// problem size
	int N = 5;
	int nx_ = 4;
	int nu_ = 1;

	// stage-wise variant size
	int nx[N+1];
	nx[0] = 0;
	for(ii=1; ii<=N; ii++)
		nx[ii] = nx_;
	nx[N] = nx_;

	int nu[N+1];
	for(ii=0; ii<N; ii++)
		nu[ii] = nu_;
	nu[N] = 0;

	int nb[N+1];
	for(ii=0; ii<=N; ii++)
		nb[ii] = 0;

	int ng[N+1];
	for(ii=0; ii<=N; ii++)
		ng[ii] = 0;
	
/************************************************
* dynamical system
************************************************/	

	double *A; d_zeros(&A, nx_, nx_); // states update matrix

	double *B; d_zeros(&B, nx_, nu_); // inputs matrix

	double *b; d_zeros(&b, nx_, 1); // states offset
	double *x0; d_zeros_align(&x0, nx_, 1); // initial state

	double Ts = 0.5; // sampling time
	mass_spring_system(Ts, nx_, nu_, N, A, B, b, x0);
	
	for(ii=0; ii<nx_; ii++)
		b[ii] = 0.1;
	
	for(ii=0; ii<nx_; ii++)
		x0[ii] = 0;
	x0[0] = 2.5;
	x0[1] = 2.5;

//	d_print_mat(nx_, nx_, A, nx_);
//	d_print_mat(nx_, nu_, B, nx_);
//	d_print_mat(1, nx_, b, 1);
//	d_print_mat(1, nx_, x0, 1);

/************************************************
* cost function
************************************************/	

	double *R; d_zeros(&R, nu_, nu_);
	for(ii=0; ii<nu_; ii++) R[ii*(nu_+1)] = 2.0;

	double *S; d_zeros(&S, nu_, nx_);

	double *Q; d_zeros(&Q, nx_, nx_);
	for(ii=0; ii<nx_; ii++) Q[ii*(nx_+1)] = 1.0;

	double *r; d_zeros(&r, nu_, 1);
	for(ii=0; ii<nu_; ii++) r[ii] = 0.2;

	double *q; d_zeros(&q, nx_, 1);
	for(ii=0; ii<nx_; ii++) q[ii] = 0.1;

//	d_print_mat(nu_, nu_, R, nu_);
//	d_print_mat(nu_, nx_, S, nu_);
//	d_print_mat(nx_, nx_, Q, nx_);
//	d_print_mat(1, nu_, r, 1);
//	d_print_mat(1, nx_, q, 1);

/************************************************
* matrices as strmat
************************************************/	

	struct d_strmat sA;
	d_allocate_strmat(nx_, nx_, &sA);
	d_cvt_mat2strmat(nx_, nx_, A, nx_, &sA, 0, 0);
	struct d_strvec sb;
	d_allocate_strvec(nx_, &sb);
	d_cvt_vec2strvec(nx_, b, &sb, 0);
	struct d_strvec sx0;
	d_allocate_strvec(nx_, &sx0);
	d_cvt_vec2strvec(nx_, x0, &sx0, 0);
	struct d_strvec sb0;
	d_allocate_strvec(nx_, &sb0);
	double *b0; d_zeros(&b0, nx_, 1); // states offset
	dgemv_n_libstr(nx_, nx_, 1.0, &sA, 0, 0, &sx0, 0, 1.0, &sb, 0, &sb0, 0);
//	d_print_tran_strvec(nx_, &sb0, 0);

	struct d_strmat sBbt0;
	d_allocate_strmat(nu_+nx_+1, nx_, &sBbt0);
	d_cvt_tran_mat2strmat(nx_, nx_, B, nx_, &sBbt0, 0, 0);
	drowin_libstr(nx_, 1.0, &sb0, 0, &sBbt0, nu_, 0);
//	d_print_strmat(nu_+1, nx_, &sBbt0, 0, 0);

	struct d_strmat sBAbt1;
	d_allocate_strmat(nu_+nx_+1, nx_, &sBAbt1);
	d_cvt_tran_mat2strmat(nx_, nu_, B, nx_, &sBAbt1, 0, 0);
	d_cvt_tran_mat2strmat(nx_, nx_, A, nx_, &sBAbt1, nu_, 0);
	d_cvt_tran_mat2strmat(nx_, 1, b, nx_, &sBAbt1, nu_+nx_, 0);
//	d_print_strmat(nu_+nx_+1, nx_, &sBAbt1, 0, 0);

	struct d_strvec sr0; // XXX no need to update r0 since S=0
	d_allocate_strvec(nu_, &sr0);
	d_cvt_vec2strvec(nu_, r, &sr0, 0);

	struct d_strmat sRr0;
	d_allocate_strmat(nu_+1, nu_, &sRr0);
	d_cvt_mat2strmat(nu_, nu_, R, nu_, &sRr0, 0, 0);
	drowin_libstr(nu_, 1.0, &sr0, 0, &sRr0, nu_, 0);
//	d_print_strmat(nu_+1, nu_, &sRr0, 0, 0);

	struct d_strvec srq1;
	d_allocate_strvec(nu_+nx_, &srq1);
	d_cvt_vec2strvec(nu_, r, &srq1, 0);
	d_cvt_vec2strvec(nx_, q, &srq1, nu_);

	struct d_strmat sRSQrq1;
	d_allocate_strmat(nu_+nx_+1, nu_+nx_, &sRSQrq1);
	d_cvt_mat2strmat(nu_, nu_, R, nu_, &sRSQrq1, 0, 0);
	d_cvt_tran_mat2strmat(nu_, nx_, S, nu_, &sRSQrq1, nu_, 0);
	d_cvt_mat2strmat(nx_, nx_, Q, nx_, &sRSQrq1, nu_, nu_);
	drowin_libstr(nu_+nx_, 1.0, &srq1, 0, &sRSQrq1, nu_+nx_, 0);
//	d_print_strmat(nu_+nx_+1, nu_+nx_, &sRSQrq1, 0, 0);

	struct d_strvec sqN;
	d_allocate_strvec(nx_, &sqN);
	d_cvt_vec2strvec(nx_, q, &sqN, 0);

	struct d_strmat sQqN;
	d_allocate_strmat(nx_+1, nx_, &sQqN);
	d_cvt_mat2strmat(nx_, nx_, Q, nx_, &sQqN, 0, 0);
	drowin_libstr(nx_, 1.0, &sqN, 0, &sQqN, nx_, 0);
//	d_print_strmat(nx_+1, nx_, &sQqN, 0, 0);

/************************************************
* array of matrices
************************************************/	
	
	struct d_strmat hsBAbt[N];
	struct d_strvec hsb[N];
	struct d_strmat hsRSQrq[N+1];
	struct d_strvec hsrq[N+1];
	struct d_strmat hsDCt[N+1];
	struct d_strvec hsQx[N+1];
	struct d_strvec hsqx[N+1];
	struct d_strmat hsL[N+1];
	struct d_strmat hsLxt[N+1];
	struct d_strvec hsPb[N+1];
	struct d_strvec hsux[N+1];
	struct d_strvec hspi[N+1];
	struct d_strmat hswork_mat[1];
	struct d_strvec hswork_vec[1];
	int *hidxb[N+1];

	// a different matrix per stage !!!!!
#if 1
	d_allocate_strmat(nu_+1, nx_, &hsBAbt[0]);
	dgecp_libstr(nu_+1, nx_, 1.0, &sBbt0, 0, 0, &hsBAbt[0], 0, 0);
	d_allocate_strvec(nx_, &hsb[0]);
	dveccp_libstr(nx_, 1.0, &sb0, 0, &hsb[0], 0);
	d_allocate_strmat(nu_+nx_+1, nu_+nx_, &hsRSQrq[0]);
	dgecp_libstr(nu_+1, nu_, 1.0, &sRr0, 0, 0, &hsRSQrq[0], 0, 0);
	d_allocate_strvec(nu_, &hsrq[0]);
	dveccp_libstr(nu_, 1.0, &sr0, 0, &hsrq[0], 0);
#else
	hsBAbt[0] = sBbt0;
	hsb[0] = sb0;
	hsRSQrq[0] = sRr0;
	hsrq[0] = sr0;
#endif
	d_allocate_strmat(nu_+1, nu_, &hsL[0]);
//	d_allocate_strmat(nu_+1, nu_, &hsLxt[0]);
	d_allocate_strvec(nx_, &hsPb[1]);
	d_allocate_strvec(nx_+nu_+1, &hsux[0]);
	d_allocate_strvec(nx_, &hspi[1]);
	for(ii=1; ii<N; ii++)
		{
#if 1
		d_allocate_strmat(nu_+nx_+1, nx_, &hsBAbt[ii]);
		dgecp_libstr(nu_+nx_+1, nx_, 1.0, &sBAbt1, 0, 0, &hsBAbt[ii], 0, 0);
		d_allocate_strvec(nx_, &hsb[ii]);
		dveccp_libstr(nx_, 1.0, &sb, 0, &hsb[ii], 0);
		d_allocate_strmat(nu_+nx_+1, nu_+nx_, &hsRSQrq[ii]);
		dgecp_libstr(nu_+nx_+1, nu_+nx_, 1.0, &sRSQrq1, 0, 0, &hsRSQrq[ii], 0, 0);
		d_allocate_strvec(nu_+nx_, &hsrq[ii]);
		dveccp_libstr(nu_+nx_, 1.0, &srq1, 0, &hsrq[ii], 0);
#else
		hsBAbt[ii] = sBAbt1;
		hsb[ii] = sb;
		hsRSQrq[ii] = sRSQrq1;
		hsrq[ii] = srq1;
#endif
		d_allocate_strmat(nu_+nx_+1, nu_+nx_, &hsL[ii]);
		d_allocate_strmat(nx_, nu_+nx_, &hsLxt[ii]);
		d_allocate_strvec(nx_, &hsPb[ii+1]);
		d_allocate_strvec(nx_+nu_+1, &hsux[ii]);
		d_allocate_strvec(nx_, &hspi[ii+1]);
		}
#if 1
	d_allocate_strmat(nx_+1, nx_, &hsRSQrq[N]);
	dgecp_libstr(nx_+1, nx_, 1.0, &sQqN, 0, 0, &hsRSQrq[N], 0, 0);
	d_allocate_strvec(nx_, &hsrq[N]);
	dveccp_libstr(nx_, 1.0, &sqN, 0, &hsrq[N], 0);
#else
	hsRSQrq[N] = sQqN;
	hsrq[N] = sqN;
#endif
	d_allocate_strmat(nx_+1, nx_, &hsL[N]);
	d_allocate_strmat(nx_, nx_, &hsLxt[N]);
	d_allocate_strvec(nx_+nu_+1, &hsux[N]);

	// riccati work space
	void *work_ric;
	v_zeros_align(&work_ric, d_back_ric_rec_work_space_size_bytes_libstr(N, nx, nu, nb, ng));

//	for(ii=0; ii<N; ii++)
//		d_print_strmat(nu[ii]+nx[ii]+1, nx[ii+1], &hsBAbt[ii], 0, 0);
//	return 0;

/************************************************
* call Riccati solver
************************************************/	
	
	// timing 
	struct timeval tv0, tv1, tv2, tv3;
	int nrep = 1000;
	int rep;

//	for(ii=0; ii<=N; ii++)
//		printf("\n%d %d\n", nu[ii], nx[ii]);

	// warm up ???
	for(rep=0; rep<nrep; rep++)
		{
		d_back_ric_rec_trf_libstr(N, nx, nu, nb, hidxb, ng, hsBAbt, hsRSQrq, hsDCt, hsQx, hsL, hsLxt, work_ric);
		}

	gettimeofday(&tv0, NULL); // time

	for(rep=0; rep<nrep; rep++)
		{
//		d_back_ric_sv_libstr(N, nx, nu, hsBAbt, hsRSQrq, hsL, hsLxt, hsux, hspi, hswork_mat, hswork_vec);
		}

	gettimeofday(&tv1, NULL); // time

	for(rep=0; rep<nrep; rep++)
		{
		d_back_ric_rec_trf_libstr(N, nx, nu, nb, hidxb, ng, hsBAbt, hsRSQrq, hsDCt, hsQx, hsL, hsLxt, work_ric);
		}

	gettimeofday(&tv2, NULL); // time

	for(rep=0; rep<nrep; rep++)
		{
		d_back_ric_rec_trs_libstr(N, nx, nu, nb, hidxb, ng, hsBAbt, hsb, hsrq, hsDCt, hsqx, hsux, 1, hspi, 1, hsPb, hsL, hsLxt, work_ric);
		}

	gettimeofday(&tv3, NULL); // time

	float time_sv  = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
	float time_trf = (float) (tv2.tv_sec-tv1.tv_sec)/(nrep+0.0)+(tv2.tv_usec-tv1.tv_usec)/(nrep*1e6);
	float time_trs = (float) (tv3.tv_sec-tv2.tv_sec)/(nrep+0.0)+(tv3.tv_usec-tv2.tv_usec)/(nrep*1e6);

	// print sol
	printf("\nux = \n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(nu[ii]+nx[ii], &hsux[ii], 0);

	printf("\npi = \n\n");
	for(ii=1; ii<=N; ii++)
		d_print_tran_strvec(nx[ii], &hspi[ii], 0);

//	printf("\nL = \n\n");
//	for(ii=0; ii<=N; ii++)
//		d_print_strmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], &hsL[ii], 0, 0);

	printf("\ntime sv\t\ttime trf\t\ttime trs\n");
	printf("\n%e\t%e\t%e\n", time_sv, time_trf, time_trs);
	printf("\n");

/************************************************
* scenario-tree MPC
************************************************/	
	
	int Nh = N; // control horizion
	int Nr = 2; // robust horizion
	int md = 3; // number of realizations

	int Nn = get_number_of_nodes(md, Nr, Nh);
	printf("\nnumber of nodes = %d\n", Nn);

	struct node tree[Nn];

	// setup the tree
printf("\n%d %d %d %d\n", md, Nr, Nh, Nn);
	setup_tree(md, Nr, Nh, Nn, tree);
printf("\nciao\n");

	// print the tree
//	for(ii=0; ii<Nn; ii++)
//		print_node(&tree[ii]);

	// data structure

	int stage;

	// stage-wise variant size (tmp)
	int t_nx[Nn];
	int t_nu[Nn];
	int t_nb[Nn];
	int t_ng[Nn];
	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage;
		t_nx[ii] = nx[stage];
		t_nu[ii] = nu[stage];
		t_nb[ii] = nb[stage];
		t_ng[ii] = ng[stage];
		}

	// dynamics indexed by node (ecluding the root) // no real atm
	struct d_strmat t_hsBAbt[Nn];
	struct d_strvec t_hsb[Nn];
	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage-1;
		if(stage<=N)
			{
			t_hsBAbt[ii] = hsBAbt[stage];
			t_hsb[ii] = hsb[stage];
			}
		}
//	for(ii=1; ii<Nn; ii++)
//		{
//		stage = tree[ii].stage;
//		printf("\nstage = %d\n", stage);
//		d_print_strmat(t_nu[stage-1]+t_nx[stage-1]+1, t_nx[stage], &t_hsBAbt[ii], 0, 0);
//		}
//	return;

	// temporary cost function indexed by stage
	struct d_strmat tmp_hsRSQrq[Nh+1];
	struct d_strvec tmp_hsrq[Nh+1];
	// first stages: scale cost function
	for(ii=0; ii<Nr; ii++)
		{
		d_allocate_strmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], &tmp_hsRSQrq[ii]);
		d_allocate_strvec(nu[ii]+nx[ii], &tmp_hsrq[ii]);
		}
	// last stages: original cost function
	for(ii=Nr; ii<Nh; ii++)
		{
		tmp_hsRSQrq[ii] = sRSQrq1;
		tmp_hsrq[ii] = srq1;
		}
	// last stage
	tmp_hsRSQrq[Nh] = sQqN;
	tmp_hsrq[Nh] = sqN;
	// scale at first stages
	for(ii=Nr-1; ii>0; ii--)
		{
		dgecp_libstr(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], md, &tmp_hsRSQrq[ii+1], 0, 0, &tmp_hsRSQrq[ii], 0, 0);
		dveccp_libstr(nu[ii]+nx[ii], md, &tmp_hsrq[ii+1], 0, &tmp_hsrq[ii], 0);
		}
	// scale first stage
	ii = 0;
	dgecp_libstr(nu[ii]+nx[ii], nu[ii]+nx[ii], md, &tmp_hsRSQrq[ii+1], 0, 0, &tmp_hsRSQrq[ii], 0, 0);
	dgecp_libstr(1, nu[ii]+nx[ii], md, &tmp_hsRSQrq[ii+1], nu[ii+1]+nx[ii+1], 0, &tmp_hsRSQrq[ii], nu[ii]+nx[ii], 0);
	dveccp_libstr(nu[ii]+nx[ii], md, &tmp_hsrq[ii+1], 0, &tmp_hsrq[ii], 0);
//	for(ii=0; ii<=Nh; ii++)
//		{
//		d_print_strmat(t_nu[ii]+t_nx[ii]+1, t_nu[ii]+t_nx[ii], &tmp_hsRSQrq[ii], 0, 0);
//		}
//	return;

	// cost function indexed by node
	struct d_strmat t_hsRSQrq[Nn];
	struct d_strvec t_hsrq[Nn];
	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage;
		t_hsRSQrq[ii] = tmp_hsRSQrq[stage];
		t_hsrq[ii] = tmp_hsrq[stage];
		}

	// store factorization indexed by node
	struct d_strmat t_hsL[Nn];
	for(ii=0; ii<Nn; ii++)
		{
		d_allocate_strmat(t_nu[ii]+t_nx[ii]+1, t_nu[ii]+t_nx[ii], &t_hsL[ii]);
		}
	struct d_strmat t_hsLxt[Nn];
	for(ii=0; ii<Nn; ii++)
		{
		d_allocate_strmat(t_nx[ii], t_nx[ii], &t_hsLxt[ii]);
		}
	struct d_strvec t_hsPb[Nn];
	for(ii=0; ii<Nn; ii++)
		{
		d_allocate_strvec(t_nx[ii], &t_hsPb[ii]);
		}
	
	// solution indexed by node
	struct d_strvec t_hsux[Nn];
	for(ii=0; ii<Nn; ii++)
		{
		d_allocate_strvec(t_nu[ii]+t_nx[ii], &t_hsux[ii]);
		}
	struct d_strvec t_hspi[Nn];
	for(ii=0; ii<Nn; ii++)
		{
		d_allocate_strvec(t_nx[ii], &t_hspi[ii]);
		}
	
	// dummy
	int *t_hidxb[Nn];
	struct d_strmat hsmatdummy[Nn];
	struct d_strvec hsvecdummy[Nn];


	// call riccati
	gettimeofday(&tv1, NULL); // time

	for(rep=0; rep<nrep; rep++)
		{
		d_tree_back_ric_rec_trf_libstr(Nn, tree, t_nx, t_nu, t_nb, t_hidxb, t_ng, t_hsBAbt, t_hsRSQrq, hsmatdummy, hsvecdummy, t_hsL, t_hsLxt, work_ric);
		}

	gettimeofday(&tv2, NULL); // time

	for(rep=0; rep<nrep; rep++)
		{
		d_tree_back_ric_rec_trs_libstr(Nn, tree, t_nx, t_nu, t_nb, t_hidxb, t_ng, t_hsBAbt, t_hsb, t_hsrq, hsmatdummy, hsvecdummy, t_hsux, 1, t_hspi, 1, t_hsPb, t_hsL, t_hsLxt, work_ric);
		}

	gettimeofday(&tv3, NULL); // time

	// print factorization
//	for(ii=0; ii<Nn; ii++)
//		{
//		d_print_strmat(t_nu[ii]+t_nx[ii]+1, t_nu[ii]+t_nx[ii], &t_hsL[ii], 0, 0);
//		}
//	for(ii=0; ii<Nn; ii++)
//		{
//		stage = tree[ii].stage;
//		d_print_strmat(t_nx[stage], t_nx[stage], &t_hsLxt[ii], 0, 0);
//		}
	for(ii=0; ii<Nn; ii++)
		{
		d_print_tran_strvec(t_nu[ii]+t_nx[ii], &t_hsux[ii], 0);
		}
	for(ii=0; ii<Nn; ii++)
		{
		d_print_tran_strvec(t_nx[ii], &t_hspi[ii], 0);
		}

	float time_tree_trf = (float) (tv2.tv_sec-tv1.tv_sec)/(nrep+0.0)+(tv2.tv_usec-tv1.tv_usec)/(nrep*1e6);
	float time_tree_trs = (float) (tv3.tv_sec-tv2.tv_sec)/(nrep+0.0)+(tv3.tv_usec-tv2.tv_usec)/(nrep*1e6);

	printf("\ntime sv\t\ttime trf\t\ttime trs\n");
	printf("\n%e\t%e\t%e\n", 0.0, time_tree_trf, time_tree_trs);
	printf("\n");

/************************************************
* scenario-tree MPC with standard solver
************************************************/	

	int nx2[N+1];
	int nu2[N+1];
	for(ii=0; ii<=N; ii++)
		{
		nx2[ii] = 0;
		nu2[ii] = 0;
		}
	for(ii=Nn-1; ii>=0; ii--)
		{
		stage = tree[ii].stage;
		nx2[stage] += t_nx[ii];
		nu2[stage] += t_nu[ii];
		}
//	for(ii=0; ii<=N; ii++)
//		printf("\n%d\n", nx2[ii]);
//	for(ii=0; ii<=N; ii++)
//		printf("\n%d\n", nu2[ii]);
	
	// cost function
	struct d_strmat hsRSQrq2[N+1];
	struct d_strvec hsrq2[N+1];
	int tmp0[N+1];
	int tmp1[N+1];
	// allocate
	for(ii=0; ii<=N; ii++)
		{
		d_allocate_strmat(nu2[ii]+nx2[ii]+1, nu2[ii]+nx2[ii], &hsRSQrq2[ii]);
		d_allocate_strvec(nu2[ii]+nx2[ii], &hsrq2[ii]);
		}
	// zero index
	for(ii=0; ii<=N; ii++)
		{
		tmp0[ii] = 0;
		tmp1[ii] = 0;
		}
	// R
	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage;
		dgecp_libstr(t_nu[ii], t_nu[ii], 1.0, &t_hsRSQrq[ii], 0, 0, &hsRSQrq2[stage], tmp0[stage], tmp0[stage]);
		tmp0[stage] += t_nu[ii];
		}
	// Q
	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage;
		dgecp_libstr(t_nx[ii], t_nx[ii], 1.0, &t_hsRSQrq[ii], t_nu[ii], t_nu[ii], &hsRSQrq2[stage], tmp0[stage], tmp0[stage]);
		tmp0[stage] += t_nx[ii];
		}
	// r
	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage;
		dgecp_libstr(1, t_nu[ii], 1.0, &t_hsRSQrq[ii], t_nu[ii]+t_nx[ii], 0, &hsRSQrq2[stage], tmp0[stage], tmp1[stage]);
		dveccp_libstr(t_nu[ii], 1.0, &t_hsrq[ii], 0, &hsrq2[stage], tmp1[stage]);
		tmp1[stage] += t_nu[ii];
		}
	// q
	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage;
		dgecp_libstr(1, t_nx[ii], 1.0, &t_hsRSQrq[ii], t_nu[ii]+t_nx[ii], t_nu[ii], &hsRSQrq2[stage], tmp0[stage], tmp1[stage]);
		dveccp_libstr(t_nx[ii], 1.0, &t_hsrq[ii], t_nu[ii], &hsrq2[stage], tmp1[stage]);
		tmp1[stage] += t_nx[ii];
		}
	// print
//	for(ii=0; ii<=N; ii++)
//		{
//		d_print_strmat(nu2[ii]+nx2[ii]+1, nu2[ii]+nx2[ii], &hsRSQrq2[ii], 0, 0);
//		d_print_tran_strvec(nu2[ii]+nx2[ii], &hsrq2[ii], 0);
//		}
	
	// dynamical system
	struct d_strmat hsBAbt2[N];
	struct d_strvec hsb2[N];
	int idxkid, kid_stage;
//	int tmp0;
	for(ii=0; ii<N; ii++)
		{
		d_allocate_strmat(nu2[ii]+nx2[ii]+1, nx2[ii+1], &hsBAbt2[ii]);
		d_allocate_strvec(nx2[ii+1], &hsb2[ii]);
		}
	// zero index
	for(ii=0; ii<=N; ii++)
		{
		tmp0[ii] = 0;
		tmp1[ii] = 0;
		}
	// B
	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage;
		for(jj=0; jj<tree[ii].nkids; jj++)
			{
			idxkid = tree[ii].kids[jj];
			kid_stage = tree[idxkid].stage;
			dgecp_libstr(t_nu[ii], t_nx[idxkid], 1.0, &t_hsBAbt[idxkid], 0, 0, &hsBAbt2[stage], tmp1[stage], tmp0[kid_stage]);
			tmp0[kid_stage] += t_nx[idxkid];
			}
		tmp1[stage] += t_nu[ii];
		}
	// zero index
	for(ii=0; ii<=N; ii++)
		{
		tmp0[ii] = 0;
		}
	// A
	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage;
		for(jj=0; jj<tree[ii].nkids; jj++)
			{
			idxkid = tree[ii].kids[jj];
			kid_stage = tree[idxkid].stage;
			dgecp_libstr(t_nx[ii], t_nx[idxkid], 1.0, &t_hsBAbt[idxkid], t_nu[ii], 0, &hsBAbt2[stage], tmp1[stage], tmp0[kid_stage]);
			tmp0[kid_stage] += t_nx[idxkid];
			}
		tmp1[stage] += t_nx[ii];
		}
	// zero index
	for(ii=0; ii<=N; ii++)
		{
		tmp0[ii] = 0;
		}
	// b
	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage;
		for(jj=0; jj<tree[ii].nkids; jj++)
			{
			idxkid = tree[ii].kids[jj];
			kid_stage = tree[idxkid].stage;
			dgecp_libstr(1, t_nx[idxkid], 1.0, &t_hsBAbt[idxkid], t_nu[ii]+t_nx[ii], 0, &hsBAbt2[stage], tmp1[stage], tmp0[kid_stage]);
			dveccp_libstr(t_nx[idxkid], 1.0, &t_hsb[idxkid], 0, &hsb2[stage], tmp0[kid_stage]);
			tmp0[kid_stage] += t_nx[idxkid];
			}
//		tmp1[stage] += 1;
		}
	// print
//	for(ii=0; ii<N; ii++)
//		{
//		d_print_strmat(nu2[ii]+nx2[ii]+1, nx2[ii+1], &hsBAbt2[ii], 0, 0);
//		d_print_tran_strvec(nx2[ii+1], &hsb2[ii], 0);
//		}
	
	// work space
	struct d_strvec hsux2[N+1];
	struct d_strvec hspi2[N+1];
	struct d_strmat hsL2[N+1];
	struct d_strmat hsLxt2[N+1];
	struct d_strvec hsPb2[N+1];
	for(ii=0; ii<=N; ii++)
		{
		d_allocate_strvec(nu2[ii]+nx2[ii], &hsux2[ii]);
		d_allocate_strvec(nx2[ii], &hspi2[ii]);
		d_allocate_strmat(nu2[ii]+nx2[ii]+1, nu2[ii]+nx2[ii], &hsL2[ii]);
		d_allocate_strmat(nx2[ii], nx2[ii], &hsLxt2[ii]);
		d_allocate_strvec(nx2[ii], &hsPb2[ii]);
		}

	// riccati work space
	void *work_ric2;
	v_zeros_align(&work_ric2, d_back_ric_rec_work_space_size_bytes_libstr(N, nx2, nu2, nb, ng));

	// call Riccati solver
	

//	for(ii=0; ii<=N; ii++)
//		printf("\n%d %d\n", nu2[ii], nx2[ii]);

	gettimeofday(&tv0, NULL); // time

	for(rep=0; rep<nrep; rep++)
		{
//		d_back_ric_sv_libstr(N, nx, nu, hsBAbt, hsRSQrq, hsL, hsLxt, hsux, hspi, hswork_mat, hswork_vec);
		}

	gettimeofday(&tv1, NULL); // time

	for(rep=0; rep<nrep; rep++)
		{
		d_back_ric_rec_trf_libstr(N, nx2, nu2, nb, hidxb, ng, hsBAbt2, hsRSQrq2, hsDCt, hsQx, hsL2, hsLxt2, work_ric2);
		}

	gettimeofday(&tv2, NULL); // time

	for(rep=0; rep<nrep; rep++)
		{
		d_back_ric_rec_trs_libstr(N, nx2, nu2, nb, hidxb, ng, hsBAbt2, hsb2, hsrq2, hsDCt, hsqx, hsux2, 1, hspi2, 1, hsPb2, hsL2, hsLxt2, work_ric2);
		}

	gettimeofday(&tv3, NULL); // time

	float time_sv2  = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
	float time_trf2 = (float) (tv2.tv_sec-tv1.tv_sec)/(nrep+0.0)+(tv2.tv_usec-tv1.tv_usec)/(nrep*1e6);
	float time_trs2 = (float) (tv3.tv_sec-tv2.tv_sec)/(nrep+0.0)+(tv3.tv_usec-tv2.tv_usec)/(nrep*1e6);

	// print
//	printf("\nL\n");
//	for(ii=0; ii<=N; ii++)
//		{
//		d_print_strmat(nu2[ii]+nx2[ii]+1, nu2[ii]+nx2[ii], &hsL2[ii], 0, 0);
//		}
	printf("\nux\n");
	for(ii=0; ii<=N; ii++)
		{
		d_print_tran_strvec(nu2[ii]+nx2[ii], &hsux2[ii], 0);
		}
	printf("\npi\n");
	for(ii=0; ii<=N; ii++)
		{
		d_print_tran_strvec(nx2[ii], &hspi2[ii], 0);
		}

	printf("\ntime sv2\ttime trf2\ttime trs2\n");
	printf("\n%e\t%e\t%e\n", time_sv2, time_trf2, time_trs2);
	printf("\n");

	return 0;

/************************************************
* free memory
************************************************/	

	d_free(A);
	d_free(B);
	d_free(b);
	d_free_align(x0);
	d_free(R);
	d_free(S);
	d_free(Q);
	d_free(r);
	d_free(q);
	d_free(b0);
	d_free_strmat(&sA);
	d_free_strvec(&sb);
	d_free_strmat(&sBbt0);
	d_free_strvec(&sb0);
	d_free_strmat(&sBAbt1);
	d_free_strmat(&sRr0);
	d_free_strvec(&sr0);
	d_free_strmat(&sRSQrq1);
	d_free_strvec(&srq1);
	d_free_strmat(&sQqN);
	d_free_strvec(&sqN);
	d_free_strmat(&hsL[0]);
//	d_free_strmat(&hsLxt[0]);
	d_free_strvec(&hsPb[1]);
	d_free_strvec(&hsux[0]);
	d_free_strvec(&hspi[1]);
	for(ii=1; ii<N; ii++)
		{
		d_free_strmat(&hsL[ii]);
		d_free_strmat(&hsLxt[ii]);
		d_free_strvec(&hsPb[ii+1]);
		d_free_strvec(&hsux[ii]);
		d_free_strvec(&hspi[ii+1]);
		}
	d_free_strmat(&hsL[N]);
	d_free_strmat(&hsLxt[N]);
	d_free_strvec(&hsux[N]);
	d_free_strmat(&hswork_mat[0]);
	d_free_strvec(&hswork_vec[0]);

	// free memory allocated in the tree
	free_tree(md, Nr, Nh, Nn, tree);


/************************************************
* return
************************************************/	

	return 0;

	}



