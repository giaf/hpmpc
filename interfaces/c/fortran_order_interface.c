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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ACADO auto-generated header */
/*#include "acado_common.h"*/
/* HPMPC configuration */
/*#include "hpmpc_pro/target.h"*/
/*#include "hpmpc_pro/block_size.h"*/
/*#include "hpmpc_pro/aux_d.h"*/
/*#include "hpmpc_pro/aux_s.h"*/
/*#include "hpmpc_pro/mpc_solvers.h"*/

#include "../../include/target.h"
#include "../../include/block_size.h"
#include "../../include/aux_d.h"
#include "../../include/aux_s.h"
#include "../../include/blas_d.h"
#include "../../include/lqcp_solvers.h"
#include "../../include/mpc_solvers.h"

// problem size (states, inputs, horizon)
/*#define NX ACADO_NX*/
/*#define NU ACADO_NU*/
/*#define NN ACADO_N*/

// free initial state: 0 mpc, 1 mhe
//#define FREE_X0 0 // TODO remve

// ip method: 1 primal-dual, 2 predictor-corrector primal-dual
#define IP 2

// warm-start with user-provided solution (otherwise initialize x and u with 0 or something feasible)
#define WARM_START 0

// double/single ('d'/'s') precision
#define PREC 'd' // TODO remve

// minimum accepted step length
#define ALPHA_MIN 1e-12

// Debug flag
#ifndef PC_DEBUG
#define PC_DEBUG 0
#endif /* PC_DEBUG */

/* version dealing with equality constratins: is lb=ub, then fix the variable (corresponding column in A or B set to zero, and updated b) */
int fortran_order_ip_hard_mpc( int *kk, int k_max, double mu0, double mu_tol, char prec,
                          int N, int nx, int nu, int nb, int ng, int ngN, 
                          double* A, double* B, double* b, 
                          double* Q, double* Qf, double* S, double* R, 
                          double* q, double* qf, double* r, 
						  double *lb, double *ub,
                          double *C, double *D, double *lg, double *ug,
						  double *CN, double *lgf, double *ugf,
                          double* x, double* u,
						  double *work0, 
                          double *stat,
						  int compute_res, double *inf_norm_res,
						  int compute_mult, double *pi, double *lam, double *t)

	{

//printf("\nstart of wrapper\n");

	//printf("\n%d %d %d %d %d %d\n", N, nx, nu, nb, ng, ngN);

	int hpmpc_status = -1;

    //char prec = PREC;

    if(prec=='d')
	    {
	    
		//const int nb = nx+nu; // number of box constraints
		//const int ng = 0; // number of general constraints // TODO remove when not needed any longer
		const int nbu = nb<nu ? nb : nu ;

		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = D_MR*D_NCL;
	
		const int nz   = nx+nu+1;
		const int pnz  = bs*((nz+bs-1)/bs);
		const int pnu  = bs*((nu+bs-1)/bs);
		const int pnx  = bs*((nx+bs-1)/bs);
		const int pnb  = bs*((nb+bs-1)/bs);
		const int png  = bs*((ng+bs-1)/bs);
		const int pngN = bs*((ngN+bs-1)/bs);
		const int cnz  = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx  = ncl*((nx+ncl-1)/ncl);
		const int cng  = ncl*((ng+ncl-1)/ncl);
		const int cngN = ncl*((ngN+ncl-1)/ncl);
		const int anz  = nal*((nz+nal-1)/nal);
		const int anx  = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		//const int anb = nal*((2*nb+nal-1)/nal);

		double alpha_min = ALPHA_MIN; // minimum accepted step length
        static double sigma_par[] = {0.4, 0.1, 0.001}; // control primal-dual IP behaviour
/*      static double stat[5*K_MAX]; // statistics from the IP routine*/
//      double *work0 = (double *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 5*anz + 3*anx + 7*anb) + 3*anz)*sizeof(double));
        int warm_start = WARM_START;
        int compute_mult = 1; // compute multipliers
		double temp;
        
        int info = 0;

        int i, ii, jj, ll;



        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        double *ptr = work0 + offset / 8;



        /* array or pointers */
        double *(hpBAbt[N]);
        double *(hpDCt[N+1]);
        double *(hpQ[N+1]);
        double *(hux[N+1]);
        double *(hd[N+1]);
        double *(hpi[N+1]);
        double *(hlam[N+1]);
        double *(ht[N+1]);
		double *(hq[N+1]);
		double *(hrb[N]);
		double *(hrq[N+1]);
		double *(hrd[N+1]);
		double *work;

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

		if(ng>0)
			{
			for(ii=0; ii<N; ii++)
				{
				hpDCt[ii] = ptr;
				ptr += pnz*cng;
				}
			}
		if(ngN>0)
			{
			hpDCt[N] = ptr;
			ptr += pnz*cngN;
			}

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<N; ii++) // time Variant box constraints
	        {
            hd[ii] = ptr;
            ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
	        }
		hd[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        for(ii=0; ii<N; ii++) // time Variant box constraints
	        {
            hlam[ii] = ptr;
            ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
	        }
		hlam[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

        for(ii=0; ii<N; ii++) // time Variant box constraints
	        {
            ht[ii] = ptr;
            ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
	        }
		ht[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

		if(compute_res)
			{

			for(ii=0; ii<=N; ii++)
				{
				hq[ii] = ptr;
				ptr += anz;
				}

			for(ii=0; ii<=N; ii++)
				{
				hrb[ii] = ptr;
				ptr += anx;
				}

			for(ii=0; ii<=N; ii++)
				{
				hrq[ii] = ptr;
				ptr += anz;
				}

			for(ii=0; ii<N; ii++)
				{
				hrd[ii] = ptr;
				ptr += 2*pnb+2*png;
				}
			hrd[N] = ptr;
			ptr += 2*pnb+2*pngN;

			}

		work = ptr;



        /* pack matrices 	*/

		//printf("\n%d %d %d %d %d\n", N, nx, nu, nb, ng);

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            d_cvt_tran_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
            d_cvt_tran_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
	        }
		//d_print_mat(nx, nu, B, nx);
		//d_print_mat(nx, nx, A, nx);
		//d_print_pmat(nz, nx, bs, hpBAbt[0], cnx);
		//d_print_pmat(nz, nx, bs, hpBAbt[N-1], cnx);
		//exit(1);

		// general constraints
		if(ng>0)
			{
			for(ii=0; ii<N; ii++)
				{
				d_cvt_tran_mat2pmat(ng, nu, 0, bs, D+ii*nu*ng, ng, hpDCt[ii], cng);
				d_cvt_tran_mat2pmat(ng, nx, nu, bs, C+ii*nx*ng, ng, hpDCt[ii]+nu/bs*cng*bs+nu%bs, cng);
				}
			}
		if(ngN>0)
			{
			for(ii=0; ii<pnu*cngN; ii++) hpDCt[N][ii] = 0.0; // make sure D is zero !!!!!
			d_cvt_tran_mat2pmat(ngN, nx, nu, bs, CN, ngN, hpDCt[N]+nu/bs*cngN*bs+nu%bs, cngN);
			}
		//d_print_mat(ngN, nx, CN, ngN);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[0], cng);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[1], cng);
		//d_print_pmat(nx+nu, ngN, bs, hpDCt[N], cngN);
		//exit(1);
		//printf("\n%d %d\n", nb, ng);
		//d_print_mat(ng, nx, C, ng);
		//d_print_mat(ng, nx, C+nx*ng, ng);
		//d_print_mat(ng, nu, D, ng);
		//d_print_mat(ng, nu, D+nu*ng, ng);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[0], cng);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[1], cng);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[N], cng);
		//exit(1);

        // cost function
        for(jj=0; jj<N; jj++)
	        {
            d_cvt_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
            d_cvt_tran_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
            d_cvt_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
            for(ii=0; ii<nu; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = r[ii+jj*nu];
            for(ii=0; ii<nx; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = q[ii+nx*jj];
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        d_cvt_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
        for(jj=0; jj<nx; jj++)
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = qf[jj];

		// estimate mu0 if not user-provided
		if(mu0<=0)
			{
			for(jj=0; jj<N; jj++)
				{
				for(ii=0; ii<nu; ii++) for(ll=0; ll<nu; ll++) mu0 = fmax(mu0, R[jj*nu*nu+ii*nu+ll]);
				for(ii=0; ii<nx*nu; ii++) mu0 = fmax(mu0, S[jj*nu*nx+ii]);
				for(ii=0; ii<nx; ii++) for(ll=0; ll<nx; ll++) mu0 = fmax(mu0, Q[jj*nx*nx+ii*nx+ll]);
				for(ii=0; ii<nu; ii++) mu0 = fmax(mu0, r[jj*nu+ii]);
				for(ii=0; ii<nx; ii++) mu0 = fmax(mu0, q[jj*nx+ii]);
				}
			for(ii=0; ii<nx; ii++) for(ll=0; ll<nx; ll++) mu0 = fmax(mu0, Qf[ii*nx+ll]);
			for(jj=0; jj<nx; jj++) mu0 = fmax(mu0, qf[ii]);
			}

		//d_print_pmat(nz, nz, bs, hpQ[0], cnz);
		//d_print_pmat(nz, nz, bs, hpQ[1], cnz);
		//d_print_pmat(nz, nz, bs, hpQ[N], cnz);
		//exit(1);

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nbu; ii++)
				{
				if(lb[ii+nb*jj]!=ub[ii+nb*jj]) // equality constraint
					{
					hd[jj][ii+0]   =   lb[ii+nb*jj];
					hd[jj][ii+pnb] = - ub[ii+nb*jj];
					}
				else
					{
					for(ll=0; ll<nx; ll++)
						{
						// update linear term
						hpBAbt[jj][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+ll*bs] += hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs]*lb[ii+nb*jj];
						// zero corresponding B column
						hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs] = 0;
						}
					
					// inactive box constraints
					hd[jj][ii+0]   =   lb[ii+nb*jj] + 1e3;
					hd[jj][ii+pnb] = - ub[ii+nb*jj] - 1e3;

					}
				}
			}
		// state constraints 
		for(jj=1; jj<=N; jj++)
			{
			for(ii=nbu; ii<nb; ii++)
				{
				hd[jj][ii+0]   =   lb[ii+nb*jj];
				hd[jj][ii+pnb] = - ub[ii+nb*jj];
				//hd[jj][2*nu+2*ii+0] =   lb[N*nu+ii+nx*jj];
				//hd[jj][2*nu+2*ii+1] = - ub[N*nu+ii+nx*jj];
				}
			}
		// general constraints
		if(ng>0)
			{
			for(jj=0; jj<N; jj++)
				{
				for(ii=0; ii<ng; ii++)
					{
					hd[jj][2*pnb+ii+0]   =   lg[ii+ng*jj];
					hd[jj][2*pnb+ii+png] = - ug[ii+ng*jj];
					}
				}
			}
		if(ngN>0) // last stage
			{
			for(ii=0; ii<ngN; ii++)
				{
				hd[N][2*pnb+ii+0]    =   lgf[ii];
				hd[N][2*pnb+ii+pngN] = - ugf[ii];
				}
			}
		//d_print_mat(1, 2*pnb+2*png, hd[0], 1);
		//d_print_mat(1, 2*pnb+2*png, hd[1], 1);
		//d_print_mat(1, 2*pnb+2*pngN, hd[N], 1);
		//exit(1);

        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = x[ii+nx*jj];



        // call the IP solver
	    if(IP==1)
	        hpmpc_status = d_ip_hard_mpc(kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);
	    else
	        hpmpc_status = d_ip2_hard_mpc(kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);



        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = hux[jj][nu+ii];



		// check for input and states equality constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nbu; ii++)
				{
				if(lb[ii+nb*jj]==ub[ii+nb*jj]) // equality constraint
					{
	                u[ii+nu*jj] = lb[ii+nb*jj];
					}
				}
			}

		// compute infinity norm of residuals on exit
		if(compute_res)
			{

			// restore linear part of cost function 
			for(ii=0; ii<N; ii++)
				{
				for(jj=0; jj<nu; jj++) 
					hq[ii][jj]    = r[jj+nu*ii];
				for(jj=0; jj<nx; jj++) 
					hq[ii][nu+jj] = q[jj+nx*ii];
				}
			for(jj=0; jj<nx; jj++) 
				hq[N][nu+jj] = qf[jj];

			double mu;

			d_res_ip_hard_mpc(nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hq, hux, hpDCt, hd, hpi, hlam, ht, hrq, hrb, hrd, &mu);

			temp = fabs(hrq[0][0]);
			for(jj=0; jj<nu; jj++) 
				temp = fmax( temp, fabs(hrq[0][jj]) );
			for(ii=1; ii<N; ii++)
				for(jj=0; jj<nu+nx; jj++) 
					temp = fmax( temp, fabs(hrq[ii][jj]) );
			for(jj=nu; jj<nu+nx; jj++) 
				temp = fmax( temp, fabs(hrq[N][jj]) );
			inf_norm_res[0] = temp;

			temp = fabs(hrb[0][0]);
			for(ii=0; ii<N; ii++)
				for(jj=0; jj<nx; jj++) 
					temp = fmax( temp, fabs(hrb[ii][jj]) );
			inf_norm_res[1] = temp;

			temp = fabs(hrd[0][0]);
			for(jj=0; jj<nbu; jj++) 
				{
				temp = fmax( temp, fabs(hrd[0][jj]) );
				temp = fmax( temp, fabs(hrd[0][pnb+jj]) );
				}
			for(ii=1; ii<N; ii++)
				for(jj=0; jj<nb; jj++) 
					{
					temp = fmax( temp, fabs(hrd[ii][jj]) );
					temp = fmax( temp, fabs(hrd[ii][pnb+jj]) );
					}
			for(jj=nbu; jj<nb; jj++) 
				{
				temp = fmax( temp, fabs(hrq[N][jj]) );
				temp = fmax( temp, fabs(hrq[N][pnb+jj]) );
				}
			for(ii=0; ii<N; ii++)
				for(jj=2*pnb; jj<2*pnb+ng; jj++) 
					{
					temp = fmax( temp, fabs(hrd[ii][jj]) );
					temp = fmax( temp, fabs(hrd[ii][png+jj]) );
					}
			for(jj=2*pnb; jj<2*pnb+ngN; jj++) 
				{
				temp = fmax( temp, fabs(hrd[N][jj]) );
				temp = fmax( temp, fabs(hrd[N][pngN+jj]) );
				}
			inf_norm_res[2] = temp;

			inf_norm_res[3] = mu;

			//printf("\n%e %e %e %e\n", norm_res[0], norm_res[1], norm_res[2], norm_res[3]);

			}

		if(compute_mult)
			{

			for(ii=0; ii<=N; ii++)
				for(jj=0; jj<nx; jj++)
					pi[jj+ii*nx] = hpi[ii][jj];

			ii = 0;
			for(jj=0; jj<nbu; jj++)
				{
				lam[jj+ii*(2*nb+2*ng)]       = hlam[ii][jj];
				lam[jj+ii*(2*nb+2*ng)+nb+ng] = hlam[ii][pnb+jj];
				t[jj+ii*(2*nb+2*ng)]       = ht[ii][jj];
				t[jj+ii*(2*nb+2*ng)+nb+ng] = ht[ii][pnb+jj];
				}
			for(ii=1; ii<N; ii++)
				{
				for(jj=0; jj<nb; jj++)
					{
					lam[jj+ii*(2*nb+2*ng)]       = hlam[ii][jj];
					lam[jj+ii*(2*nb+2*ng)+nb+ng] = hlam[ii][pnb+jj];
					t[jj+ii*(2*nb+2*ng)]       = ht[ii][jj];
					t[jj+ii*(2*nb+2*ng)+nb+ng] = ht[ii][pnb+jj];
					}
				}
			ii = N;
			for(jj=nbu; jj<nb; jj++)
				{
				lam[jj+ii*(2*nb+2*ng)]        = hlam[ii][jj];
				lam[jj+ii*(2*nb+2*ng)+nb+ngN] = hlam[ii][pnb+jj];
				t[jj+ii*(2*nb+2*ng)]        = ht[ii][jj];
				t[jj+ii*(2*nb+2*ng)+nb+ngN] = ht[ii][pnb+jj];
				}

			for(ii=0; ii<N; ii++)
				for(jj=0; jj<ng; jj++)
					{
					lam[jj+ii*(2*nb+2*ng)+nb]       = hlam[ii][2*pnb+jj];
					lam[jj+ii*(2*nb+2*ng)+nb+ng+nb] = hlam[ii][2*pnb+png+jj];
					t[jj+ii*(2*nb+2*ng)+nb]       = ht[ii][2*pnb+jj];
					t[jj+ii*(2*nb+2*ng)+nb+ng+nb] = ht[ii][2*pnb+png+jj];
					}
			for(jj=0; jj<ngN; jj++)
				{
				lam[jj+N*(2*nb+2*ng)+nb]        = hlam[N][2*pnb+jj];
				lam[jj+N*(2*nb+2*ng)+nb+ngN+nb] = hlam[N][2*pnb+pngN+jj];
				t[jj+N*(2*nb+2*ng)+nb]        = ht[N][2*pnb+jj];
				t[jj+N*(2*nb+2*ng)+nb+ngN+nb] = ht[N][2*pnb+pngN+jj];
				}

			}



#if PC_DEBUG == 1
        for (jj = 0; jj < *kk; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */


	    }
    else if(prec=='s')
	    {
	    
		const int bs = S_MR; //d_get_mr();
		const int ncl = S_NCL;
		const int nal = S_MR*S_NCL;

		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		const int nb = nx+nu; // number of box constraints
		const int anb = nal*((2*nb+nal-1)/nal);

		float alpha_min = ALPHA_MIN; // minimum accepted step length
        static float sigma_par[] = {0.4, 0.1, 0.01}; // control primal-dual IP behaviour
/*        static float stat[5*K_MAX]; // statistics from the IP routine*/
        //float *work0 = (float *) malloc((16 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 5*anz + 3*anx + 7*anb) + 3*anz)*sizeof(float));
        int warm_start = WARM_START;
        int compute_mult = 1; // compute multipliers
        
        int info = 0;

        int i, ii, jj, ll;


        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        float *ptr = ((float *) work0) + offset / 4;



        /* array or pointers */
        float *(hpBAbt[N]);
        float *(hpQ[N + 1]);
        float *(hux[N + 1]);
        float *(hdb[N + 1]);
        float *(hpi[N + 1]);
        float *(hlam[N + 1]);
        float *(ht[N + 1]);
		float *work;

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hdb[ii] = ptr;
            ptr += anb; //nb; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hlam[ii] = ptr;
            ptr += anb; //nb; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            ht[ii] = ptr;
            ptr += anb; //nb; // for alignment of ptr
	        }

		work = ptr;



        /* pack matrices 	*/

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            cvt_tran_d2s_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
            cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = (float) b[ii*nx+jj];
	        }

        // cost function
        for(jj=0; jj<N; jj++)
	        {
            cvt_d2s_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
            cvt_tran_d2s_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
            cvt_d2s_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
            for(ii=0; ii<nu; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = (float) r[ii+jj*nu];
            for(ii=0; ii<nx; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = (float) q[ii+nx*jj];
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        cvt_d2s_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
        for(jj=0; jj<nx; jj++)
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = (float) qf[jj];

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				if(lb[ii+nu*jj]!=ub[ii+nu*jj]) // equality constraint
					{
					hdb[jj][2*ii+0] = (float)   lb[ii+nu*jj];
					hdb[jj][2*ii+1] = (float) - ub[ii+nu*jj];
					}
				else
					{
					for(ll=0; ll<nx; ll++)
						{
						// update linear term
						hpBAbt[jj][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+ll*bs] += hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs]*lb[ii+nu*jj];
						// zero corresponding B column
						hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs] = 0;
						}
					
					// inactive box constraints
					hdb[jj][2*ii+0] = (float)   lb[ii+nu*jj] + 1e3;
					hdb[jj][2*ii+1] = (float) - ub[ii+nu*jj] - 1e3;

/*		            d_print_pmat(nx+nu, nx, bs, hpBAbt[jj], cnx);*/
					}
				}
			}
		// state constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nx; ii++)
				{
				hdb[jj+1][2*nu+2*ii+0] = (float)   lb[N*nu+ii+nx*jj];
				hdb[jj+1][2*nu+2*ii+1] = (float) - ub[N*nu+ii+nx*jj];
				}
			}


        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = (float) u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = (float) x[ii+nx*jj];



        // call the IP solver
		if(IP==1)
		    hpmpc_status = s_ip_box_mpc(kk, k_max, mu_tol, alpha_min, warm_start, sigma_par, (float *) stat, nx, nu, N, nb, hpBAbt, hpQ, hdb, hux, compute_mult, hpi, hlam, ht, work);
		else
		    hpmpc_status = s_ip2_box_mpc(kk, k_max, mu_tol, alpha_min, warm_start, sigma_par, (float *) stat, nx, nu, N, nb, hpBAbt, hpQ, hdb, hux, compute_mult, hpi, hlam, ht, work);



		// convert stat into double (start fom end !!!)
		float *ptr_stat = (float *) stat;
		for(ii=5*k_max-1; ii>=0; ii--)
			{
			stat[ii] = (double) ptr_stat[ii];
			}



        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = (double) hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = (double) hux[jj][nu+ii];

		// check for input equality constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				if(lb[ii+nu*jj]==ub[ii+nu*jj]) // equality constraint
					{
	                u[ii+nu*jj] = lb[ii+nu*jj];
					}
				}
			}


#if PC_DEBUG == 1
        for (jj = 0; jj < *kk; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */


		}
	else
		{
		printf("\nUnsopported precision type: %s\n\n", &prec);
		return -1;
		}
	
/*printf("\nend of wrapper\n");*/

    return hpmpc_status;

	}



int fortran_order_ip_soft_mpc( int k_max, double mu_tol, const char prec,
                               const int nx, const int nu, const int N,
                               double* A, double* B, double* b, 
                               double* Q, double* Qf, double* S, double* R, 
                               double* q, double* qf, double* r, 
                               double* lZ, double* uZ,
                               double* lz, double* uz,
                               double* lb, double* ub, 
                               double* x, double* u,
                               double* work0, 
                               int* nIt, double* stat )

	{

//printf("\nstart of wrapper\n");

	int hpmpc_status = -1;

    //char prec = PREC;

    if(prec=='d')
	    {
	    
		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = D_MR*D_NCL;
	
		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		const int nb = nx+nu; // number of box constraints
		const int anb = nal*((2*nb+nal-1)/nal);

		double alpha_min = ALPHA_MIN; // minimum accepted step length
        static double sigma_par[] = {0.4, 0.1, 0.001}; // control primal-dual IP behaviour
/*      static double stat[5*K_MAX]; // statistics from the IP routine*/
//      double *work0 = (double *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 5*anz + 3*anx + 7*anb) + 3*anz)*sizeof(double));
        int warm_start = WARM_START;
        int compute_mult = 1; // compute multipliers
        
        int info = 0;

        int i, ii, jj, ll;



        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        double *ptr = work0 + offset / 8;



        /* array or pointers */
        double *(hpBAbt[N]);
        double *(hpQ[N + 1]);
		double *(hZ[N + 1]);
		double *(hz[N + 1]);
        double *(hux[N + 1]);
        double *(hdb[N + 1]);
        double *(hpi[N + 1]);
        double *(hlam[N + 1]);
        double *(ht[N + 1]);
		double *work;

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

		for(ii=0; ii<=N; ii++)
	        {
			hZ[ii] = ptr;
			ptr += anb;
			}

		for(ii=0; ii<=N; ii++)
	        {
			hz[ii] = ptr;
			ptr += anb;
			}

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hdb[ii] = ptr;
            ptr += anb; //nb; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hlam[ii] = ptr;
            ptr += 2*anb; //nb; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            ht[ii] = ptr;
            ptr += 2*anb; //nb; // for alignment of ptr
	        }

		work = ptr;



        /* pack matrices 	*/

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            d_cvt_tran_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
            d_cvt_tran_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
	        }

        // cost function
		double mu0 = 1.0;
        for(jj=0; jj<N; jj++)
	        {
            d_cvt_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
			for(ii=0; ii<nu; ii++) for(ll=0; ll<nu; ll++) mu0 = fmax(mu0, R[jj*nu*nu+ii*nu+ll]);
            d_cvt_tran_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
			for(ii=0; ii<nx*nu; ii++) mu0 = fmax(mu0, S[jj*nu*nx+ii]);
            d_cvt_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
			for(ii=0; ii<nx; ii++) for(ll=0; ll<nx; ll++) mu0 = fmax(mu0, Q[jj*nx*nx+ii*nx+ll]);
            for(ii=0; ii<nu; ii++)
				{
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = r[ii+jj*nu];
				mu0 = fmax(mu0, r[jj*nu+ii]);
				}
            for(ii=0; ii<nx; ii++)
				{
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = q[ii+nx*jj];
				mu0 = fmax(mu0, q[jj*nx+ii]);
				}
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        d_cvt_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
		for(ii=0; ii<nx; ii++) for(ll=0; ll<nx; ll++) mu0 = fmax(mu0, Qf[ii*nx+ll]);
        for(jj=0; jj<nx; jj++)
			{
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = qf[jj];
			mu0 = fmax(mu0, qf[ii]);
			}

		// cost function of soft constraint slack variable
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nx; jj++)
				{
				hZ[ii+1][2*nu+2*jj+0] = lZ[nx*ii+jj];
				hZ[ii+1][2*nu+2*jj+1] = uZ[nx*ii+jj];
				mu0 = fmax(mu0, lZ[nx*ii+jj]);
				mu0 = fmax(mu0, uZ[nx*ii+jj]);
				}
			for(jj=0; jj<nx; jj++)
				{
				hz[ii+1][2*nu+2*jj+0] = lz[nx*ii+jj];
				hz[ii+1][2*nu+2*jj+1] = uz[nx*ii+jj];
				mu0 = fmax(mu0, lz[nx*ii+jj]);
				mu0 = fmax(mu0, uz[nx*ii+jj]);
				}
			}

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				if(lb[ii+nu*jj]!=ub[ii+nu*jj]) // equality constraint
					{
					hdb[jj][2*ii+0] =   lb[ii+nu*jj];
					hdb[jj][2*ii+1] = - ub[ii+nu*jj];
					}
				else
					{
					for(ll=0; ll<nx; ll++)
						{
						// update linear term
						hpBAbt[jj][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+ll*bs] += hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs]*lb[ii+nu*jj];
						// zero corresponding B column
						hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs] = 0;
						}
					
					// inactive box constraints
					hdb[jj][2*ii+0] =   lb[ii+nu*jj] + 1e3;
					hdb[jj][2*ii+1] = - ub[ii+nu*jj] - 1e3;

					}
				}
			}
		// state constraints 
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nx; ii++)
				{
				hdb[jj+1][2*nu+2*ii+0] =   lb[N*nu+ii+nx*jj];
				hdb[jj+1][2*nu+2*ii+1] = - ub[N*nu+ii+nx*jj];
				}
			}

        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = x[ii+nx*jj];




        // call the IP solver
	    if(IP==1)
	        hpmpc_status = d_ip_soft_mpc(nIt, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, nx, nu, N, nu, nb-nu, hpBAbt, hpQ, hZ, hz, hdb, hux, compute_mult, hpi, hlam, ht, work);
	    else
	        hpmpc_status = d_ip2_soft_mpc(nIt, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, nx, nu, N, nu, nb-nu, hpBAbt, hpQ, hZ, hz, hdb, hux, compute_mult, hpi, hlam, ht, work);



        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = hux[jj][nu+ii];

		// check for input and states equality constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				if(lb[ii+nu*jj]==ub[ii+nu*jj]) // equality constraint
					{
	                u[ii+nu*jj] = lb[ii+nu*jj];
					}
				}
			}



#if PC_DEBUG == 1
        for (jj = 0; jj < *nIt; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */


	    }
#if 0
    else if(prec=='s')
	    {
	    
		const int bs = S_MR; //d_get_mr();
		const int ncl = S_NCL;
		const int nal = S_MR*S_NCL;

		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		const int nb = nx+nu; // number of box constraints
		const int anb = nal*((2*nb+nal-1)/nal);

		float alpha_min = ALPHA_MIN; // minimum accepted step length
        static float sigma_par[] = {0.4, 0.1, 0.01}; // control primal-dual IP behaviour
/*        static float stat[5*K_MAX]; // statistics from the IP routine*/
        //float *work0 = (float *) malloc((16 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 5*anz + 3*anx + 7*anb) + 3*anz)*sizeof(float));
        int warm_start = WARM_START;
        int compute_mult = 1; // compute multipliers
        
        int info = 0;

        int i, ii, jj, ll;


        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        float *ptr = ((float *) work0) + offset / 4;



        /* array or pointers */
        float *(hpBAbt[N]);
        float *(hpQ[N + 1]);
        float *(hux[N + 1]);
        float *(hdb[N + 1]);
        float *(hpi[N + 1]);
        float *(hlam[N + 1]);
        float *(ht[N + 1]);
		float *work;

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hdb[ii] = ptr;
            ptr += anb; //nb; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hlam[ii] = ptr;
            ptr += anb; //nb; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            ht[ii] = ptr;
            ptr += anb; //nb; // for alignment of ptr
	        }

		work = ptr;



        /* pack matrices 	*/

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            cvt_tran_d2s_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
            cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = (float) b[ii*nx+jj];
	        }

        // cost function
        for(jj=0; jj<N; jj++)
	        {
            cvt_d2s_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
            cvt_tran_d2s_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
            cvt_d2s_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
            for(ii=0; ii<nu; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = (float) r[ii+jj*nu];
            for(ii=0; ii<nx; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = (float) q[ii+nx*jj];
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        cvt_d2s_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
        for(jj=0; jj<nx; jj++)
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = (float) qf[jj];

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				if(lb[ii+nu*jj]!=ub[ii+nu*jj]) // equality constraint
					{
					hdb[jj][2*ii+0] = (float)   lb[ii+nu*jj];
					hdb[jj][2*ii+1] = (float) - ub[ii+nu*jj];
					}
				else
					{
					for(ll=0; ll<nx; ll++)
						{
						// update linear term
						hpBAbt[jj][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+ll*bs] += hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs]*lb[ii+nu*jj];
						// zero corresponding B column
						hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs] = 0;
						}
					
					// inactive box constraints
					hdb[jj][2*ii+0] = (float)   lb[ii+nu*jj] + 1e3;
					hdb[jj][2*ii+1] = (float) - ub[ii+nu*jj] - 1e3;

/*		            d_print_pmat(nx+nu, nx, bs, hpBAbt[jj], cnx);*/
					}
				}
			}
		// state constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nx; ii++)
				{
				hdb[jj+1][2*nu+2*ii+0] = (float)   lb[N*nu+ii+nx*jj];
				hdb[jj+1][2*nu+2*ii+1] = (float) - ub[N*nu+ii+nx*jj];
				}
			}


        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = (float) u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = (float) x[ii+nx*jj];



        // call the IP solver
		if(IP==1)
		    hpmpc_status = s_ip_box_mpc(nIt, k_max, mu_tol, alpha_min, warm_start, sigma_par, (float *) stat, nx, nu, N, nb, hpBAbt, hpQ, hdb, hux, compute_mult, hpi, hlam, ht, work);
		else
		    hpmpc_status = s_ip2_box_mpc(nIt, k_max, mu_tol, alpha_min, warm_start, sigma_par, (float *) stat, nx, nu, N, nb, hpBAbt, hpQ, hdb, hux, compute_mult, hpi, hlam, ht, work);



		// convert stat into double (start fom end !!!)
		float *ptr_stat = (float *) stat;
		for(ii=5*k_max-1; ii>=0; ii--)
			{
			stat[ii] = (double) ptr_stat[ii];
			}



        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = (double) hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = (double) hux[jj][nu+ii];

		// check for input equality constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				if(lb[ii+nu*jj]==ub[ii+nu*jj]) // equality constraint
					{
	                u[ii+nu*jj] = lb[ii+nu*jj];
					}
				}
			}


#if PC_DEBUG == 1
        for (jj = 0; jj < *nIt; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */


		}
#endif
	else
		{
		printf("\nUnsopported precision type: %s\n\n", &prec);
		return -1;
		}
	
/*printf("\nend of wrapper\n");*/

    return hpmpc_status;

	}



int fortran_order_riccati_mpc( const char prec,
                               const int nx, const int nu, const int N,
                               double *A, double *B, double *b, 
                               double *Q, double *Qf, double *S, double *R, 
                               double *q, double *qf, double *r, 
                               double *x, double *u, double *pi, 
                               double *work0 )
	{

	//char prec = PREC;

	if(prec=='d')
		{

		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = D_MR*D_NCL;

		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		//const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;

		//double *work = (double *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 2*anz + 2*anx) + 3*anz)*sizeof(double));

		int compute_mult = 1; // compute multipliers

		int i, ii, jj, ll;


		/* align work space */
		size_t align = 64;
		size_t addr = (size_t) work0;
		size_t offset = addr % 64;
		double *ptr = work0 + offset / 8;

		/* array or pointers */
		double *(hpBAbt[N]);
		double *(hpQ[N + 1]);
		double *(hpL[N + 1]);
		double *(hpl[N + 1]);
		double *(hux[N + 1]);
		double *(hpi[N + 1]);
		double *diag;
		double *work;

		// dynamic system
		for(ii=0; ii<N; ii++)
			{
			hpBAbt[ii] = ptr;
			ptr += pnz*cnx;
			}

		// cost function
		for(ii=0; ii<=N; ii++)
			{
			hpQ[ii] = ptr;
			ptr += pnz*cnz;
			}

		// work space (matrices)
		for(jj=0; jj<=N; jj++)
			{
			hpL[jj] = ptr;
			ptr += pnz*cnl;
			}

		// work space (vectors)
		for(jj=0; jj<=N; jj++)
			{
			hpl[jj] = ptr;
			ptr += anz;
			}

		// states and inputs
		for(ii=0; ii<=N; ii++)
			{
			hux[ii] = ptr;
			ptr += anz;
			}
		
		// eq. constr. multipliers
        for(ii=0; ii<=N; ii++) 
			{
			hpi[ii] = ptr;
			ptr += anx;
			}

		// inverted diagonal
		diag = ptr;
		ptr += anz;

		// work space
		work = ptr;
		ptr += 2*anz;



		/* pack matrices 	*/

		// dynamic system
		for(ii=0; ii<N; ii++)
			{
			d_cvt_tran_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
			d_cvt_tran_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
			for (jj = 0; jj<nx; jj++)
				hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
			}

		// cost function
		for(jj=0; jj<N; jj++)
			{
			d_cvt_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
			d_cvt_tran_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
			d_cvt_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
			for(ii=0; ii<nu; ii++)
				hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = r[ii+jj*nu];
			for(ii=0; ii<nx; ii++)
				hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = q[ii+nx*jj];
			}

		for(jj=0; jj<nu; jj++)
			for(ii=0; ii<nz; ii+=bs)
				for(i=0; i<bs; i++)
					hpQ[N][ii*cnz+i+jj*bs] = 0.0;
		for(jj=0; jj<nu; jj++)
			hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
		d_cvt_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
		for(jj=0; jj<nx; jj++)
			hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = qf[jj];



		// initial state
		for(ii=0; ii<nx; ii++)
            hux[0][nu+ii] = x[ii];


		// TODO
		double **dummy;

		// call Riccati solver
		d_ric_sv_mpc(nx, nu, N, hpBAbt, hpQ, 0, dummy, dummy, hux, hpL, work, diag, compute_mult, hpi, 0, 0, 0, dummy, dummy, dummy, 0);



		// copy back inputs
		for(jj=0; jj<N; jj++)
			for(ii=0; ii<nu; ii++)
				u[ii+nu*jj] = hux[jj][ii];

		// copy back states
		for(jj=0; jj<N; jj++)
			for(ii=0; ii<nx; ii++)
				x[ii+nx*(jj+1)] = hux[jj+1][nu+ii];

		// copy back lagrangian multipliers
		for(jj=0; jj<N; jj++)
			for(ii=0; ii<nx; ii++)
				pi[ii+nx*jj] = hpi[jj+1][ii];


		
		}
    else if(prec=='s')
	    {
	    
		const int bs = S_MR; //d_get_mr();
		const int ncl = S_NCL;
		const int nal = S_MR*S_NCL;
	
		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		//float *work = (float *) malloc((16 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 2*anz + 2*anx) + 3*anz)*sizeof(float));

		int compute_mult = 1; // compute multipliers

		int i, ii, jj, ll;


		/* align work space */
		size_t align = 64; // max cache line size for all supported architectures
		size_t addr = (size_t) work0;
		size_t offset = addr % 64;
		float *ptr = ((float *) work0) + offset / 8;

		/* array or pointers */
		float *(hpBAbt[N]);
		float *(hpQ[N + 1]);
		float *(hpL[N + 1]);
		float *(hpl[N + 1]);
		float *(hux[N + 1]);
		float *(hpi[N + 1]);
		float *diag;
		float *work;

		// dynamic system
		for(ii=0; ii<N; ii++)
			{
			hpBAbt[ii] = ptr;
			ptr += pnz*cnx;
			}

		// cost function
		for(ii=0; ii<=N; ii++)
			{
			hpQ[ii] = ptr;
			ptr += pnz*cnz;
			}

		// work space (matrices)
		for(jj=0; jj<=N; jj++)
			{
			hpL[jj] = ptr;
			ptr += pnz*cnl;
			}

		// work space (vectors)
		for(jj=0; jj<=N; jj++)
			{
			hpl[jj] = ptr;
			ptr += anz;
			}

		// states and inputs
		for(ii=0; ii<=N; ii++)
			{
			hux[ii] = ptr;
			ptr += anz;
			}
		
		// eq. constr. multipliers
        for(ii=0; ii<=N; ii++) 
			{
			hpi[ii] = ptr;
			ptr += anx;
			}

		// inverted diagonal
		diag = ptr;
		ptr += anz;

		// work space
		work = ptr;
		ptr += 2*anz;



		/* pack matrices 	*/

		// dynamic system
		for(ii=0; ii<N; ii++)
			{
			cvt_tran_d2s_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
			cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
			for (jj = 0; jj<nx; jj++)
				hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
			}

		// cost function
		for(jj=0; jj<N; jj++)
			{
			cvt_d2s_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
			cvt_tran_d2s_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
			cvt_d2s_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
			for(ii=0; ii<nu; ii++)
				hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = r[ii+jj*nu];
			for(ii=0; ii<nx; ii++)
				hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = q[ii+nx*jj];
			}

		for(jj=0; jj<nu; jj++)
			for(ii=0; ii<nz; ii+=bs)
				for(i=0; i<bs; i++)
					hpQ[N][ii*cnz+i+jj*bs] = 0.0;
		for(jj=0; jj<nu; jj++)
			hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
		cvt_d2s_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
		for(jj=0; jj<nx; jj++)
			hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = qf[jj];



		// initial state
		for(ii=0; ii<nx; ii++)
            hux[0][nu+ii] = (float) x[ii];
        


		// call Riccati solver
		s_ric_sv_mpc(nx, nu, N, hpBAbt, hpQ, hux, hpL, work, diag, compute_mult, hpi);



		// copy back inputs
		for(jj=0; jj<N; jj++)
			for(ii=0; ii<nu; ii++)
				u[ii+nu*jj] = (double) hux[jj][ii];

		// copy back states
		for(jj=0; jj<N; jj++)
			for(ii=0; ii<nx; ii++)
				x[ii+nx*(jj+1)] = (double) hux[jj+1][nu+ii];

		// copy back lagrangian multipliers
		for(jj=0; jj<N; jj++)
			for(ii=0; ii<nx; ii++)
				pi[ii+nx*jj] = (double) hpi[jj+1][ii];


		
		}
	else
		{
		printf("\nUnsopported precision type: %s\n\n", &prec);
		return -1;
		}
	
	return 0;
	
	}



int fortran_order_riccati_mhe( const char prec, const int smooth,
                               const int nx, const int nw, const int ny, const int N,
                               double *A, double *G, double *C, double *f, 
                               double *Q, double *R, double *q, double *r, 
                               double *y, double *x0, double *L0,
                               double *xe, double *Le, double *w, double *lam,
                               double *work0 )
	{

//	printf("\nenter wrapper\n");

	int hpmpc_status = -1;

	//char prec = 'd';

	if(prec=='d')
		{

		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = bs*ncl;

		const int nt = nx+ny; 
		const int ant = nal*((nt+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);
		const int anw = nal*((nw+nal-1)/nal);
		const int any = nal*((ny+nal-1)/nal);
		const int pnt = bs*((nt+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int pnw = bs*((nw+bs-1)/bs);
		const int pny = bs*((ny+bs-1)/bs);
		const int cnt = ncl*((nt+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int cnw = ncl*((nw+ncl-1)/ncl);
		const int cny = ncl*((ny+ncl-1)/ncl);
		const int cnf = cnt<cnx+ncl ? cnx+ncl : cnt;

		const int pad = (ncl-(nx+nw)%ncl)%ncl; // packing between AGL & P
		const int cnj = nx+nw+pad+cnx;

		//double *work0 = (double *) malloc((8 + (N+1)*(pnx*cnx+pnx*cnw+pny*cnx+5*anx+pnw*cnw+pny*cny+2*anw+2*any+pnx*cnj+pnt*cnf) + 2*pny*cnx+pnt*cnt+ant+pnw*cnw+pnx*cnx)*sizeof(double));
//		printf("\nwork space allocated\n");

//		int compute_mult = 1; // compute multipliers

		int i, ii, jj, ll;



		/* align work space */
		size_t align = 64;
		size_t addr = (size_t) work0;
		size_t offset = addr % 64;
		double *ptr = work0 + offset / 8;

		//for(ii=0; ii<(N+1)*(pnx*cnx+pnx*cnw+pny*cnx+5*anx+pnw*cnw+pny*cny+2*anw+2*any+pnx*cnj+pnt*cnf) + 2*pny*cnx+pnt*cnt+ant+pnw*cnw+pnx*cnx; ii++)
		//	ptr[ii] = 0.0;

		/* array or pointers */
		double *(hpA[N]);
		double *(hpG[N]);
		double *(hpC[N+1]);
		double *(hf[N]);
		double *(hpQ[N]);
		double *(hpR[N+1]);
		double *(hq[N]);
		double *(hr[N+1]);
		double *(hpLp[N+1]);
		double *(hdLp[N+1]);
		double *(hpLe[N+1]);
		double *(hxe[N+1]);
		double *(hxp[N+1]);
		double *(hw[N]);
		double *(hy[N+1]);
		double *(hlam[N]);

		double *diag;
		double *work;


		for(ii=0; ii<N; ii++)
			{
			// dynamic system
			hpA[ii] = ptr;
			ptr += pnx*cnx;
			hpG[ii] = ptr;
			ptr += pnx*cnw;
			hpC[ii] = ptr;
			ptr += pny*cnx;
			hf[ii] = ptr;
			ptr += anx;
			// cost function
			hpQ[ii] = ptr;
			ptr += pnw*cnw;
			hpR[ii] = ptr;
			ptr += pny*cny;
			hq[ii] = ptr;
			ptr += anw;
			hr[ii] = ptr;
			ptr += any;
			// covariances
			hpLp[ii] = ptr;
			ptr += pnx*cnj;
			hdLp[ii] = ptr;
			ptr += anx;
			hpLe[ii] = ptr;
			ptr += pnt*cnf;
			// estimates and measurements
			hxe[ii] = ptr;
			ptr += anx;
			hxp[ii] = ptr;
			ptr += anx;
			hw[ii] = ptr;
			ptr += anw;
			hy[ii] = ptr;
			ptr += any;
			hlam[ii] = ptr;
			ptr += anx;

		//if(ii==1)
		//for(jj=0; jj<(N-ii)*(pnx*cnx+pnx*cnw+pny*cnx+5*anx+pnw*cnw+pny*cny+2*anw+2*any+pnx*cnj+pnt*cnf) + 2*pny*cnx+pnt*cnt+ant+pnw*cnw+pnx*cnx; jj++)
		//	ptr[jj] = 0.0;

			}
		// stage N
		// dynamic system
		hpC[N] = ptr;
		ptr += pny*cnx;
		// cost function
		hpR[N] = ptr;
		ptr += pny*cny;
		hr[N] = ptr;
		ptr += any;
		// covariances
		hpLp[N] = ptr;
		ptr += pnx*cnj;
		hdLp[N] = ptr;
		ptr += anx;
		hpLe[N] = ptr;
		ptr += pnt*cnf;
		// estimates and measurements
		hxe[N] = ptr;
		ptr += anx;
		hxp[N] = ptr;
		ptr += anx;
		hy[N] = ptr;
		ptr += any;

		// diagonal backup
		diag = ptr;
		ptr += ant;

		// work space
		work = ptr;
		ptr += 2*pny*cnx+pnt*cnt+ant+pnw*cnw+pnx*cnx;
		//for(ii=0; ii<2*pny*cnx+pnt*cnt+ant+pnw*cnw+pnx*cnx; ii++)
		//	work[ii] = 0.0;

//		printf("\nmatrix space allocated\n");


		// convert into panel matrix format

		// stage 0
		// covariances
		//d_print_mat(nx, nx, L0, nx);
		d_cvt_mat2pmat(nx, nx, 0, bs, L0, nx, hpLp[0]+(nx+nw+pad)*bs, cnj);
		//d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[0], cnj);
		// estimates 
		for(jj=0; jj<nx; jj++) hxp[0][jj] = x0[jj];
		//d_print_mat(1, nx, hxp[0], 1);
		// stages 0 to N-1
		for(ii=0; ii<N; ii++)
			{
			//printf("\nii = %d\n", ii);
			// dynamic system
			//d_print_mat(nx, nx, A+ii*nx*nx, nx);
			d_cvt_mat2pmat(nx, nx, 0, bs, A+ii*nx*nx, nx, hpA[ii], cnx);
			d_cvt_mat2pmat(nx, nw, 0, bs, G+ii*nx*nw, nx, hpG[ii], cnw);
			d_cvt_mat2pmat(ny, nx, 0, bs, C+ii*ny*nx, ny, hpC[ii], cnx);
			for(jj=0; jj<nx; jj++) hf[ii][jj] = f[ii*nx+jj];
			// cost function
			d_cvt_mat2pmat(nw, nw, 0, bs, Q+ii*nw*nw, nw, hpQ[ii], cnw);
			d_cvt_mat2pmat(ny, ny, 0, bs, R+ii*ny*ny, ny, hpR[ii], cny);
			for(jj=0; jj<nw; jj++) hq[ii][jj] = q[ii*nw+jj];
			for(jj=0; jj<ny; jj++) hr[ii][jj] = r[ii*ny+jj];
			// measurements
			for(jj=0; jj<ny; jj++) hy[ii][jj] = y[ii*ny+jj];
			}
		// stage N
		// dynamic system
		d_cvt_mat2pmat(ny, nx, 0, bs, C+N*ny*nx, ny, hpC[N], cnx);
		// cost function
		d_cvt_mat2pmat(ny, ny, 0, bs, R+N*ny*ny, ny, hpR[N], cny);
		for(jj=0; jj<ny; jj++) hr[N][jj] = r[N*ny+jj];
		// measurements
		for(jj=0; jj<ny; jj++) hy[N][jj] = y[N*ny+jj];
		//d_print_mat(1, ny, hy[0], 1);

#if 0
		printf("\nmatrices converted\n");
		printf("\nn = 0\n");
		d_print_pmat(nx, nx, bs, hpA[0], cnx);
		d_print_pmat(nx, nw, bs, hpG[0], cnw);
		d_print_pmat(ny, nx, bs, hpC[0], cnx);
		d_print_pmat(nw, nw, bs, hpQ[0], cnw);
		d_print_pmat(ny, ny, bs, hpR[0], cny);
		d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[0], cnj);
		d_print_pmat(nt, nt, bs, hpLe[0], cnf);
		printf("\nn = 1\n");
		d_print_pmat(nt, nt, bs, hpLe[1], cnf);
		d_print_pmat(nx, nx, bs, hpA[1], cnx);
		d_print_pmat(nx, nw, bs, hpG[1], cnw);
		d_print_pmat(ny, nx, bs, hpC[1], cnx);
		d_print_pmat(nw, nw, bs, hpQ[1], cnw);
		d_print_pmat(ny, ny, bs, hpR[1], cny);
		d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[1], cnj);
		d_print_pmat(nt, nt, bs, hpLe[1], cnf);
#endif


		// call Riccati solver
		// factorization
		d_ric_trf_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, work);
		// solution
		//d_ric_trs_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, hq, hr, hf, hxp, hxe, hw, hy, 0, hlam, work1);
		// smoothed solution
		hpmpc_status = d_ric_trs_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, hq, hr, hf, hxp, hxe, hw, hy, smooth, hlam, work);

#if 0
		//d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[N], cnj);
		//d_print_pmat(nt, nt, bs, hpLe[N], cnf);
		printf("\nxp = \n");
		d_print_mat(1, nx, hxp[0], 1);
		d_print_mat(1, nx, hxp[1], 1);
		d_print_mat(1, nx, hxp[2], 1);
		printf("\nxe = \n");
		d_print_mat(1, nx, hxe[0], 1);
		d_print_mat(1, nx, hxe[1], 1);
		d_print_mat(1, nx, hxe[2], 1);
		printf("\nsystem solved\n");
		d_print_pmat(nx, nx, bs, hpLp[0]+(nx+nw+pad)*bs, cnj);
		d_print_pmat(nx, nx, bs, hpLp[1]+(nx+nw+pad)*bs, cnj);
		d_print_pmat(nx, nx, bs, hpLp[2]+(nx+nw+pad)*bs, cnj);
		d_print_pmat(nt, nt, bs, hpLe[0], cnf);
		d_print_pmat(nt, nt, bs, hpLe[1], cnf);
		d_print_pmat(nt, nt, bs, hpLe[2], cnf);
		return;
#endif


		// copy back estimate and covariance at first stage (Extended Kalma Filter update of initial condition)
		for(jj=0; jj<nx; jj++) x0[jj] = hxp[1][jj];
		d_cvt_pmat2mat(nx, nx, 0, bs, hpLp[1]+(nx+nw+pad)*bs, cnj, L0, nx);


		// copy back estimates at all stages 0,1,...,N
		for(ii=0; ii<=N; ii++)
			for(jj=0; jj<nx; jj++)
				xe[ii*nx+jj] = hxe[ii][jj];

		// copy back process disturbance at all stages 0,1,...,N-1
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nw; jj++)
				w[ii*nw+jj] = hw[ii][jj];
			
		// copy back mulipliers at all stages 0,1,...,N-1
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nx; jj++)
				lam[ii*nx+jj] = hlam[ii][jj];

		// copy back covariance at last stage
		d_cvt_pmat2mat(nx, nx, ny, bs, hpLe[N]+(ny/bs)*bs*cnf+ny%bs+ny*bs, cnf, Le, nx);


		
		}
	else
		{
		printf("\nUnsopported precision type: %s\n\n", &prec);
		return -1;
		}

	return hpmpc_status;
	
	}



int fortran_order_riccati_mhe_if( char prec, int alg,
                                  int nx, int nw, int ny, int N,
                                  double *A, double *G, double *C, double *f, 
                                  double *R, double *Q, double *Qf, double *r, double *q, double *qf,
                                  double *y, double *x0, double *L0,
                                  double *xe, double *Le, double *w, double *lam,
                                  double *work0 )
	{

//	printf("\nenter wrapper\n");

	if(alg!=0 && alg!=1 && alg!=2)
		{
		printf("\nUnsopported algorithm type: %d\n\n", alg);
		return -2;
		}

	int hpmpc_status = 0;

	//char prec = 'd';

	if(prec=='d')
		{

		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = bs*ncl;

		const int nwx = nw+nx; 
		const int anx = nal*((nx+nal-1)/nal);
		const int anw = nal*((nw+nal-1)/nal);
		const int any = nal*((ny+nal-1)/nal);
		const int pnx = bs*((nx+bs-1)/bs);
		const int pny = bs*((ny+bs-1)/bs);
		const int pnx2 = bs*((2*nx+bs-1)/bs);
		const int pnwx = bs*((nwx+bs-1)/bs);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int cnw = ncl*((nw+ncl-1)/ncl);
		const int cny = ncl*((ny+ncl-1)/ncl);
		const int cnx2 = 2*(ncl*((nx+ncl-1)/ncl));

		const int pad = (ncl-(nx+nw)%ncl)%ncl; // padding
		const int cnj = nx+nw+pad+cnx;

//		double *work0 = (double *) malloc(hpmpc_ric_mhe_if_dp_work_space(nx, nw, ny, N)*sizeof(double));
//		printf("\nwork space allocated\n");

//		int compute_mult = 1; // compute multipliers

		int i, ii, jj, ll;



		/* align work space */
		size_t align = 64;
		size_t addr = (size_t) work0;
		size_t offset = addr % 64;
		double *ptr = work0 + offset / 8;

		//for(ii=0; ii<(N+1)*(pnx*cnx+pnx*cnw+pny*cnx+5*anx+pnw*cnw+pny*cny+2*anw+2*any+pnx*cnj+pnt*cnf) + 2*pny*cnx+pnt*cnt+ant+pnw*cnw+pnx*cnx; ii++)
		//	ptr[ii] = 0.0;

		/* array or pointers */
		double *(hpRG[N]);
		double *(hpQA[N+1]);
		double *(hpCt[N+1]);
		double *(hpGLr[N]);
		double *(hpALe[N+1]);

		double *(hf[N]);
		double *(hr[N]);
		double *(hq[N+1]);
		double *(hy[N+1]);
		double *(hxe[N+1]);
		double *(hxp[N+1]);
		double *(hw[N]);
		double *(hlam[N]);

		double *pL0; 
		double *pL0_inv;
		double *Q_temp;
		double *q_temp;
		double *Ct_temp;

		double *diag;
		double *work;



		//if(1 || alg==0)
		if(alg==0)
			{
			Q_temp = ptr;
			ptr += pny*cny;
			Ct_temp = ptr;
			ptr += pnx*cny;
			diag = ptr;
			ptr += anx;
			}

		for(ii=0; ii<N; ii++)
			{
			hpRG[ii] = ptr;
			ptr += pnwx*cnw;
			d_cvt_mat2pmat(nw, nw, 0, bs, R+ii*nw*nw, nw, hpRG[ii], cnw);
			d_cvt_mat2pmat(nx, nw, nw, bs, G+ii*nx*nw, nx, hpRG[ii]+(nw/bs)*bs*cnw+nw%bs, cnw);
			}
		//d_print_pmat(nx+nw, nw, bs, hpQG[0], cnw);

		for(ii=0; ii<N; ii++)
			{
			hpQA[ii] = ptr;
			ptr += pnx2*cnx;
			if(alg==2)
				{
				d_cvt_mat2pmat(nx, nx, 0, bs, Q+ii*nx*nx, nx, hpQA[ii], cnx);
				}
			else // if(alg==0 || alg==1)
				{
				for(jj=0; jj<pnx*cnx; jj++) hpQA[ii][jj] = 0.0;
				d_cvt_mat2pmat(ny, ny, 0, bs, Q+ii*ny*ny, ny, hpQA[ii], cnx);
				}
			d_cvt_mat2pmat(nx, nx, nx, bs, A+ii*nx*nx, nx, hpQA[ii]+(nx/bs)*bs*cnx+nx%bs, cnx);
			if(alg==0)
				{
				hpCt[ii] = ptr;
				ptr += pnx*cny;
				d_cvt_tran_mat2pmat(ny, nx, 0, bs, C+ii*ny*nx, ny, hpCt[ii], cny);
				dpotrf_lib(ny, ny, hpQA[ii], cnx, Q_temp, cny, diag);
				dtrtr_l_lib(ny, 0, Q_temp, cny, Q_temp, cny);	
				dtrmm_l_lib(nx, ny, hpCt[ii], cny, Q_temp, cny, Ct_temp, cny);
				dsyrk_nt_lib(nx, nx, ny, Ct_temp, cny, Ct_temp, cny, hpQA[ii], cnx, hpQA[ii], cnx, 0);
				//d_print_pmat(nx, nx, bs, hpQA[ii], cnx);
				//return 0;
				}
			}
		hpQA[N] = ptr;
		ptr += pnx*cnx;
		if(alg==2)
			{
			//d_cvt_mat2pmat(nx, nx, 0, bs, Q+N*nx*nx, nx, hpQA[N], cnx);
			d_cvt_mat2pmat(nx, nx, 0, bs, Qf, nx, hpQA[N], cnx);
			}
		else // if(alg==0 || alg==1)
			{
			for(jj=0; jj<pnx*cnx; jj++) hpQA[N][jj] = 0.0;
			//d_cvt_mat2pmat(ny, ny, 0, bs, Q+N*ny*ny, ny, hpQA[N], cnx);
			d_cvt_mat2pmat(ny, ny, 0, bs, Qf, ny, hpQA[N], cnx);
			}
		if(alg==0)
			{
			hpCt[N] = ptr;
			ptr += pnx*cny;
			d_cvt_tran_mat2pmat(ny, nx, 0, bs, C+N*ny*nx, ny, hpCt[N], cny);
			dpotrf_lib(ny, ny, hpQA[N], cnx, Q_temp, cny, diag);
			dtrtr_l_lib(ny, 0, Q_temp, cny, Q_temp, cny);	
			dtrmm_l_lib(nx, ny, hpCt[N], cny, Q_temp, cny, Ct_temp, cny);
			dsyrk_nt_lib(nx, nx, ny, Ct_temp, cny, Ct_temp, cny, hpQA[N], cnx, hpQA[N], cnx, 0);
			//d_print_pmat(nx, nx, bs, hpQA[ii], cnx);
			//return 0;
			}
		//d_print_pmat(nx+nx, nx, bs, hpQA[0], cnx);
		//d_print_pmat(nx, nx, bs, hpQA[N], cnx);

		for(ii=0; ii<N; ii++)
			{
			hpGLr[ii] = ptr;
			ptr += pnwx*cnw;
			}

		for(ii=0; ii<=N; ii++)
			{
			hpALe[ii] = ptr;
			ptr += pnx2*cnx2;
			}
		pL0 = ptr;
		ptr += pnx*cnx; // TODO use work space ???
		d_cvt_mat2pmat(nx, nx, 0, bs, L0, nx, pL0, cnx);
		dtrinv_lib(nx, pL0, cnx, hpALe[0], cnx2);
		//d_print_pmat(nx, nx, bs, hpALe[0], cnx2);

		for(ii=0; ii<N; ii++)
			{
			hf[ii] = ptr;
			ptr += anx;
			for(jj=0; jj<nx; jj++) hf[ii][jj] = f[ii*nx+jj];
			}

		for(ii=0; ii<N; ii++)
			{
			hr[ii] = ptr;
			ptr += anw;
			for(jj=0; jj<nw; jj++) hr[ii][jj] = r[ii*nw+jj];
			}

		if(alg==0 || alg==1)
			{
			for(ii=0; ii<=N; ii++)
				{
				hy[ii] = ptr;
				ptr += any;
				for(jj=0; jj<ny; jj++) hy[ii][jj] = y[ii*ny+jj];
				}
			}

		//d_print_mat(nx, N+1, hr[0], anx);
		if(alg==2)
			{
			//for(ii=0; ii<=N; ii++)
			for(ii=0; ii<N; ii++)
				{
				hq[ii] = ptr;
				ptr += anx;
				for(jj=0; jj<nx; jj++) hq[ii][jj] = q[ii*nx+jj];
				}
			hq[N] = ptr;
			ptr += anx;
			for(jj=0; jj<nx; jj++) hq[N][jj] = qf[jj];
			}
		else // if(alg==0 || alg==1)
			{
			q_temp = ptr;
			ptr += any;
			//for(ii=0; ii<=N; ii++)
			for(ii=0; ii<N; ii++)
				{
				hq[ii] = ptr;
				ptr += anx;
				for(jj=0; jj<ny; jj++) q_temp[jj] = - q[ii*ny+jj];
				//d_print_pmat(nx, nx, bs, hpQA[ii], cnx);
				//d_print_mat(1, nx, q_temp, 1);
				dsymv_lib(ny, ny, hpQA[ii], cnx, hy[ii], q_temp, -1);
				//d_print_mat(1, nx, q_temp, 1);
				if(alg==0)
					{
					dgemv_n_lib(nx, ny, hpCt[ii], cny, q_temp, hq[ii], 0);
					}
				else
					{
					for(jj=0; jj<ny; jj++) hq[ii][jj] = q_temp[jj];
					for( ; jj<nx; jj++) hq[ii][jj] = 0;
					}
				}
			hq[N] = ptr;
			ptr += anx;
			for(jj=0; jj<ny; jj++) q_temp[jj] = - qf[jj];
			//d_print_pmat(nx, nx, bs, hpQA[ii], cnx);
			//d_print_mat(1, nx, q_temp, 1);
			dsymv_lib(ny, ny, hpQA[N], cnx, hy[N], q_temp, -1);
			//d_print_mat(1, nx, q_temp, 1);
			if(alg==0)
				{
				dgemv_n_lib(nx, ny, hpCt[N], cny, q_temp, hq[N], 0);
				}
			else
				{
				for(jj=0; jj<ny; jj++) hq[N][jj] = q_temp[jj];
				for( ; jj<nx; jj++) hq[N][jj] = 0;
				}
			}
		//d_print_pmat(nx, ny, bs, hpCt[0], cny);
		//d_print_mat(nx, N+1, hq[0], anx);

		for(ii=0; ii<=N; ii++)
			{
			hxp[ii] = ptr;
			ptr += anx;
			}
		for(jj=0; jj<nx; jj++) hxp[0][jj] = x0[jj];

		for(ii=0; ii<=N; ii++)
			{
			hxe[ii] = ptr;
			ptr += anx;
			}

		for(ii=0; ii<N; ii++)
			{
			hw[ii] = ptr;
			ptr += anw;
			}

		for(ii=0; ii<=N; ii++)
			{
			hlam[ii] = ptr;
			ptr += anx;
			}

		work = ptr;
		ptr += pnx*cnj+anx;



		// factorize KKT matrix
		hpmpc_status = d_ric_trf_mhe_if(nx, nw, N, hpQA, hpRG, hpALe, hpGLr, work);

		if(hpmpc_status!=0)
			return hpmpc_status;

		// solve KKT system
		d_ric_trs_mhe_if(nx, nw, N, hpALe, hpGLr, hq, hr, hf, hxp, hxe, hw, hlam, work);



		// residuals computation
		//#if 1
		#ifdef DEBUG_MODE
		printf("\nstart of print residuals\n\n");
		double *(hq_res[N+1]);
		double *(hr_res[N]);
		double *(hf_res[N]);
		double *p_hq_res; d_zeros_align(&p_hq_res, anx, N+1);
		double *p_hr_res; d_zeros_align(&p_hr_res, anw, N);
		double *p_hf_res; d_zeros_align(&p_hf_res, anx, N);

		for(jj=0; jj<N; jj++)
			{
			hq_res[jj] = p_hq_res+jj*anx;
			hr_res[jj] = p_hr_res+jj*anw;
			hf_res[jj] = p_hf_res+jj*anx;
			}
		hq_res[N] = p_hq_res+N*anx;

		double *pL0_inv2; d_zeros_align(&pL0_inv2, pnx, cnx);
		dtrinv_lib(nx, pL0, cnx, pL0_inv2, cnx);

		double *p0; d_zeros_align(&p0, anx, 1);
		double *x_temp; d_zeros_align(&x_temp, anx, 1);
		dtrmv_u_t_lib(nx, pL0_inv2, cnx, x0, x_temp, 0);
		dtrmv_u_n_lib(nx, pL0_inv2, cnx, x_temp, p0, 0);

		d_res_mhe_if(nx, nw, N, hpQA, hpRG, pL0_inv2, hq, hr, hf, p0, hxe, hw, hlam, hq_res, hr_res, hf_res, work);

		d_print_mat(nx, N+1, hq_res[0], anx);
		d_print_mat(nw, N, hr_res[0], anw);
		d_print_mat(nx, N, hf_res[0], anx);

		free(p_hq_res);
		free(p_hr_res);
		free(p_hf_res);
		free(pL0_inv2);
		free(p0);
		free(x_temp);
		printf("\nend of print residuals\n\n");
		#endif



		// copy back estimate and covariance at first stage (Extended Kalman Filter update of initial condition)
		for(jj=0; jj<nx; jj++) x0[jj] = hxp[1][jj];

		// save L0 for next step
		pL0_inv = ptr;
		ptr += pnx*cnx; // TODO use work space ??? remove ???
		dgetr_lib(nx, 0, nx, 0, hpALe[1], cnx2, pL0_inv, cnx); // TODO write dtrtr_u to transpose in place
		dtrinv_lib(nx, pL0_inv, cnx, pL0, cnx);
		d_cvt_tran_pmat2mat(nx, nx, 0, bs, pL0, cnx, L0, nx);


		// copy back estimates at all stages 0,1,...,N
		for(ii=0; ii<=N; ii++)
			for(jj=0; jj<nx; jj++)
				xe[ii*nx+jj] = hxe[ii][jj];

		// copy back process disturbance at all stages 0,1,...,N-1
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nw; jj++)
				w[ii*nw+jj] = hw[ii][jj];
			
		// copy back multipliers at all stages 0,1,...,N-1
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nx; jj++)
				lam[ii*nx+jj] = hlam[ii][jj];

		// copy back cholesky factor of information matrix at last stage
		//d_print_pmat(pnx2, cnx2, bs, hpALe[N], cnx2);
		d_cvt_pmat2mat(nx, nx, 0, bs, hpALe[N]+cnx*bs, cnx2, Le, nx);
		for(jj=0; jj<nx; jj++) Le[jj*(nx+1)] = 1.0/Le[jj*(nx+1)];


		//d_print_pmat(nx, nx, bs, hpALe[1], cnx2);
		//d_print_pmat(pnx2, cnx2, bs, hpALe[N-1], cnx2);
		//d_print_pmat(pnx2, cnx2, bs, hpALe[N], cnx2);
		//d_print_pmat(pnwx, cnw, bs, hpGLq[N-1], cnw);
		//d_print_pmat(nx, nx, bs, pL0_inv, cnx);
		//d_print_pmat(nx, nx, bs, pL0, cnx);
		//d_print_mat(nx, nx, L0, nx);
		//d_print_mat(nx, N+1, hxe[0], anx);


//		free(work0); // TODO remove
		
		}
	else
		{
		printf("\nUnsopported precision type: %s\n\n", &prec);
		return -1;
		}

	return hpmpc_status;
	
	}



int fortran_order_admm_box_mpc( int k_max, double tol,
                                     double rho, double alpha,
                                     int nx, int nu, int N,
                                     double* A, double* B, double* b, 
                                     double* Q, double* Qf, double* S, double* R, 
                                     double* q, double* qf, double* r, 
                                     double* lb, double* ub, 
                                     double* x, double* u,
                                     int* nIt, double *stat )

	{

/*printf("\nstart of wrapper\n");*/

    char prec = PREC;

    if(prec=='d')
	    {
	    
		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = D_MR*D_NCL;

		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		const int nb = nx+nu; // number of box constraints
/*		const int pnb = bs*((2*nb+bs-1)/bs);*/
		const int anb = nal*((2*nb+nal-1)/nal);

		// work space
        double *work0 = (double *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 10*anz + 7*anx) + 3*anz)*sizeof(double));

		// parameters
/*		double rho = 10.0; // penalty parameter*/
/*		double alpha = 1.9; // relaxation parameter*/
        int warm_start = 0;//WARM_START;
        int compute_mult = 0;//1; // compute multipliers
        
        int info = 0;

        int i, ii, jj, ll;


        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        double *ptr = work0 + offset / 8;

        /* array or pointers */
        double *(hpBAbt[N]);
        double *(hpQ[N+1]);
        double *(hux[N+1]);
		double *(hux_v[N+1]);
		double *(hux_w[N+1]);
		double *(hlb[N+1]);
		double *(hub[N+1]);
        double *(hpi[N+1]);

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_v[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_w[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hlb[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hub[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        /* pack matrices 	*/

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            d_cvt_tran_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
            d_cvt_tran_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
	        }
/*	    d_print_pmat(nx+nu+1, nx, bs, hpBAbt[0], cnx);*/
/*	    d_print_pmat(nx+nu+1, nx, bs, hpBAbt[1], cnx);*/
/*	    d_print_pmat(nx+nu+1, nx, bs, hpBAbt[N-1], cnx);*/

/*return 1;*/
        // cost function
        for(jj=0; jj<N; jj++)
	        {
            d_cvt_tran_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
            d_cvt_tran_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
            d_cvt_tran_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
            for(ii=0; ii<nu; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = r[ii+jj*nu];
            for(ii=0; ii<nx; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = q[ii+nx*jj];
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        d_cvt_tran_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
        for(jj=0; jj<nx; jj++)
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = qf[jj];

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				hlb[jj][ii] = lb[ii+nu*jj];
				hub[jj][ii] = ub[ii+nu*jj];
				}
			}
		// state constraints 
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nx; ii++)
				{
				hlb[jj+1][nu+ii] = lb[N*nu+ii+nx*jj];
				hub[jj+1][nu+ii] = ub[N*nu+ii+nx*jj];
				}
			}



        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = x[ii+nx*jj];



/*printf("\nstart of ip solver\n");*/

        // call the soft ADMM solver
		d_admm_box_mpc(nIt, k_max, tol, tol, warm_start, 1, rho, alpha, stat, nx, nu, N, hpBAbt, hpQ, hlb, hub, hux, hux_v, hux_w, compute_mult, hpi, ptr);

/*printf("\nend of ip solver\n");*/


        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = hux[jj][nu+ii];

#if PC_DEBUG == 1
        for(jj=0; jj<*nIt; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */

		free(work0);

	    }
    else if(prec=='s')
	    {

		const int bs = S_MR; //d_get_mr();
		const int ncl = S_NCL;
		const int nal = S_MR*S_NCL;

		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		const int nb = nx+nu; // number of box constraints
/*		const int pnb = bs*((2*nb+bs-1)/bs);*/
		const int anb = nal*((2*nb+nal-1)/nal);

		// work space
        float *work0 = (float *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 9*anz + 7*anx) + 3*anz)*sizeof(float));

		// parameters
/*		float rho = 10.0; // penalty parameter*/
/*		float alpha = 1.9; // relaxation parameter*/
        int warm_start = 0;//WARM_START;
        int compute_mult = 0;//1; // compute multipliers
        
        int info = 0;

        int i, ii, jj, ll;


        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        float *ptr = work0 + offset / 4;

        /* array or pointers */
        float *(hpBAbt[N]);
        float *(hpQ[N + 1]);
        float *(hux[N + 1]);
		float *(hux_v[N+1]);
		float *(hux_w[N+1]);
		float *(hlb[N+1]);
		float *(hub[N+1]);
        float *(hpi[N + 1]);

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_v[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_w[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hlb[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hub[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        /* pack matrices 	*/

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            cvt_tran_d2s_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
            cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = (float) b[ii*nx+jj];
	        }

        // cost function
        for(jj=0; jj<N; jj++)
	        {
            cvt_tran_d2s_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
            cvt_tran_d2s_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
            cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
            for(ii=0; ii<nu; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = (float) r[ii+jj*nu];
            for(ii=0; ii<nx; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = (float) q[ii+nx*jj];
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
        for(jj=0; jj<nx; jj++)
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = (float) qf[jj];

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				hlb[jj][ii] = (float) lb[ii+nu*jj];
				hub[jj][ii] = (float) ub[ii+nu*jj];
				}
			}
		// state constraints 
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nx; ii++)
				{
				hlb[jj+1][nu+ii] = (float) lb[N*nu+ii+nx*jj];
				hub[jj+1][nu+ii] = (float) ub[N*nu+ii+nx*jj];
				}
			}



        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = (float) u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = (float) x[ii+nx*jj];



        // call the soft ADMM solver
		s_admm_box_mpc(nIt, k_max, (float) tol, (float) tol, warm_start, 1, (float) rho, (float) alpha, (float *)stat, nx, nu, N, hpBAbt, hpQ, hlb, hub, hux, hux_v, hux_w, compute_mult, hpi, ptr);



		// convert stat into double (start fom end !!!)
		float *ptr_stat = (float *) stat;
		for(ii=5*k_max-1; ii>=0; ii--)
			{
			stat[ii] = (double) ptr_stat[ii];
			}



        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = (double) hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = (double) hux[jj][nu+ii];


#if PC_DEBUG == 1
        for (jj = 0; jj < *nIt; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */

		free(work0);

 	   }

/*printf("\nend of wrapper\n");*/

    return 0;

	}



int fortran_order_admm_soft_wrapper( int k_max, double tol,
                                     double rho, double alpha,
                                     const int nx, const int nu, const int N,
                                     double* A, double* B, double* b, 
                                     double* Q, double* Qf, double* S, double* R, 
                                     double* q, double* qf, double* r, 
                                     double* Z, double *z,
                                     double* lb, double* ub, 
                                     double* x, double* u,
                                     int* nIt, double *stat )

	{

/*printf("\nstart of wrapper\n");*/

    char prec = PREC;

    if(prec=='d')
	    {
	    
		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = D_MR*D_NCL;

		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		const int nb = nx+nu; // number of box constraints
/*		const int pnb = bs*((2*nb+bs-1)/bs);*/
		const int anb = nal*((2*nb+nal-1)/nal);

		// work space
        double *work0 = (double *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 10*anz + 17*anx) + 3*anz)*sizeof(double));

		// parameters
/*		double rho = 10.0; // penalty parameter*/
/*		double alpha = 1.9; // relaxation parameter*/
        int warm_start = 0;//WARM_START;
        int compute_mult = 0;//1; // compute multipliers
        
        int info = 0;

        int i, ii, jj, ll;


        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        double *ptr = work0 + offset / 8;

        /* array or pointers */
        double *(hpBAbt[N]);
        double *(hpQ[N + 1]);
		double *(hZ[N+1]);
		double *(hz[N+1]);
        double *(hux[N + 1]);
		double *(hux_v[N+1]);
		double *(hux_w[N+1]);
		double *(hlb[N+1]);
		double *(hub[N+1]);
		double *(hs_u[N+1]);
		double *(hs_v[N+1]);
		double *(hs_w[N+1]);
        double *(hpi[N + 1]);

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hZ[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hz[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_v[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_w[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hlb[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hub[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hs_u[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hs_v[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hs_w[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        /* pack matrices 	*/

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            d_cvt_tran_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
            d_cvt_tran_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
	        }
/*	    d_print_pmat(nx+nu+1, nx, bs, hpBAbt[0], cnx);*/
/*	    d_print_pmat(nx+nu+1, nx, bs, hpBAbt[1], cnx);*/
/*	    d_print_pmat(nx+nu+1, nx, bs, hpBAbt[N-1], cnx);*/

/*return 1;*/
        // cost function
        for(jj=0; jj<N; jj++)
	        {
            d_cvt_tran_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
            d_cvt_tran_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
            d_cvt_tran_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
            for(ii=0; ii<nu; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = r[ii+jj*nu];
            for(ii=0; ii<nx; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = q[ii+nx*jj];
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        d_cvt_tran_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
        for(jj=0; jj<nx; jj++)
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = qf[jj];

		// soft constraints cost function
        for(jj=0; jj<=N; jj++)
	        {
			for(ii=0; ii<nx; ii++) hZ[jj][ii]     = Z[jj*2*nx+ii]; // upper
			for(ii=0; ii<nx; ii++) hZ[jj][anx+ii] = Z[jj*2*nx+nx+ii]; // lower
			for(ii=0; ii<nx; ii++) hz[jj][ii]     = z[jj*2*nx+ii]; // upper
			for(ii=0; ii<nx; ii++) hz[jj][anx+ii] = z[jj*2*nx+nx+ii]; // lower
			}

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				hlb[jj][ii] = lb[ii+nu*jj];
				hub[jj][ii] = ub[ii+nu*jj];
				}
			}
		// state constraints 
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nx; ii++)
				{
				hlb[jj+1][nu+ii] = lb[N*nu+ii+nx*jj];
				hub[jj+1][nu+ii] = ub[N*nu+ii+nx*jj];
				}
			}



        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = x[ii+nx*jj];



/*printf("\nstart of ip solver\n");*/

        // call the soft ADMM solver
	        d_admm_soft_mpc(nIt, k_max, tol, tol, warm_start, 1, rho, alpha, stat, nx, nu, N, hpBAbt, hpQ, hZ, hz, hlb, hub, hux, hux_v, hux_w, hs_u, hs_v, hs_w, compute_mult, hpi, ptr);

/*printf("\nend of ip solver\n");*/


        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = hux[jj][nu+ii];

#if PC_DEBUG == 1
        for (jj = 0; jj < *nIt; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */

		free(work0);

	    }
    else if(prec=='s')
	    {

		const int bs = S_MR; //d_get_mr();
		const int ncl = S_NCL;
		const int nal = S_MR*S_NCL;

		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		const int nb = nx+nu; // number of box constraints
/*		const int pnb = bs*((2*nb+bs-1)/bs);*/
		const int anb = nal*((2*nb+nal-1)/nal);

		// work space
        float *work0 = (float *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 9*anz + 17*anx) + 3*anz)*sizeof(float));

		// parameters
/*		float rho = 10.0; // penalty parameter*/
/*		float alpha = 1.9; // relaxation parameter*/
        int warm_start = 0;//WARM_START;
        int compute_mult = 0;//1; // compute multipliers
        
        int info = 0;

        int i, ii, jj, ll;


        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        float *ptr = work0 + offset / 4;

        /* array or pointers */
        float *(hpBAbt[N]);
        float *(hpQ[N + 1]);
		float *(hZ[N+1]);
		float *(hz[N+1]);
        float *(hux[N + 1]);
		float *(hux_v[N+1]);
		float *(hux_w[N+1]);
		float *(hlb[N+1]);
		float *(hub[N+1]);
		float *(hs_u[N+1]);
		float *(hs_v[N+1]);
		float *(hs_w[N+1]);
        float *(hpi[N + 1]);

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hZ[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hz[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_v[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_w[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hlb[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hub[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hs_u[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hs_v[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hs_w[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        /* pack matrices 	*/

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            cvt_tran_d2s_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
            cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = (float) b[ii*nx+jj];
	        }

        // cost function
        for(jj=0; jj<N; jj++)
	        {
            cvt_tran_d2s_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
            cvt_tran_d2s_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
            cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
            for(ii=0; ii<nu; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = (float) r[ii+jj*nu];
            for(ii=0; ii<nx; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = (float) q[ii+nx*jj];
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
        for(jj=0; jj<nx; jj++)
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = (float) qf[jj];

		// soft constraints cost function
        for(jj=0; jj<=N; jj++)
	        {
			for(ii=0; ii<nx; ii++) hZ[jj][ii]     = (float) Z[jj*2*nx+ii]; // upper
			for(ii=0; ii<nx; ii++) hZ[jj][anx+ii] = (float) Z[jj*2*nx+nx+ii]; // lower
			for(ii=0; ii<nx; ii++) hz[jj][ii]     = (float) z[jj*2*nx+ii]; // upper
			for(ii=0; ii<nx; ii++) hz[jj][anx+ii] = (float) z[jj*2*nx+nx+ii]; // lower
			}

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				hlb[jj][ii] = (float) lb[ii+nu*jj];
				hub[jj][ii] = (float) ub[ii+nu*jj];
				}
			}
		// state constraints 
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nx; ii++)
				{
				hlb[jj+1][nu+ii] = (float) lb[N*nu+ii+nx*jj];
				hub[jj+1][nu+ii] = (float) ub[N*nu+ii+nx*jj];
				}
			}



        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = (float) u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = (float) x[ii+nx*jj];



        // call the soft ADMM solver
		s_admm_soft_mpc(nIt, k_max, (float) tol, (float) tol, warm_start, 1, (float) rho, (float) alpha, (float *)stat, nx, nu, N, hpBAbt, hpQ, hZ, hz, hlb, hub, hux, hux_v, hux_w, hs_u, hs_v, hs_w, compute_mult, hpi, ptr);



		// convert stat into double (start fom end !!!)
		float *ptr_stat = (float *) stat;
		for(ii=5*k_max-1; ii>=0; ii--)
			{
			stat[ii] = (double) ptr_stat[ii];
			}



        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = (double) hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = (double) hux[jj][nu+ii];


#if PC_DEBUG == 1
        for (jj = 0; jj < *nIt; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */

		free(work0);

 	   }

/*printf("\nend of wrapper\n");*/

    return 0;

	}

