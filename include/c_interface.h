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

#include "target.h"
#include "block_size.h"



// MPC problem: define NX (state dimension), NU (input dimension), NN (horizon length)
// MHE problem: define NX (state dimension), NU (process disturbance dimension), NY (measurement dimension), NDN (number of equality constraints on last stage), NN (horizon length)


// define common quantities
#define NZ (NX+NU+1)
#define NT (NX+NY)
#ifndef NB
	#define NB (NU+NX) // number of two-sided box constraints
#endif
#ifndef NG
	#define NG 0 // number of two-sided general constraints
#endif
#ifndef NGN
	#define NGN 0 // number of two-sided general constraints at last stage
#endif
#ifndef NDN
	#define NDN 0 // number of equality constraints at last stage of MHE
#endif
// double precision constants
#define D_NAL (D_MR*D_NCL)
//#define D_PADX ((D_NCL-NX%D_NCL)%D_NCL) // padding after nx
#define D_PADXU ((D_NCL-(NX+NU)%D_NCL)%D_NCL) // padding after (nx+nw)
#define D_ANB (2*D_NAL*((NB+D_NAL-1)/(D_NAL))) // TODO change once updated lam and t order in soft constraints too
#define D_ANT (D_NAL*((NT+D_NAL-1)/(D_NAL)))
#define D_ANU (D_NAL*((NU+D_NAL-1)/(D_NAL)))
#define D_ANX (D_NAL*((NX+D_NAL-1)/(D_NAL)))
#define D_ANY (D_NAL*((NY+D_NAL-1)/(D_NAL)))
#define D_ANZ (D_NAL*((NZ+D_NAL-1)/(D_NAL)))
#define D_CNF (D_CNT<D_CNX+D_NCL ? D_CNX+D_NCL : D_CNT)
#define D_CNJ (NX+NU+D_PADXU+D_CNX)
//#define D_CNL (D_CNZ<D_CNX+D_NCL ? NX+D_PADX+D_CNX+D_NCL : NX+D_PADX+D_CNZ)
#define D_CNL (D_CNZ<D_CNX+D_NCL ? D_CNX+D_NCL : D_CNZ)
#define D_CNT (D_NCL*((NT+D_NCL-1)/D_NCL))
#define D_CNU (D_NCL*((NU+D_NCL-1)/D_NCL))
#define D_CNX (D_NCL*((NX+D_NCL-1)/D_NCL))
#define D_CNY (D_NCL*((NY+D_NCL-1)/D_NCL))
#define D_CNZ (D_NCL*((NZ+D_NCL-1)/D_NCL))
#define D_CNG (D_NCL*((NG+D_NCL-1)/D_NCL))
#define D_CNGN (D_NCL*((NGN+D_NCL-1)/D_NCL))
#define D_CNX2 (2*D_NCL*((NX+D_NCL-1)/D_NCL))
#define D_CNXG (D_NCL*((NX+NG+D_NCL-1)/D_NCL))
#define D_CNDN (D_NCL*((NDN+D_NCL-1)/D_NCL))
#define D_PNB (D_MR*((NB+D_MR-1)/D_MR))
#define D_PNG (D_MR*((NG+D_MR-1)/D_MR))
#define D_PNGN (D_MR*((NGN+D_MR-1)/D_MR))
#define D_PNT (D_MR*((NT+D_MR-1)/D_MR))
#define D_PNU (D_MR*((NU+D_MR-1)/D_MR))
#define D_PNX (D_MR*((NX+D_MR-1)/D_MR))
#define D_PNY (D_MR*((NY+D_MR-1)/D_MR))
#define D_PNZ (D_MR*((NZ+D_MR-1)/D_MR))
#define D_PNX2 (D_MR*((NX+NX+D_MR-1)/D_MR))
#define D_PNUX (D_MR*((NU+NX+D_MR-1)/D_MR))
#define D_PNDN (D_MR*((NDN+D_MR-1)/D_MR))
// single precision constants
#define S_NAL (S_MR*S_NCL)
#define S_PADX ((S_NCL-NX%S_NCL)%S_NCL) // padding between BAbtL & P
#define S_ANB (S_NAL*((2*NB+S_NAL-1)/(S_NAL)))
#define S_ANX (S_NAL*((NX+S_NAL-1)/(S_NAL)))
#define S_ANZ (S_NAL*((NZ+S_NAL-1)/(S_NAL)))
#define S_CNL (S_CNZ<S_CNX+S_NCL ? NX+S_PADX+S_CNX+S_NCL : NX+S_PADX+S_CNZ)
#define S_CNX (S_NCL*((NX+S_NCL-1)/S_NCL))
#define S_CNZ (S_NCL*((NZ+S_NCL-1)/S_NCL))
#define S_PNB (S_MR*((2*NB+S_MR-1)/S_MR))
#define S_PNX (S_MR*((NX+S_MR-1)/S_MR))
#define S_PNZ (S_MR*((NZ+S_MR-1)/S_MR))



// work space: static definition

// Riccati-based IP method for box-constrained MPC, double precision
#define HPMPC_IP_MPC_DP_WORK_SPACE (8 + (NN+1)*(D_PNZ*D_CNX + D_PNZ*D_CNZ + D_PNZ*D_CNL + D_PNZ*D_CNG + 8*D_ANZ + 4*D_ANX + 18*(D_PNB+D_PNG)) + D_PNZ*(D_CNGN-D_CNG) + 18*(D_PNGN-D_PNG) + D_ANZ + (D_CNGN<D_CNXG ? D_PNZ*D_CNXG : D_PNZ*D_CNGN ) )
// Riccati-based IP method for box-constrained MPC, single precision
#define HPMPC_IP_MPC_SP_WORK_SPACE (16 + (NN+1)*(S_PNZ*S_CNX + S_PNZ*S_CNZ + S_PNZ*S_CNL + 5*S_ANZ + 3*S_ANX + 7*S_ANB) + S_ANZ + D_PNZ*P_CNX)
// Riccati-based IP method for soft-constrained MPC, double precision
#define HPMPC_IP_SOFT_MPC_DP_WORK_SPACE (8 + (NN+1)*(D_PNZ*D_CNX + D_PNZ*D_CNZ + D_PNZ*D_CNL + 6*D_ANZ + 3*D_ANX + 17*D_ANB) + D_ANZ + D_PNZ*P_CNX)
// Riccati-based solver for unconstrained MPC, double precision
#define HPMPC_RIC_MPC_DP_WORK_SPACE (8 + (NN+1)*(D_PNZ*D_CNX + D_PNZ*D_CNZ + D_PNZ*D_CNL + 2*D_ANZ + 2*D_ANX) + D_ANZ + D_PNZ*P_CNX)
// Riccati-based solver for unconstrained MPC, single precision
#define HPMPC_RIC_MPC_SP_WORK_SPACE (16 + (NN+1)*(S_PNZ*S_CNX + S_PNZ*S_CNZ + S_PNZ*S_CNL + 2*S_ANZ + 2*S_ANX) + S_ANZ + D_PNZ*P_CNX)
// Riccati-based solver for unconstrained MHE, double precision
#define HPMPC_RIC_MHE_DP_WORK_SPACE (8 + (NN+1)*(D_PNX*D_CNX+D_PNX*D_CNU+D_PNY*D_CNX+5*D_ANX+D_PNU*D_CNU+D_PNY*D_CNY+2*D_ANU+2*D_ANY+D_PNX*D_CNJ+D_PNT*D_CNF) + 2*D_PNY*D_CNX+D_PNT*D_CNT+D_ANT+D_PNU*D_CNU+D_PNX*D_CNX)
// Riccati-based solver for unconstrained MHE, Information Filter version, double precision
#define HPMPC_RIC_MHE_IF_DP_WORK_SPACE (8 + (NN+1)*(D_PNUX*D_CNU+D_PNX2*D_CNX+D_PNUX*D_CNU+D_PNX2*D_CNX2+D_PNX*D_CNY+2*D_ANU+D_ANY+5*D_ANX) + 2*D_PNX*D_CNX+D_PNX*D_CNJ+D_ANX+D_PNY*D_CNY+D_PNX*D_CNY+D_ANX + D_PNDN*D_CNDN)

// work space: dynamic definition as function return value

// Riccati-based IP method for box-constrained MPC, double precision
int hpmpc_ip_hard_mpc_dp_work_space(int N, int nx, int nu, int nb, int ng, int ngN);

// Riccati-based IP method for box-constrained MPC, single precision
int hpmpc_ip_box_mpc_sp_work_space(int nx, int nu, int N);
    
// Riccati-based IP method for box-constrained MPC, double precision
int hpmpc_ip_soft_mpc_dp_work_space(int nx, int nu, int N);

// Riccati-based solver for unconstrained MPC, double precision
int hpmpc_ric_mpc_dp_work_space(int nx, int nu, int N);

// Riccati-based solver for unconstrained MPC, single precision
int hpmpc_ric_mpc_sp_work_space(int nx, int nu, int N);

// Riccati-based solver for unconstrained MHE, covariance filter version, double precision
int hpmpc_ric_mhe_dp_work_space(int nx, int nw, int ny, int N);

// Riccati-based solver for unconstrained MHE, information filter version, double precision
int hpmpc_ric_mhe_if_dp_work_space(int nx, int nw, int ny, int ndN, int N);



// c (or row-major) order
int c_order_ip_hard_mpc( int *kk, int k_max, double mu0, double mu_tol, char prec, int N, int nx, int nu, int nb, int ng, int ngf, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double* lb, double* ub, double *C, double *D, double* lg, double* ug, double *Cf, double *lgf, double *ugf, double* x, double* u, double *work0, double *stat, int compute_res, double *inf_norm_res, int compute_mult, double *pi, double *lam, double *t );

int c_order_ip_soft_mpc( int k_max, double mu_tol, const char prec, const int nx, const int nu, const int N, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double* lZ, double* uZ, double* lz, double* uz, double* lb, double* ub, double* x, double* u, double *work0, int* nIt, double *stat );

int c_order_riccati_mpc( const char prec, const int nx, const int nu, const int N, double *A, double *B, double *b, double *Q, double *Qf, double *S, double *R, double *q, double *qf, double *r, double *x, double *u, double *pi, double *work0 );

int c_order_riccati_mhe( const char prec, const int smooth, const int nx, const int nw, const int ny, const int N, double *A, double *G, double *C, double *f, double *Q, double *R, double *q, double *r, double *y, double *x0, double *L0, double *xe, double *Le, double *w, double *lam, double *work0 );

int c_order_riccati_mhe_if( char prec, int alg, int nx, int nw, int ny, int ndN, int N, double *A, double *G, double *C, double *f, double *D, double *d, double *R, double *Q, double *Qf, double *r, double *q, double *qf, double *y, double *x0, double *L0, double *xe, double *Le, double *w, double *lam, double *work0 );



// fortran (or column-major) order
int fortran_order_ip_hard_mpc( int *kk, int k_max, double mu0, double mu_tol, char prec, int N, int nx, int nu, int nb, int ng, int ngf, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double* lb, double* ub, double *C, double *D, double* lg, double* ug, double *Cf, double *lgf, double *ugf, double* x, double* u, double *work0, double *stat, int compute_res, double *inf_norm_res, int compute_mult, double *pi, double *lam, double *t );

int fortran_order_ip_soft_mpc( int k_max, double mu_tol, const char prec, const int nx, const int nu, const int N, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double* lZ, double* uZ, double* lz, double* uz, double* lb, double* ub, double* x, double* u, double *work0, int* nIt, double *stat );

int fortran_order_riccati_mpc( const char prec, const int nx, const int nu, const int N, double *A, double *B, double *b, double *Q, double *Qf, double *S, double *R, double *q, double *qf, double *r, double *x, double *u, double *pi, double *work0 );

int fortran_order_riccati_mhe( const char prec, const int smooth, const int nx, const int nw, const int ny, const int N, double *A, double *G, double *C, double *f, double *Q, double *R, double *q, double *r, double *y, double *x0, double *L0, double *xe, double *Le, double *w, double *lam, double *work0 );

int fortran_order_riccati_mhe_if( char prec, int alg, int nx, int nw, int ny, int ndN, int N, double *A, double *G, double *C, double *f, double *D, double *d, double *R, double *Q, double *Qf, double *r, double *q, double *qf, double *y, double *x0, double *L0, double *xe, double *Le, double *w, double *lam, double *work0 );
