#include "../include/c_interface.h"

#include <gtest/gtest.h>

#include <assert.h>
#include <stdio.h>
#include <memory.h>

#include "data/test_qp/qp0010.c"

double ** alloc_lambda(int N, int const * nb, int const * ng)
{
	double ** lambda = (double **)malloc((N + 1) * sizeof(double *));
	assert(lambda != NULL);
	
	for (int i = 0; i <= N; ++i)
	{
		lambda[i] = (double *)malloc(2 * (nb[i] + ng[i]) * sizeof(double));
		assert(lambda[i] != NULL);
	}

	return lambda;
}

double ** alloc_pi(int N, int const * nx)
{
	double ** pi = (double **)malloc(N * sizeof(double *));
	assert(pi != NULL);
	
	for (int i = 0; i < N; ++i)
	{
		pi[i] = (double *)malloc(nx[i + 1] * sizeof(double));
		assert(pi[i] != NULL);
	}

	return pi;
}

void free_array_of_arrays(double ** arr, int N)
{
	for (int i = 0; i < N; ++i)
		free(arr[i]);
	free(arr);
}

void free_lambda(double ** lambda, int N)
{
	free_array_of_arrays(lambda, N + 1);
}

void free_pi(double ** pi, int N)
{
	free_array_of_arrays(pi, N);
}

void print_vector(double const * x, int N)
{
    for (int i = 0; i < N; ++i)
        printf("%e\t", x[i]);
}

/*
Test case for https://github.com/giaf/hpmpc/issues/10
This is a badly conditioned problem (condition number 6.5488e+10)
*/
TEST(test_qp, qp10_return_code_ok)
{
	ProblemStruct * qp = (ProblemStruct *)malloc(sizeof(ProblemStruct));
	memset(qp, sizeof(ProblemStruct), 0xff);

	init_problem(qp);

	int num_iter = -1;
	double inf_norm_res[4];
	int const work_size = hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(qp->N, qp->nx, qp->nu, qp->nb, qp->ng);

	void * const work = malloc(work_size);
	double * const stat = (double *)malloc(qp->k_max * 5 * sizeof(double));
	double ** lam = alloc_lambda(qp->N, qp->nb, qp->ng);
	double ** t = alloc_lambda(qp->N, qp->nb, qp->ng);
	double ** pi = alloc_pi(qp->N, qp->nx);

	int const ret = c_order_d_ip_ocp_hard_tv(&num_iter,
					qp->k_max, qp->mu0, qp->mu_tol,
					qp->N, qp->nx, qp->nu, qp->nb, qp->ng,
					qp->warm_start,
					qp->A, qp->B, qp->b,
					qp->Q, qp->S, qp->R, qp->q, qp->r,
					qp->lb, qp->ub,
					qp->C, qp->D, qp->lg, qp->ug,
					qp->x, qp->u, pi, lam, t,
					inf_norm_res,
					work,
					stat);

	printf("ret = %d, num_iter = %d\n", ret, num_iter);
	printf("**** Solution ****\n");

	for (int i = 0; i <= qp->N; ++i)
	{
		printf("%d:\t", i);

		if (i < qp->N)
		{            
		    print_vector(qp->u[i], qp->nu[i]);
		    printf("|\t");
		}

		print_vector(qp->x[i], qp->nx[i]);
		printf("\n");
	}

	free_pi(pi, qp->N);
	free_lambda(t, qp->N);
	free_lambda(lam, qp->N);
	free(stat);
	free(work);
	free(qp);

	EXPECT_EQ(ret, 0);
}
