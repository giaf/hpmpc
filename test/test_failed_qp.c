#include <c_interface.h>

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

#include "failed_qp.c"

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

int main(int argc, char ** argv)
{
	ProblemStruct * qp = malloc(sizeof(ProblemStruct));
	memset(qp, sizeof(ProblemStruct), 0xff);

	init_problem(qp);

	int num_iter = -1;
	double inf_norm_res[4];
	int const work_size = hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(qp->N, qp->nx, qp->nu, qp->nb, qp->ng);

	void * const work = malloc(work_size);
	double * const stat = malloc(qp->k_max * 5 * sizeof(double));
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

	free_pi(pi, qp->N);
	free_lambda(t, qp->N);
	free_lambda(lam, qp->N);
	free(stat);
	free(work);
	free(qp);

	return ret == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
