#include "../include/c_interface.h"

#include <gtest/gtest.h>

#include <assert.h>
#include <stdio.h>
#include <memory.h>

#include <iostream>

/*********************

  Utility functions

*********************/

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

template <typename QP>
void print_solution(QP const& qp, int ret, int num_iter)
{
	printf("ret = %d, num_iter = %d\n", ret, num_iter);
	printf("**** Solution ****\n");

	for (int i = 0; i <= qp.N; ++i)
	{
		printf("%d:\t", i);

		if (i < qp.N)
		{            
		    print_vector(qp.u[i], qp.nu[i]);
		    printf("|\t");
		}

		print_vector(qp.x[i], qp.nx[i]);
		printf("\n");
	}
}

/**
 * \brief Data parameter for QPTest
 */
struct QpData
{
	int k_max;
	double mu0;
	double mu_tol;
	int N;
	int const *nx;
	int const *nu;
	int const *nb;
	int const *ng;
	int warm_start;
	double const * const *A;
	double const * const *B;
	double const * const *b;
	double const * const *Q;
	double const * const *S;
	double const * const *R;
	double const * const *q;
	double const * const *r;
	double const * const *lb;
	double const * const *ub;
	double const * const *C;
	double const * const *D;
	double const * const *lg;
	double const * const *ug;
	double * const *x;
	double * const *u;

	/**
	 * \brief Construct from any structure with the same fields.
	 */
	template <typename ProblemStruct>
	QpData(ProblemStruct const& qp)
	:	k_max(qp.k_max)
	,   mu0(qp.mu0)
	,   mu_tol(qp.mu_tol)
	,   N(qp.N)
	,   nx(qp.nx)
	,   nu(qp.nu)
	,   nb(qp.nb)
	,   ng(qp.ng)
	,   warm_start(qp.warm_start)
	,   A(qp.A)
	,   B(qp.B)
	,   b(qp.b)
	,   Q(qp.Q)
	,   S(qp.S)
	,   R(qp.R)
	,   q(qp.q)
	,   r(qp.r)
	,   lb(qp.lb)
	,   ub(qp.ub)
	,   C(qp.C)
	,   D(qp.D)
	,   lg(qp.lg)
	,   ug(qp.ug)
	,   x(qp.x)
	,   u(qp.u)
	{
	}
};

/**
 * \brief Wraps T::ProblemStruct and calls T::init_problem() to initialize it.
 */
template <typename T>
struct ProblemStructWrapper : T::ProblemStruct
{
	ProblemStructWrapper()
	{
		T().init_problem(this);
	}
};

template <typename QP_>
class QPTest : public ::testing::Test
{
protected:
	typedef QP_ QP;

/*
	QPTest()
	:	qp_(new QP::ProblemStruct)
	{
		memset(qp, sizeof(ProblemStruct), 0xff);
	}

	std::unique_ptr<QP::ProblemStruct> qp_;
*/
};

/*
Test case for https://github.com/giaf/hpmpc/issues/10
This is a badly conditioned problem (condition number 6.5488e+10)
*/
struct qp0010 {
	#include "data/test_qp/qp0010.c"
};

struct qp0012 {
	#include "data/test_qp/qp0012.c"
};

/*
Relaxed version of qp0012: terminal equality constraints
replaced with inequalities.
*/
struct qp0012_relaxed {
	#include "data/test_qp/qp0012_relaxed.c"
};

typedef ::testing::Types<
		qp0010,
		qp0012,
		qp0012_relaxed
	> QPTypes;

TYPED_TEST_CASE(QPTest, QPTypes);

TYPED_TEST(QPTest, return_code_ok)
{
	typedef ProblemStructWrapper<typename TestFixture::QP> Wrapper;

	std::unique_ptr<Wrapper const> const wrapper(new Wrapper);
	QpData qp(*wrapper);

	int num_iter = -1;
	double inf_norm_res[4];
	int const work_size = hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(qp.N, qp.nx, qp.nu, qp.nb, qp.ng);

	void * const work = malloc(work_size);
	double * const stat = (double *)malloc(qp.k_max * 5 * sizeof(double));
	double ** lam = alloc_lambda(qp.N, qp.nb, qp.ng);
	double ** t = alloc_lambda(qp.N, qp.nb, qp.ng);
	double ** pi = alloc_pi(qp.N, qp.nx);

	int const ret = c_order_d_ip_ocp_hard_tv(&num_iter,
					qp.k_max, qp.mu0, qp.mu_tol,
					qp.N, qp.nx, qp.nu, qp.nb, qp.ng,
					qp.warm_start,
					qp.A, qp.B, qp.b,
					qp.Q, qp.S, qp.R, qp.q, qp.r,
					qp.lb, qp.ub,
					qp.C, qp.D, qp.lg, qp.ug,
					qp.x, qp.u, pi, lam, t,
					inf_norm_res,
					work,
					stat);

	std::cout << "ret = " << ret << "\tnum_iter = " << num_iter << std::endl;
	free_pi(pi, qp.N);
	free_lambda(t, qp.N);
	free_lambda(lam, qp.N);
	free(stat);
	free(work);

	EXPECT_EQ(ret, 0);
}

