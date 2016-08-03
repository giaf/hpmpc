/*
 * hpmpc_test.cpp
 *
 *  Created on: Jun 17, 2016
 *      Author: kotlyar
 */

#include "../include/c_interface.h" // HPMPC C interface

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <iostream>
#include <vector>

unsigned const NX = 2;
unsigned const NU = 1;
unsigned const NC = 0;
unsigned const NCT = 0;
unsigned const NT = 2;

typedef Eigen::Matrix<double, NX, 1> StateVector;
typedef Eigen::Matrix<double, NU, 1> InputVector;

TEST(hpmpc_test, problem_1_solution_is_correct)
{
	int const nx[NT + 1] = {0, NX, NX};
	int const nu[NT + 1] = {NU, NU, 0};
	int const nb[NT + 1] = {nx[0] + nu[0], nx[1] + nu[1], nx[2] + nu[2]};
	int const ng[NT + 1] = {0, 0, 0};

	double const A0[NX * NX] = {1., 1., 0., 1.};
	double const * const A[NT] = {A0, A0};

	double const B0[NX * NU] = {0.5, 1.};
	double const * const B[NT] = {B0, B0};

	double const x0[NX] = {1., 0.};
	double const b0[NX] = {A0[NX * 0 + 0] * x0[0] + A0[NX * 0 + 1] * x0[1], A0[NX * 1 + 0] * x0[0] + A0[NX * 1 + 1] * x0[1]};
	double const b1[NX] = {0., 0.};
	double const * const b[NT] = {b0, b1};

	double const Q0[NX * NX] = {66., 78., 78., 93.};
	double const QT[NX * NX] = {10., 14., 14., 20.};
	double const * const Q[NT + 1] = {nullptr, Q0, QT};

	double const S0[NU * NX] = {90., 108};
	double const * const S[NT] = {nullptr, S0};

	double const R0[NU * NU] = {126.};
	double const * const R[NT] = {R0, R0};

	double const q0[NX] = {0., 0.};
	double const * const q[NT + 1] = {nullptr, q0, q0};

	double const r0[NU] = {S0[0] * x0[0] + S0[1] * x0[1]};
	double const r1[NU] = {0.};
	double const * const r[NT] = {r0, r1};

	double const lb0[NU     ] = {-1.};
	double const lb1[NU + NX] = {-1., -1., -1.};
	double const lbT[NX] = {-1., -1.};
	double const * const lb[NT + 1] = {lb0, lb1, lbT};

	double const ub0[NU     ] = { 1.};
	double const ub1[NU + NX] = {1., 1., 1.};
	double const ubT[NX] = {1., 1.};
	double const * const ub[NT + 1] = {ub0, ub1, ubT};

	double const * const C[NT + 1] = {nullptr, nullptr, nullptr};
	double const * const D[NT + 1] = {nullptr, nullptr, nullptr};
	double const * const lg[NT + 1] = {nullptr, nullptr, nullptr};
	double const * const ug[NT + 1] = {nullptr, nullptr, nullptr};

	StateVector x[NT + 1];	double * px[NT + 1] = {nullptr, x[1].data(), x[2].data()};
	InputVector u[NT];        double * pu[NT    ] = {u[0].data(), u[1].data()};
	double pi [NT    ][NX]; double * ppi[NT    ] = {pi[0], pi[1]};
	double lam[NT + 1][2 * (NX + NU)];	double * plam[NT + 1] = {lam[0], lam[1], lam[2]};
	double t  [NT + 1][2 * (NX + NU)];	double * pt  [NT + 1] = {t  [0], t  [1], t  [2]};
	double inf_norm_res[4];

	int const max_iter = 100;
	double stat[5 * max_iter];

	std::vector<char> workspace(hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(NT, nx, nu, nb, ng));

	int num_iter = 0;
	double const mu0 = 0;
	double const mu_tol = 1e-10;

	auto const ret = c_order_d_ip_ocp_hard_tv(&num_iter, max_iter, mu0, mu_tol, NT,
			nx, nu, nb, ng, 0, A, B, b,
			Q, S, R, q, r, lb, ub, C, D,
			lg, ug, px, pu, ppi, plam, pt, inf_norm_res,
			workspace.data(), stat);

	std::cout << "num_iter = " << num_iter << std::endl;
	for(int jj=0; jj<num_iter; jj++)
		printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\talpha = %f\tmu = %f\tmu = %e\n", jj, stat[5*jj], stat[5*jj+1], stat[5*jj+2], stat[5*jj+2], stat[5*jj+3], stat[5*jj+4], stat[5*jj+4]);
	printf("\n");

	ASSERT_EQ(ret, 0);

	std::cout << "u[0] = " << u[0].transpose() << std::endl;
	std::cout << "x[1] = " << x[1].transpose() << "\tu[1] = " << u[1].transpose() << std::endl;
	std::cout << "x[2] = " << x[2].transpose() << std::endl;

	InputVector u0_expected;	u0_expected << -0.690877362606266;
	EXPECT_TRUE(u[0].isApprox(u0_expected, 1e-6));

	StateVector x1_expected;	x1_expected << 0.654561318696867, -0.690877362606266;
	InputVector u1_expected;	u1_expected << 0.215679569867116;
	EXPECT_TRUE(x[1].isApprox(x1_expected, 1e-6));
	EXPECT_TRUE(u[1].isApprox(u1_expected, 1e-6));

	StateVector x2_expected;	x2_expected << 0.0715237410241597, -0.475197792739149;
	EXPECT_TRUE(x[2].isApprox(x2_expected, 1e-6));
}

TEST(hpmpc_test, problem_0_solution_is_correct)
{
	typedef Eigen::Matrix<double, NX + NU, 1> StateInputVector;

	int const nx[NT + 1] = {NX, NX, NX};
	int const nu[NT + 1] = {NU, NU, 0};
	int const nb[NT + 1] = {NU + NX, NU + NX, NX};
	int const ng[NT + 1] = {0, 0, 0};

	double const A0[NX * NX] = {1., 1., 0., 1.};
	double const * const A[NT] = {A0, A0};

	double const B0[NX * NU] = {0.5, 1.};
	double const * const B[NT] = {B0, B0};

	double const b0[NX] = {1., 2.};
	double const * const b[NT] = {b0, b0};

	double const Q0[NX * NX] = {66., 78., 78., 93.};
	double const QT[NX * NX] = {10., 14., 14., 20.};
	double const * const Q[NT + 1] = {Q0, Q0, QT};

	double const S0[NU * NX] = {90., 108};
	double const * const S[NT] = {S0, S0};

	double const R0[NU * NU] = {126.};
	double const * const R[NT] = {R0, R0};

	double const q0[NX] = {0., 0.};
	double const * const q[NT + 1] = {q0, q0, q0};

	double const r0[NU] = {0.};
	double const * const r[NT] = {r0, r0};

	double const lb0[NU + NX] = {-1., -1., -1.};
	double const lbT[NX] = {-1., -1.};
	double const * const lb[NT + 1] = {lb0, lb0, lbT};

	double const ub0[NU + NX] = {1., 1., 1.};
	double const ubT[NX] = {1., 1.};
	double const * const ub[NT + 1] = {ub0, ub0, ubT};

	double const * const C[NT + 1] = {nullptr, nullptr, nullptr};
	double const * const D[NT + 1] = {nullptr, nullptr, nullptr};
	double const * const lg[NT + 1] = {nullptr, nullptr, nullptr};
	double const * const ug[NT + 1] = {nullptr, nullptr, nullptr};

	StateVector x  [NT + 1];	double * px [NT + 1] = {x [0].data(), x [1].data(), x [2].data()};
	InputVector u  [NT    ];  double * pu [NT    ] = {u [0].data(), u [1].data()};

	double pi [NT    ][NX]; double * ppi[NT    ] = {pi[0], pi[1]};
	double lam[NT + 1][2 * (NX + NU)];	double * plam[NT + 1] = {lam[0], lam[1], lam[2]};
	double t  [NT + 1][2 * (NX + NU)];	double * pt  [NT + 1] = {t  [0], t  [1], t  [2]};
	double inf_norm_res[4];

	int const max_iter = 100;
	double stat[5 * max_iter];

	std::vector<char> workspace(hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(NT, nx, nu, nb, ng));

	int num_iter = 0;
	double const mu0 = 0;
	double const mu_tol = 1e-10;

	auto const ret = c_order_d_ip_ocp_hard_tv(&num_iter, max_iter, mu0, mu_tol, NT,
			nx, nu, nb, ng, 0, A, B, b,
			Q, S, R, q, r, lb, ub, C, D,
			lg, ug, px, pu, ppi, plam, pt, inf_norm_res,
			static_cast<void *>(workspace.data()), stat);

	ASSERT_EQ(ret, 0);

	StateInputVector z0_expected;
	z0_expected << 1., -1., -1;

	StateInputVector z1_expected;
	z1_expected << 0.5, 0., -1;

	StateVector z2_expected;
	z2_expected << 1., 1;

	/*
	std::cout << "x[0] = " << x[0] << std::endl;
	std::cout << "u[0] = " << u[0] << std::endl;
	std::cout << "x[1] = " << x[1] << std::endl;
	std::cout << "u[1] = " << u[1] << std::endl;
	std::cout << "x[2] = " << x[2] << std::endl;
	*/

	EXPECT_TRUE(x[0].isApprox(z0_expected.topRows<NX>()));
	EXPECT_TRUE(u[0].isApprox(z0_expected.bottomRows<NU>()));
	EXPECT_TRUE(x[1].isApprox(z1_expected.topRows<NX>()));
	EXPECT_TRUE(u[1].isApprox(z1_expected.bottomRows<NU>()));
	EXPECT_TRUE(x[2].isApprox(z2_expected));
}
