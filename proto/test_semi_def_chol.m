% test problem for semi-definite cholesky factorization

B = [
1 1 1 1 1
1 1 1 1 1
1 0 1 1 1
1 0 1 1 1
1 1 0 1 1
]';

D = [
0 0 0 0 0
0 0 0 0 0
0 0 1 0 1
0 0 0 0 0
0 0 1 0 1
];

A = B*B' + D

eig(A)

C = semi_def_chol(A)

C*C'

err = A - C*C'

L = chol(A,'lower')

A - L*L'
