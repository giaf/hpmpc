clear;
run data/test_qp/qp0010a.m
[nlp, b, x0] = hpmpc_to_casadi(qp);
options.ipopt.linear_solver = 'ma86';
solver = casadi.nlpsol('NLPSolver', 'ipopt', nlp, options);
sol = solver('lbx', b.lbx, 'ubx', b.ubx, 'lbg', b.lbg, 'ubg', b.ubg)
x_casadi = full(sol.x);
% disp('CasADi solution:');
% disp(x.');