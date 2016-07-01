function problem = hpmpc_to_quadprog(qp)
    N = qp.N;
    
%     [Nx, Nz] = size(qp.C{1});
%     Nd = size(qp.D{1}, 1);
%     NdT = size(qp.D{end}, 1);
    
    H = cell(1, N + 1);
    g = cell(1, N + 1);
    BA = cell(1, N);
    DC = cell(1, N + 1);
    for i = 1 : N
        H{i} = [qp.R{i}  , qp.S{i}; 
                qp.S{i}.', qp.Q{i}];
        g{i} = [qp.r{i}; qp.q{i}];
        BA{i} = [qp.B{i}, qp.A{i}];
        DC{i} = [qp.D{i}, qp.C{i}];
    end
    H{N + 1} = qp.Q{N + 1};
    g{N + 1} = qp.q{N + 1};
    DC{N + 1} = qp.C{N + 1};

    problem.H = blkdiag(H{:});
    problem.f = vertcat(g{:});
    
    % x_{k+1} = A_k * x_k + B_k * u_k + b_k
    % => -A_k * x_k - B_k * u_k + x_{k+1} = b_k
    problem.Aeq = -blkdiag(BA{:});
    
    nx = qp.nx;
    nu = [qp.nu; 0];
    for i = 1 : N
        m = sum(nx(2 : i));
        n = sum(nu(1 : i)) + sum(nx(1 : i)) + nu(i + 1);
        problem.Aeq(m + (1 : nx(i + 1)), n + (1 : nx(i + 1))) = eye(nx(i + 1));
    end
    
    problem.beq = vertcat(qp.b{:});
    
    problem.Aineq = [blkdiag(DC{:}); -blkdiag(DC{:})];
    problem.bineq = [vertcat(qp.ug{:}); -vertcat(qp.lg{:})];

    problem.lb = vertcat(qp.lb{:});
    problem.ub = vertcat(qp.ub{:});
    
    problem.solver = 'quadprog';
    problem.options = optimoptions('quadprog');
end

