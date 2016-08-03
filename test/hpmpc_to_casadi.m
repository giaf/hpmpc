function [nlp, bounds, x0] = hpmpc_to_casadi(qp)
% Converts an hpmpc problem structure qp to CasADi npl
    N = qp.N;
    
    % Check matrices
    for i = 1 : N + 1
        assert(is_spd(qp.Q{i}));
    end
    
    for i = 1 : N
        assert(is_spd(qp.R{i}));
    end
    
    % Create optimization variables
    x = cell(1, N + 1);
    for i = 1 : N + 1
        x{i} = casadi.MX.sym(sprintf('x%d', i - 1), qp.nx(i));
    end
    
    u = cell(1, N);
    for i = 1 : N
        u{i} = casadi.MX.sym(sprintf('u%d', i - 1), qp.nu(i));
    end
    
    % Create objective
    J = 0;
    for i = 1 : N
        J = J + [u{i}; x{i}; 1].' * [qp.R{i}  , qp.S{i}  , qp.r{i}; 
                                     qp.S{i}.', qp.Q{i}  , qp.q{i};
                                     qp.r{i}.', qp.q{i}.',       0] * [u{i}; x{i}; 1];
    end
    
    J = J + [x{N + 1}; 1].' * [qp.Q{N + 1}  , qp.q{N + 1};
                               qp.q{N + 1}.',           0] * [x{N + 1}; 1];
                           
    % Create constraints
    c = cell(1, N);
    for i = 1 : N
        c{i} = qp.A{i} * x{i} + qp.B{i} * u{i} + qp.b{i};
    end
    
    g = cell(1, N + 1);
    for i = 1 : N
        g{i} = qp.C{i} * x{i} + qp.D{i} * u{i};
    end
    g{N + 1} = qp.C{N + 1} * x{N + 1};
    
    % Prepare CasADi nlp
    X = [reshape([u; x(1 : end-1)], 2 * N, 1); x(end)];
    nlp.x = vertcat(X{:});
    nlp.f = J;
    nlp.g = vertcat(c{:}, g{:});
    bounds.lbx = vertcat(qp.lb{:});
    bounds.ubx = vertcat(qp.ub{:});
    bounds.lbg = [zeros(sum(qp.nx(2 : end)), 1); vertcat(qp.lg{:})];
    bounds.ubg = [zeros(sum(qp.nx(2 : end)), 1); vertcat(qp.ug{:})];
    x0 = [reshape([qp.u; qp.x(1 : end-1)], 2 * N, 1); qp.x(end)];
    x0 = vertcat(x0{:});
end

function b = is_spd(m)
    % Return true if m is symmetric positive definite
    [~, p] = chol(m);
    b = isequal(m, m.') && p == 0;
end