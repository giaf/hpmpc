function KKT = kkt_matrix(problem)
% Returns the KKT matrix of quadprog problem 'problem'
    KKT = [problem.H, problem.Aeq.'; problem.Aeq, zeros(size(problem.Aeq, 1))];
end

