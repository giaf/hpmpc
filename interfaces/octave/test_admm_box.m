% compile the C code

mex HPMPC_admm_box.c /opt/hpmpc/lib/libhpmpc.a 



% test problem

nx = 12;			% number of states
nu = 5;				% number of inputs (controls)
N = 30;				% horizon length
nb = nu+nx;		% (even) number of box constraints

Ts = 0.5; % sampling time

Ac = [zeros(nx/2), eye(nx/2); diag(-2*ones(nx/2,1))+diag(ones(nx/2-1,1),-1)+diag(ones(nx/2-1,1),1), zeros(nx/2) ];
Bc = [zeros(nx/2,nu); eye(nu); zeros(nx/2-nu, nu)];

M = expm([Ts*Ac, Ts*Bc; zeros(nu, 2*nx/2+nu)]);

% dynamica system
A = M(1:nx,1:nx);
B = M(1:nx,nx+1:end);
b = 0.0*ones(nx,1);
x0 = zeros(nx, 1);
x0(1) = 3.5;
x0(2) = 3.5;
%if nx==4
%	x0 = [5 10 15 20]';
%end
AA = repmat(A, 1, N);
BB = repmat(B, 1, N);
bb = repmat(b, 1, N);

% cost function
Q = zeros(nx,nx);
for ii=1:nx
	Q(ii,ii) = 1.0;
end
Qf = Q;
R = 2*eye(nu);
S = zeros(nx, nu);
%q = zeros(nx,1);
q = zeros(nx,1);
qf = q;
%r = zeros(nu,1);
r = zeros(nu,1);

QQ = repmat(Q, 1, N);
SS = repmat(S, 1, N);
RR = repmat(R, 1, N);
qq = repmat(q, 1, N);
rr = repmat(r, 1, N);

% box constraints
lb_u = -1e2*ones(nu,1);
ub_u =  1e2*ones(nu,1);
%db(1:2*nu) = -1.5;
for ii=1:nu
	lb_u(ii) = -0.5; % lower bound
	ub_u(ii) =  0.5; % - upper bound
end
lb_x = -9.0*ones(nx,1);
ub_x =  9.0*ones(nx,1);
%db(2*nu+1:end) = -4;
llb = [repmat(lb_u, 1, N)(:) ; repmat(lb_x, 1, N)(:)];
uub = [repmat(ub_u, 1, N)(:) ; repmat(ub_x, 1, N)(:)];

% initial guess for states and inputs
x = zeros(nx, N+1); x(:,1) = x0; % initial condition
u = -1*ones(nu, N);
%pi = zeros(nx, N+1);

kk = -1;		% actual number of performed iterations
k_max = 1000;		% maximim number of iterations
tol = 1e-4;		% tolerance in primality and duality measures
infos = zeros(5, k_max);
rho = 2;        % regularization parameter
alpha = 1.5;    % relaxation parameter

tic
HPMPC_admm_box(k_max, tol, rho, alpha, nx, nu, N, AA, BB, bb, QQ, Qf, RR, SS, qq, qf, rr, llb, uub, x, u, kk, infos);
toc

kk
%infos(:,1:kk)'

%u
%x

graphics_toolkit('gnuplot')

figure()
plot([0:N], x(:,:))
title('states')
xlabel('N')

figure()
plot([1:N], u(:,:))
title('controls')
xlabel('N')

