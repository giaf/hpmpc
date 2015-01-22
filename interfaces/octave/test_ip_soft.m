% compile the C code

mex HPMPC_ip_soft.c -lhpmpc %-L. HPMPC.a



% test problem

nx = 12;			% number of states
nu = 5;				% number of inputs (controls)
N = 50;				% horizon length
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
if nx==4
	x0 = [5 10 15 20]';
end
AA = repmat(A, 1, N);
BB = repmat(B, 1, N);
%AA = repmat(A', 1, N);
%BB = repmat(B', 1, N);
bb = repmat(b, 1, N);

% cost function
Q = 0*eye(nx);
Qf = Q;
R = 2*eye(nu);
S = zeros(nx, nu);
%q = zeros(nx,1);
q = 0*Q*[ones(nx/2,1); zeros(nx/2,1)];
qf = q;
%r = zeros(nu,1);
r = 0*R*ones(nu,1);
QQ = repmat(Q, 1, N);
SS = repmat(S, 1, N);
RR = repmat(R, 1, N);
qq = repmat(q, 1, N);
rr = repmat(r, 1, N);

% cost function of soft constrained slack variables
lZ = 0.0*ones(nx, 1);
uZ = 0.0*ones(nx, 1);
lz = 100*ones(nx, 1);
uz = 100*ones(nx, 1);
llZ = repmat(lZ, 1, N);
uuZ = repmat(uZ, 1, N);
llz = repmat(lz, 1, N);
uuz = repmat(uz, 1, N);

% box constraints
lb_u = -1e2*ones(nu,1);
ub_u =  1e2*ones(nu,1);
%db(1:2*nu) = -1.5;
for ii=1:nu
	lb_u(ii) = -0.5; % lower bound
	ub_u(ii) =  0.5; % - upper bound
end
lb_x = -1*ones(nx,1);
ub_x =  1*ones(nx,1);
#db(2*nu+1:end) = -4;
llb = [repmat(lb_u, 1, N)(:) ; repmat(lb_x, 1, N)(:)];
uub = [repmat(ub_u, 1, N)(:) ; repmat(ub_x, 1, N)(:)];

% initial guess for states and inputs
x = zeros(nx, N+1); x(:,1) = x0; % initial condition
u = -1*ones(nu, N);
%pi = zeros(nx, N+1);

kk = -1;		% actual number of performed iterations
k_max = 50;		% maximim number of iterations
tol = 1e-6;		% tolerance in the duality measure
infos = zeros(5, k_max);

tic
%HPMPC_ip_box(k_max, tol, nx, nu, N, AA, BB, bb, QQ, Qf, RR, SS, qq, qf, rr, llb, uub, x, u, kk, infos);
HPMPC_ip_soft(k_max, tol, nx, nu, N, AA, BB, bb, QQ, Qf, RR, SS, qq, qf, rr, llZ, uuZ, llz, uuz, llb, uub, x, u, kk, infos);
toc

kk
infos(:,1:kk)'

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

