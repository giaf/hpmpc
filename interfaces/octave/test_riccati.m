% compile the C code

mex HPMPC_riccati.c -lhpmpc #-L. HPMPC.a



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
if nx==4
	x0 = [5 10 15 20]';
end
AA = repmat(A, 1, N);
BB = repmat(B, 1, N);
%AA = repmat(A', 1, N);
%BB = repmat(B', 1, N);
bb = repmat(b, 1, N);

% cost function
Q = eye(nx);
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


% initial guess for states and inputs
x = zeros(nx, N+1); %x(:,1) = x0; % initial condition
x(:,1) = x0;
u = zeros(nu, N);
pI = zeros(nx, N+1);


fprintf("\nRiccati factorization and solution\n\n");
tic
HPMPC_riccati(nx, nu, N, AA, BB, bb, QQ, Qf, RR, SS, qq, qf, rr, x, u, pI);
toc

u
x

graphics_toolkit('gnuplot')

figure()
plot([0:N], x(:,:))
title('states')
xlabel('N')

figure()
plot([1:N], u(:,:))
title('controls')
xlabel('N')

