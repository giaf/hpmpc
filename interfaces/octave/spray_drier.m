clear

% compile the C code
mex HPMPC_ip_box.c -lhpmpc %-L. HPMPC.a

% import cool graphic toolkit if in octave
if is_octave()
	graphics_toolkit('gnuplot')
end

% import problem data
spray_drier_data

%%% port problem into format required by the solver (no penalty on du)

% problem size
nx = nxo + nuo;
nu = nuo;
N = 180;
ny = 3;

% dynamic system
A = zeros(nx,nx);
A(1:nxo,1:nxo) = Ao;

B = zeros(nx,nu);
B(1:nxo,:) = Bo;
B(nxo+1:end,:) = eye(nu);

b = zeros(nx,1);
b = Eo*do_sp;

C = zeros(ny,nu);
C = Co(1:3,:);

% cost function
R = zeros(nu,nu);
R = So;

Qy = zeros(ny,ny);
Qy = Qo(1:ny,1:ny);

Q = zeros(nx,nx);
Q(1:nxo,1:nxo) = C'*Qy*C;
Q(nxo+1:end,nxo+1:end) = So;

S = zeros(nu,nx);
S(:,nxo+1:end) = -So;

y_sp = yo_sp(1:3);
qo = - Qy*y_sp;

r = zeros(nu,1);

q = zeros(nx,1);
q(1:nxo) = C'*qo;

% constraints
lb = -1e6*ones(nu+nx,1);
lb(1:nu) = uo_con(:,1);
ub =  1e6*ones(nu+nx,1);
%ub(1:nu) = uo_con(:,2);

% pack matrices
AA = repmat(A, 1, N);
BB = repmat(B, 1, N);
bb = repmat(b, 1, N);
QQ = repmat(Q, 1, N);
Qf = Q;
SS = repmat(S, 1, N);
RR = repmat(R, 1, N);
qq = repmat(q, 1, N);
qf = q;
rr = repmat(r, 1, N);
llb = repmat(lb, 1, N);
uub = repmat(ub, 1, N);



% initial guess for states and inputs
x = zeros(nx, N+1); 
x_sp = C(:,1:4)\y_sp;
x(1:4,1) = x_sp; % initial condition
x(nxo+1:end,1) = uo_sp;
u = ones(nu, N);
%pi = zeros(nx, N+1);

kk = -1;		% actual number of performed iterations
k_max_ip = 40;		% maximim number of iterations
k_max_admm = 1000;		% maximim number of iterations
tol = 1e-4;		% tolerance in the duality measure
infos = zeros(5, max(k_max_ip, k_max_admm));

% ADMM parameters
rho = 0.01;        % regularization parameter
alpha = 1.0;    % relaxation parameter

tic
%HPMPC_ip_box(k_max_ip, tol, nx, nu, N, AA, BB, bb, QQ, Qf, RR, SS, qq, qf, rr, llb, uub, x, u, kk, infos);
HPMPC_admm_box(k_max_admm, tol, rho, alpha, nx, nu, N, AA, BB, bb, QQ, Qf, RR, SS, qq, qf, rr, llb, uub, x, u, kk, infos);
toc

infos(:,1:kk)'
kk

u
%x


