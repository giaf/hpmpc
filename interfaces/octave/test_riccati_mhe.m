% compile the C code

mex HPMPC_riccati_mhe.c /opt/hpmpc/lib/libhpmpc.a 


% test problem

nx = 8;			% number of states
nw = 3;				% number of inputs (controls)
ny  = 1;			% number of measurements
N = 10;				% horizon length

Ts = 0.5; % sampling time

Ac = [zeros(nx/2), eye(nx/2); diag(-2*ones(nx/2,1))+diag(ones(nx/2-1,1),-1)+diag(ones(nx/2-1,1),1), zeros(nx/2) ];
Bc = [zeros(nx/2,nw); eye(nw); zeros(nx/2-nw, nw)];

M = expm([Ts*Ac, Ts*Bc; zeros(nw, 2*nx/2+nw)]);

% dynamica system
A = M(1:nx,1:nx);
G = M(1:nx,nx+1:end);
C = [eye(ny) zeros(ny, nx-ny)];
f = 0.0*ones(nx,1);

AA = repmat(A, 1, N);
GG = repmat(G, 1, N);
CC = repmat(C, 1, N+1);
ff = repmat(f, 1, N);

% cost function
Q = 1*eye(nw);
R = 1*eye(ny);
q = zeros(nw,1);
r = zeros(ny,1);

QQ = repmat(Q, 1, N);
RR = repmat(R, 1, N+1);
qq = repmat(q, 1, N);
rr = repmat(r, 1, N+1);

% initial guess for state and covariance
hx0 = zeros(nx,1);
hx0(1) = 0.0;
hx0(2) = 0.0;
hL0 = 1*eye(nx);


% simulation horizon
Ns = 100;

x = zeros(nx, Ns+1);
x(:,1) = zeros(nx, 1);
w = zeros(nw, Ns);
y = zeros(ny, Ns+1);
v = zeros(ny, Ns+1);

v(:,1) = 0.1*randn(ny,1);
y(:,1) = C*x(:,1) + v(:,1);
for ii=1:Ns
	w(:,ii) = 0.1*randn(nw,1);
	v(:,ii+1) = 0.1*randn(ny,1);
	x(:,ii+1) = A*x(:,ii) + G*w(:,ii) + f;
	y(:,ii+1) = C*x(:,ii+1) + v(:,ii+1);
end




graphics_toolkit('gnuplot')

figure()
plot([0:Ns], x(1:nx/2,:))
title('states')
xlabel('N')
axis([0 Ns -1 1])

figure()
plot([0:Ns], y(:,:))
title('measurements')
xlabel('N')
axis([0 Ns -1 1])

%figure()
%plot([1:N], u(:,:))
%title('controls')
%xlabel('N')



% print measurements to file
fid = fopen('mhe_measurements.dat', 'w');
fprintf(fid, '%d %d %d %d\n', nx, nw, ny, Ns+1);
yy = y(:);
for ii=1:size(yy)
	fprintf(fid, '%e\n', yy(ii));
end
fclose(fid);



%return;


% prediction at first stage
x0 = zeros(nx,1);
for ii=1:nx
	x0(ii) = hx0(ii);
end
% cholesky factor of prediction covariance matrix at first stage
L0 = zeros(nx,nx);
for ii=1:nx*nx
	L0(ii) = hL0(ii);
end

% estimation at all stages 0,1,...,N
xe = zeros(nx,N+1);
% cholesky factor of estimation covariance matrix at last stage
Le = zeros(nx,nx);
% process disturbance at all stages 0,1,...,N-1
we = zeros(nw,N);

fprintf("\nRiccati factorization and solution\n\n");

% number of steps in estimation test
Ne = 50;
% space to save at each step the estimation at the last stage
hxe = zeros(nx,Ne);

tic
for ii=1:Ne
	% x0 and L0 are input-output: on output they are the value needed at the next QP call (i.e. prediction x_{1|0} and relative covariance), obtained using (Extended) Kalman Filter
	HPMPC_riccati_mhe(1, nx, nw, ny, N, AA, GG, CC, ff, QQ, RR, qq, rr, y(:,ii:ii+N), x0, L0, xe, Le, we);
	% same estimation at last stage
	hxe(:,ii) = xe(:,N+1);
%	x0
%	L0
%	xe
%	Le
end
toc/Ne

figure()
plot([N+1:N+Ne], hxe(1:nx/2,1:Ne))
title('states 2')
xlabel('N')
axis([0 Ns -1 1])

%x0
%L0
%xe
%Le



% smoothed version, one run

AA = repmat(A, 1, Ns);
GG = repmat(G, 1, Ns);
CC = repmat(C, 1, Ns+1);
ff = repmat(f, 1, Ns);

QQ = repmat(Q, 1, Ns);
RR = repmat(R, 1, Ns+1);
qq = repmat(q, 1, Ns);
rr = repmat(r, 1, Ns+1);

xxe = zeros(nx,Ns+1);
ww = zeros(nw,Ns);

% prediction at first stage
for ii=1:nx
	x0(ii) = hx0(ii);
end
% cholesky factor of prediction covariance matrix at first stage
for ii=1:nx*nx
	L0(ii) = hL0(ii);
end

HPMPC_riccati_mhe(0, nx, nw, ny, Ns, AA, GG, CC, ff, QQ, RR, qq, rr, y, x0, L0, xxe, Le, ww);

figure()
plot([0:Ns], xxe(1:nx/2,:))
title('estimation')
xlabel('N')
axis([0 Ns -1 1])

figure()
plot([0:Ns], xxe(1:nx/2,:)-x(1:nx/2,:))
title('estimation error')
xlabel('N')
axis([0 Ns -1 1])

% prediction at first stage
for ii=1:nx
	x0(ii) = hx0(ii);
end
% cholesky factor of prediction covariance matrix at first stage
for ii=1:nx*nx
	L0(ii) = hL0(ii);
end

HPMPC_riccati_mhe(1, nx, nw, ny, Ns, AA, GG, CC, ff, QQ, RR, qq, rr, y, x0, L0, xxe, Le, ww);

figure()
plot([0:Ns], xxe(1:nx/2,:))
title('smooth estimation')
xlabel('N')
axis([0 Ns -1 1])

figure()
plot([0:Ns], xxe(1:nx/2,:)-x(1:nx/2,:))
title('smooth estimation error')
xlabel('N')
axis([0 Ns -1 1])


