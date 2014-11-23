% compile the C code

mex HPMPC_riccati_mhe.c -lhpmpc %-L. HPMPC.a



% test problem

nx = 12;			% number of states
nw = 5;				% number of inputs (controls)
ny  = 3;			% number of measurements
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
Q = eye(nw);
R = 1*eye(ny);
q = zeros(nw,1);
r = zeros(ny,1);

QQ = repmat(Q, 1, N);
RR = repmat(R, 1, N+1);
qq = repmat(q, 1, N);
rr = repmat(r, 1, N+1);


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

figure()
plot([0:Ns], y(:,:))
title('measurements')
xlabel('N')

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


x0 = zeros(nx,1);
L0 = zeros(nx,nx);
for ii=1:nx
	L0(ii,ii) = 1.0;
end

xe = zeros(nx,1);
Le = zeros(nx,nx);

fprintf("\nRiccati factorization and solution\n\n");

% estimation horizon
Ne = 50;
hxe = zeros(nx,Ne);

tic
for ii=1:Ne
	HPMPC_riccati_mhe(nx, nw, ny, N, AA, GG, CC, ff, QQ, RR, qq, rr, y(:,ii:ii+N), x0, L0, xe, Le);
	hxe(:,ii) = xe;
%	x0
%	L0
%	xe
%	Le
end
toc

figure()
plot([N+1:N+Ne], hxe(1:nx/2,1:Ne))
title('states 2')
xlabel('N')
axis([0 100])

x0
L0
xe
Le
