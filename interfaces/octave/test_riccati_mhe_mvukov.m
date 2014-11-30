close all
clear all
clc

%% compile the C code

%mex HPMPC_riccati_mhe.c -I/usr/local/include -L/usr/local/lib -lhpmpc %-L. HPMPC.a
mex HPMPC_riccati_mhe.c -I/usr/include -L/usr/lib -lhpmpc %-L. HPMPC.a


graphics_toolkit('gnuplot')



%% test problem

nx = 8;			% number of states
nw = 3;				% number of inputs (controls)
ny = 3;			% number of measurements
N = 10;				% horizon length

Ts = 0.5; % sampling time

Ac = [
    zeros(nx/2), eye(nx/2);
    diag(-2*ones(nx/2,1))+diag(ones(nx/2-1,1),-1)+diag(ones(nx/2-1,1),1), zeros(nx/2) ];
Bc = [
    zeros(nx/2,nw);
    eye(nw);
    zeros(nx/2-nw, nw)];

M = expm([
    Ts*Ac, Ts*Bc;
    zeros(nw, 2*nx/2+nw)]);

% dynamica system
A = M(1:nx,1:nx);
G = M(1:nx,nx+1:end);
C = [eye(ny) zeros(ny, nx-ny)];
f = 0.0*ones(nx,1);

AA = repmat(A, 1, N);
GG = repmat(G, 1, N);
CC = repmat(C, 1, N+1);
ff = repmat(f, 1, N);

%% Optimization problem

sigma_q = 0.1;
sigma_r = 0.01;

% cost function
Q = eye(nw) * sigma_q^2; % Q and R are factors or not? NO
R = eye(ny) * sigma_r^2;
q = zeros(nw,1);
r = zeros(ny,1);

QQ = repmat(Q, 1, N);
RR = repmat(R, 1, N+1);
qq = repmat(q, 1, N);
rr = repmat(r, 1, N+1);

% initial guess for state and covariance
hx0 = zeros(nx,1);
%hL0 = 1 * eye(nx);
hL0 = zeros(nx,nx);
for ii=1:nx
	hL0(ii,ii) = 1.0;
end

%% Simulation data

% Simulation length in steps
Ns = 200;

% Smooth or not

smooth = 1;

% Number of generated states and measurements
Nsim = N + 1 + Ns;

x = zeros(nx, Nsim+1);
x(:,1) = zeros(nx, 1);
w = zeros(nw, Nsim);
y = zeros(ny, Nsim+1);
v = zeros(ny, Nsim+1);

v(:,1) = 0.1*randn(ny,1);
y(:,1) = C*x(:,1) + v(:,1);
for ii=1:Nsim
	w(:,ii) = sigma_q*randn(nw,1);
	v(:,ii+1) = sigma_r*randn(ny,1);
	x(:,ii+1) = A*x(:,ii) + G*w(:,ii) + f;
	y(:,ii+1) = C*x(:,ii+1) + v(:,ii+1);
end


figure()
for ii = 1: nx
    subplot(nx/2, 2, ii)
    plot([0:Nsim], x(ii,:))
end;
title('Generated states')
%suptitle('Generated states')
xlabel('Nsim')
% axis([0 Ns -1 1])

figure()
for ii = 1: ny
    subplot(ny, 1, ii)
    plot([0:Nsim], y(ii,:))
end;
%suptitle('Generated measurements')
title('Generated measurements')
xlabel('Nsim')

x0 = hx0;
L0 = hL0;

% estimation at all stages 0,1,...,N
xe = zeros(nx,N+1);
% cholesky factor of estimation covariance matrix at last stage
Le = zeros(nx,nx);
% process disturbance at all stages 0,1,...,N-1
we = zeros(nw,N);

% fprintf('\nRiccati factorization and solution\n\n');

% space to save at each step the estimation at the last stage
log_xe = zeros(nx, Ns);
log_le = zeros(nx, Ns);
log_trace_le = zeros(1, Ns);
log_x0 = zeros(nx, Ns);
log_l0 = zeros(nx, Ns);

tic
for ii=1:Ns
	% x0 and L0 are input-output: on output they are the value needed at the next QP call (i.e. prediction x_{1|0} and relative covariance), obtained using (Extended) Kalman Filter
	HPMPC_riccati_mhe(smooth, nx, nw, ny, N, AA, GG, CC, ff, QQ, RR, qq, rr, y(:,ii: ii+N), x0, L0, xe, Le, we);
	% same estimation at last stage
	log_xe(:, ii) = xe(:, N + 1);
    log_le(:, ii) = diag( Le );
    log_trace_le(:, ii) = trace( Le );
    log_x0(:, ii) = x0;
    log_l0(:, ii) = diag( L0 );
end
tAvg = toc / Ns;

fprintf('\nAverage execution time %f usec\n\n', tAvg * 1e6);

% MSE Calculation

mseSub = log_xe - x(:, N + 1: N + Ns);
mse = 0;
for ii = 1: Ns
    mse = mse + mseSub(:, ii)' * mseSub(:, ii);
end;
mse = mse / Ns;

fprintf('\nMSE %f \n\n', mse);

figure()
xData = 0: Ns - 1;
for ii = 1: nx
    subplot(nx/2, 2, ii);
    % Uncertainty
    yu = log_xe(ii, :) + log_le(ii, :);
    yl = log_xe(ii, :) - log_le(ii, :);
    fill([xData, fliplr(xData)], [yu, fliplr(yl)], 'g', 'EdgeColor', 'g');
    hold on;
    % Real state
    plot(xData, x(ii, N + 1: N + Ns), 'r');
    % Estimated state
    plot(xData, log_xe(ii, :), 'b');
end;
%suptitle('Generated states; red - true, blue - estimated, green - uncertainty')
title('Generated states; red - true, blue - estimated, green - uncertainty')
xlabel('Nsim')

figure()
for ii = 1: nx
    subplot(nx/2, 2, ii)
    % Diagonal elements of Le
    plot(0: Ns - 1, log_le(ii, :), 'b');
end;
%suptitle('Diagonal elements of Le')
title('Diagonal elements of Le')
xlabel('Nsim')

figure()
semilogy(log_trace_le, 'b');
%suptitle('trace( Le )')
title('trace( Le )')
xlabel('Nsim')

figure()
for ii = 1: nx
    subplot(nx/2, 2, ii);
    % Uncertainty
    yu = log_x0(ii, :) + log_l0(ii, :);
    yl = log_x0(ii, :) - log_l0(ii, :);
    fill([xData, fliplr(xData)], [yu, fliplr(yl)], 'g', 'EdgeColor', 'g');
    hold on;
    plot(0: Ns - 1, log_x0(ii, :), 'b');
end;
%suptitle('x0 elements; blue: apriori estimate, green: uncertainty')
title('x0 elements; blue: apriori estimate, green: uncertainty')
xlabel('Nsim')

figure()
for ii = 1: nx
    subplot(nx/2, 2, ii)
    plot(0: Ns - 1, log_l0(ii, :), 'b');
end;
%suptitle('Diagonal elements of L0')
title('Diagonal elements of L0')
xlabel('Nsim')
