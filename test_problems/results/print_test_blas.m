%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                                 %
% This file is part of HPMPC.                                                                     %
%                                                                                                 %
% HPMPC -- Library for High-Performance implementation of solvers for MPC.                        %
% Copyright (C) 2014 by Technical University of Denmark. All rights reserved.                     %
%                                                                                                 %
% HPMPC is free software; you can redistribute it and/or                                          %
% modify it under the terms of the GNU Lesser General Public                                      %
% License as published by the Free Software Foundation; either                                    %
% version 2.1 of the License, or (at your option) any later version.                              %
%                                                                                                 %
% HPMPC is distributed in the hope that it will be useful,                                        %
% but WITHOUT ANY WARRANTY; without even the implied warranty of                                  %
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            %
% See the GNU Lesser General Public License for more details.                                     %
%                                                                                                 %
% You should have received a copy of the GNU Lesser General Public                                %
% License along with HPMPC; if not, write to the Free Software                                    %
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  %
%                                                                                                 %
% Author: Gianluca Frison, giaf (at) dtu.dk                                                       %
%                                                                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% prints the results of the performance test on blas

test_blas

graphics_toolkit('gnuplot')

f1 = figure()
plot(B(:,1), B(:,2), 'b-x', 'LineWidth', 2)
hold on
plot(B(:,1), B(:,4), 'r-o', 'LineWidth', 2)
plot(B(:,1), B(:,6), 'g-*', 'LineWidth', 2)
plot(B(:,1), B(:,8), 'r-^', 'LineWidth', 2)
plot(B(:,1), B(:,10), 'm-^', 'LineWidth', 2)
plot(B(:,1), B(:,12), 'c-d', 'LineWidth', 2)
plot(B(:,1), B(:,14), 'k-s', 'LineWidth', 2)
plot(B(:,1), B(:,16), 'c-s', 'LineWidth', 2)
plot(B(:,1), B(:,18), 'k-d', 'LineWidth', 2)
plot(B(:,1), B(:,20), 'g-^', 'LineWidth', 2)
plot(B(:,1), B(:,22), 'm-*', 'LineWidth', 2)
plot(B(:,1), B(:,24), 'b-o', 'LineWidth', 2)
plot(B(:,1), B(:,26), 'k-x', 'LineWidth', 2)
plot(B(:,1), B(:,28), 'k-o', 'LineWidth', 2)
plot(B(:,1), B(:,30), 'k-*', 'LineWidth', 2)
plot(B(:,1), B(:,32), 'k-^', 'LineWidth', 2)
hold off

title(['test HPMPC BLAS'])
xlabel('matrix size n')
ylabel('Gflops')
axis([0 B(end,1) 0 A(1)*A(2)])
legend('gemm', 'syrk\_potrf', 'trmm', 'trtr', 'gemv\_n', 'gemv\_t','trmv\_n', 'trmv\_t','trsv\_n', 'trsv\_t','symv', 'mvmv', 'Location', 'SouthEast')

W = 6; H = 5;
set(f1,'PaperUnits','inches')
set(f1,'PaperOrientation','portrait');
set(f1,'PaperSize',[H,W])
set(f1,'PaperPosition',[0,0,W,H])
FN = findall(f1,'-property','FontName');
set(FN,'FontName','/usr/share/fonts/truetype/ttf-dejavu/DejaVuSerifCondensed.ttf');
FS = findall(f1,'-property','FontSize');
set(FS,'FontSize',10);
file_name = ['test_blas_', C, '.eps'];
print(f1, file_name, '-depsc') 

