% prints the results of the performance test on blas

test_blas

f1 = figure()
plot(B(:,1), B(:,2), 'b-x', 'LineWidth', 2)
hold on
plot(B(:,1), B(:,4), 'r-o', 'LineWidth', 2)
plot(B(:,1), B(:,6), 'g-*', 'LineWidth', 2)
plot(B(:,1), B(:,8), 'm-^', 'LineWidth', 2)
plot(B(:,1), B(:,10), 'c-d', 'LineWidth', 2)
plot(B(:,1), B(:,12), 'k-s', 'LineWidth', 2)
plot(B(:,1), B(:,14), 'b-s', 'LineWidth', 2)
plot(B(:,1), B(:,16), 'r-d', 'LineWidth', 2)
plot(B(:,1), B(:,18), 'g-^', 'LineWidth', 2)
plot(B(:,1), B(:,20), 'm-*', 'LineWidth', 2)
plot(B(:,1), B(:,22), 'c-o', 'LineWidth', 2)
plot(B(:,1), B(:,24), 'k-x', 'LineWidth', 2)
hold off

title(['test HPMPC BLAS'])
xlabel('matrix size n')
ylabel('Gflops')
axis([0 B(end,1) 0 A(1)*A(2)])
legend('gemm', 'syrk', 'trmm', 'potrf\_copy','gemv\_n', 'gemv\_t','trmv\_n', 'trmv\_t','trsv\_n', 'trsv\_t','symv', 'mvmv', 'Location', 'NorthEast')

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

