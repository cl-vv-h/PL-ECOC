xm = [1,2,3];
xm = [xm;xm-1];
ym = [1,0];
model = fitcsvm(xm,ym);
models = cell(2,2);
models{1,1}=model;
x=[1,3,2];
x=[x;x];

% y = x-[1,3,2];
% disp(y);
% n = vecnorm(x,2,2);
% disp(n);
X = [[1,2,3];[2,3,4]];
t = sum(abs(X).^2,2).^(1/2);
disp(t);