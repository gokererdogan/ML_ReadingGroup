% This is a script for fitting support vector classifiers (i.e., SVMs with
% linear kernel). 
%
% Here is the formulation
%   min 0.5 w^T*w + C sum_i e_i
%       s.t. y_i (w^T*x_i - w_0) >= 1 \forall i
%            e_i >= 0
%
% We have simply added slack variables to max margin classifier formulation
% and added these slack variables as a penalty term with weight C.
%
% Using Lagrange multipliers \alpha for margin inequalities, the dual form 
% turns out to be
%   max -.5 sum_i sum_j \alpha_i \alpha_j y_i y_j x_i^T*x_j + sum_i \alpha_i
%       s.t. C >= \alpha_i >= 0 \forall i
%
% Note that the only difference from the max margin classifier dual form is
% that alpha are also bounded from above (by C)
%
% Again, setting the derivative w.r.t \alpha_i to zero, we get
%   \alpha_i = (1 - sum_{j\neq i} \alpha_j y_i y_j x_i^T x_j) / (x_i^T x_i)
%
% Note that we have ignored the inequality constraints. In the below
% optimization code, we set \alpha_i to zero if it becomes negative, and to
% C if it gets larger than C. This can be seen as a projection step in a 
% projected gradient descent procedure. This procedure gives identical 
% results to solving the optimization problem using MATLAB's quadprog.
%
% Note that data points with \alpha_i > 0 are support vectors. Support
% vectors with \alpha_i < C also lie on the margin (their e_i>=0)
% inequalities are active, i.e., e_i=0). For \alpha_i = C, e_i>=0 is not
% active; therefore, e_i can be > 0.
% Support vectors on the margin are marked with red circles.
%
% We also fit an SVM with linear kernel using MATLAB's fitcsvm function,
% however the results from this function do not match our results exactly.
% Figuring out why that happens is left for future work right now.
%
% Goker Erdogan (gokererdogan@gmail.com)
% 20 May 2015

colormap([0 0 1; 1 0 0; 0 1 0; 0 1 1; 1 1 0; 1 0 1])

% number of data points
n = 20;
% number of passes over all alpha
runs = 50;

% penalty cost
C = 2;

% sample 4 point data. useful for testing. 
%   w should be [1 -1], w0 should be 0
% x = [-1 1; -.5 2; 1 -1; 2 -1.5];
% y = [-1; -1; 1; 1];

% generate random data
% randn('seed', 0);
x = randn(n, 2);
y = ones(n, 1);
y((x(:,1) - x(:,2)) < 0) = -1;

% plot the data
scatter(x(:,1), x(:,2), 25, y, 'filled')
hold on

% randomly initialize alpha
alpha = abs(randn(n, 1).*0.001);

% optimization loop
for r = 1:runs
    old_alpha = alpha;
    for i = 1:n
        k = x(i,:) * x((1:n)~=i,:)';
        a = 1 - sum((y((1:n)~=i) .* alpha((1:n)~=i)) .* y(i) .* k');
        a = a ./ (x(i,:) * x(i,:)');
        alpha(i) = a; %alpha(i) + stepsize * a;
        if alpha(i) < 0
            alpha(i) = 0;
        elseif alpha(i) > C
            alpha(i) = C;
        end
    end
    % look at how much alpha changed
    da = sum((alpha - old_alpha).^2);
    fprintf('Iter %d. Change in alpha: %f\n', r, sum((alpha - old_alpha).^2));
    if da < 1e-9
        break;
    end
end

% print support vector indices and weights
svi = find(alpha>0);
sv = x(svi,:);
svi_onmargin = find(alpha>0 & alpha<C);
sv_onmargin = x(svi_onmargin, :);
fprintf('Support vector indices: %s\n', sprintf('%d ', svi));
fprintf('Support vector weights: %s\n', sprintf('%.4f ', alpha(svi)));

% calculate weight vector
w = sum(repmat(alpha .* y, 1, 2) .* x, 1);
w = w';
% calculate bias term
w0 = ((y(svi_onmargin(1)) * w' * x(svi_onmargin(1),:)') - 1) ./ y(svi_onmargin(1));
fprintf('Estimated weight vector: %s\n', sprintf('%.4f ', w));
fprintf('Bias: %f\n', w0);
fprintf('\n');

% plot support vectors
hsv = plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);
% plot the support vectors on margin with a different color
hsv_om = plot(sv_onmargin(:,1),sv_onmargin(:,2),'ro','MarkerSize',10);

% plot the fitted line
hl = refline(-w(1)/w(2), -w0/w(2));
set(hl, 'Color', 'r');
% plot margins
hmp = refline(-w(1)/w(2), (1-w0)/w(2));
set(hmp, 'Color', 'r');
set(hmp, 'LineStyle', '--');
hmn = refline(-w(1)/w(2), (-1-w0)/w(2));
set(hmn, 'Color', 'r');
set(hmn, 'LineStyle', '--');

%% ALTERNATIVE 1: Use MATLAB's quadprog to solve the optimization problem
H = repmat(y, 1, 2) .* x;
H = H*H';
alpha_qp = quadprog(H, -ones(n,1), [], [], [], [], zeros(n,1), C*ones(n,1));

% print support vector indices and weights
svi_qp = find(alpha_qp>1e-6);
svi_om_qp = find(alpha < C & alpha_qp>1e-6);
fprintf('QP Support vector indices: %s\n', sprintf('%d ', svi_qp));
fprintf('QP Support vector weights: %s\n', sprintf('%.4f ', alpha_qp(svi_qp)));

% calculate weight vector
w_qp = sum(repmat(alpha_qp .* y, 1, 2) .* x, 1);
w_qp = w_qp';
% calculate bias term
w0_qp = ((y(svi_om_qp(1)) * w_qp' * x(svi_om_qp(1),:)') - 1) ./ y(svi_om_qp(1));
fprintf('QP Estimated weight vector: %s\n', sprintf('%.4f ', w_qp));
fprintf('QP Bias: %f\n', w0_qp);
fprintf('\n');

% plot the fitted line
hl_qp = refline(-w_qp(1)/w_qp(2), -w0_qp/w_qp(2));
set(hl_qp, 'Color', 'g');

%% ALTERNATIVE 2: Fit using MATLAB's fitcsvm
svm = fitcsvm(x, y, 'BoxConstraint', C, 'KernelFunction', 'linear', 'Alpha', abs(randn(n, 1).*0.01));
w_m = sum(repmat(svm.Alpha .* y(svm.IsSupportVector), 1, 2) .* x(svm.IsSupportVector,:), 1);
w_m = w_m';
svi_m = find(svm.IsSupportVector);
w0_m = ((y(svi_m(1)) * w_m' * x(svi_m(1),:)') - 1) ./ y(svi_m(1));
fprintf('MATLAB Support vector indices: %s\n', sprintf('%d ', svi_m));
fprintf('MATLAB Support vector weights: %s\n', sprintf('%.4f ', svm.Alpha));
fprintf('MATLAB Estimated weight vector: %s\n', sprintf('%.4f ', w_m));
fprintf('MATLAB Bias: %f\n', w0_m);

% hl_m = refline(-w_m(1)/w_m(2), -w0_m/w_m(2));
% set(hl_m, 'Color', 'b');
% axis equal

% legend([hsv, hl, hl_qp, hl_m],'Support Vectors', 'Our SVC', 'SVC with QP', 'MATLAB SVM');
legend([hsv, hl, hl_qp],'Support Vectors', 'Our SVC', 'SVC with QP');
