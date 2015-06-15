% This is a script for finding the maximum margin linear classifier for
% linearly separable data. This is a simple problem usually studied as a
% motivating example for SVM.
%
% Here is the formulation
%   min 0.5 w^T*w
%       s.t. y_i (w^T*x_i - w_0) >= 1 \forall i
%
% Using Lagrange multipliers \alpha, the dual form turns out to be
%   max -.5 sum_i sum_j \alpha_i \alpha_j y_i y_j x_i^T*x_j + sum_i \alpha_i
%       s.t. \alpha_i >= 0 \forall i
%
% Setting the derivative w.r.t \alpha_i to zero, we get
%   \alpha_i = (1 - sum_{j\neq i} \alpha_j y_i y_j x_i^T x_j) / (x_i^T x_i)
%
% Note that we have ignored the inequality constraints. In the below
% optimization code, we set \alpha_i to zero if it becomes negative. This
% can be seen as a projection step in a projected gradient descent
% procedure.
%
% Goker Erdogan (gokererdogan@gmail.com)
% 20 May 2015

colormap([0 0 1; 1 0 0; 0 1 0;])

% number of data points
n = 10;
% number of passes over all alpha
runs = 50;

% sample 4 point data. useful for testing. 
%   w should be [1 -1], w0 should be 0
% x = [-1 1; -.5 2; 1 -1; 2 -1.5];
% y = [-1; -1; 1; 1];

% generate random data
x = randn(n, 2);
y = ones(n, 1);
y((x(:,1) - x(:,2)) < 0) = -1;

% plot the data
scatter(x(:,1), x(:,2), 50, y, 'filled')
% axis equal

% randomly initialize alpha
alpha = abs(randn(n, 1).*0.01);

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
fprintf('Support vector indices: %s\n', sprintf('%d ', svi));
fprintf('Support vector weights: %s\n', sprintf('%.4f ', alpha(svi)));

% calculate weight vector
w = sum(repmat(alpha .* y, 1, 2) .* x, 1);
w = w';
% calculate bias term
w0 = ((y(svi(1)) * w' * x(svi(1),:)') - 1) ./ y(svi(1));
fprintf('Estimated weight vector: %s\n', sprintf('%.4f ', w));
fprintf('Bias: %f\n', w0);

hold on
% plot support vectors
hsv = plot(sv(:,1),sv(:,2),'ko','MarkerSize',10);

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