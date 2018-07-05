%% Logistic regression for two classes
% Implements the iteratively reweighthed least squares algorithm (IRLS)
% See Elements of Statistical Learning, pp. 120-121 for details of the
% algorithm.
%
% 29 April 2015
% Goker Erdogan

show_plots(1)

%% generate data
% NOTE: if the data is linearly separable, IRLS won't converge because log
% likelihood can be increased indefinitely by making the dividing plane
% steeper and steeper.
X = mvnrnd([-2 -2], 3*eye(2), 100);
X = [X; mvnrnd([2 2], 3*eye(2), 100)];
% add bias term
X = [X ones(200, 1)];
y = zeros(200, 1);
y(1:100) = 1;
figure
hold on
scatter(X(:,1), X(:,2), 30, y, 'filled')

%% IRLS
eps = 1e-6;

% number of input samples
N = size(X,1);
% number of dimensions
D = size(X,2);

% regression weights
w = randn(D, 1) * 0.0001;

ll = zeros(100,1);
ll_old = 0;
while 1
    % calculate predictions
    p = exp(X*w) ./ (1 + exp(X*w));
    ll = sum(y.*log(p) + (1-y).*log(1-p));
    ll
    % reweighting matrix
    W = diag(p.*(1-p));
    % update regression weights
    w = w + (X'*W*X)\X'*(y-p);

    if abs(ll - ll_old) < eps
        break
    end
    
    ll_old = ll;
end

%% plot decision line
slope = -w(1)/w(2);
intercept = -w(3) - log(2);
refline(slope, intercept)