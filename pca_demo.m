%% PCA Demonstration
% This script illustrates PCA and Kernel PCA with some toy examples.
% 
% Goker Erdogan
% 4 June 2015

%% PCA for bivariate normal
% generate bivariate normal data
X = mvnrnd([0, 0], [1, .9; .9, 1], 200);

C = X'*X;
[ev, ed] = eig(C);

%% plots for bivariate normal
hold on
scatter(X(:,1), X(:,2))
axis('equal')
% plot the eigenvectors
[xf, yf] = ds2nfu([0, ev(1,1)], [0, ev(1,2)]);
annotation('arrow', xf, yf)
[xf, yf] = ds2nfu([0, ev(2,1)], [0, ev(2,2)]);
annotation('arrow', xf, yf)

%% Kernel PCA for circle data
% generate data
a = rand(200, 1) .* 2*pi;
X = [sin(a), cos(a)] + (.1 .* randn(200, 2));

% calculate kernel matrix (using radial, i.e., gaussian, kernel)
K = exp(-squareform(pdist(X)));

% kernel PCA. calculate eigenvectors.
[ev, ed] = eigs(K, 2);

%% plots for circle data
figure
scatter(X(:,1), X(:,2), 40, a, 'filled')
axis('equal')
figure
scatter(ev(:,1), ev(:,2), 40, a, 'filled')

%% Kernel PCA for quadratic data

a = randn(200, 1);
X = [a, .5*a.^2] + (.15 .* randn(200, 2));

% calculate kernel matrix (using polynomial kernel)
K = (X*X' + 1).^2;

% kernel PCA. calculate eigenvectors.
[ev, ed] = eigs(K, 2);

%% plots for quadratic data
figure
scatter(X(:,1), X(:,2), 40, a, 'filled')
axis('equal')
figure
scatter(ev(:,1), ev(:,2), 40, a, 'filled')


