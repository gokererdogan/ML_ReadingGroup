% Sampling from a bivariate normal using Gibbs sampling
% 
% Goker Erdogan
% gokererdogan@gmail.com
% 12 June 2015

% correlation coefficient
r = 0.5;
% mean
mu = [0, 0];
% variance of x1
vx1 = 1;
sx1 = sqrt(vx1);
% variance of x2
vx2 = 1;
sx2 = sqrt(vx2);
covx1x2 = r * sx1 * sx2;
% covariance matrix
S = [vx1, covx1x2; covx1x2, vx2];

% number of samples
N = 1000;
samples = zeros(N, 2);

% initial point
x = [0, 0];
for n = 1:N
    % sample from conditionals
    % (https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case)
    x(1) = normrnd(mu(1) + (sx1 / sx2) * r * (x(2) - mu(2)), (1 - r^2) * vx1);
    x(2) = normrnd(mu(2) + (sx2 / sx1) * r * (x(1) - mu(1)), (1 - r^2) * vx2);
    samples(n, :) = x;
end

% plot samples
hold on
axis equal
scatter(samples(:,1), samples(:,2))

% plot contour lines of bivariate gaussian
draw_contours(@(x) (mvnpdf(x, mu, S)), [-5, 5], [-5, 5], 100);

% estimated mean and covariance matrices
mean(samples)
cov(samples)