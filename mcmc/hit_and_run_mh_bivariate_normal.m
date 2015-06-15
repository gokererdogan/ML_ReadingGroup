% Sampling from a bivariate normal using hit-and-run Metropolis-Hastings
%   Hit-and-run algorithm is simply a Gibbs sampling algorithm using random
%   directions rather than fixed axial directions. We first randomly pick a
%   direction and then do one step of Gibbs sampling along that direction
%   using the conditional along that direction.
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
x = [0; 0];
for n = 1:N
    % randomly draw a direction
    theta = rand() * 2 * pi;
    d = [cos(theta); sin(theta)];
    % conditional mean and variance
    % These horrendous expressions are the conditional mean and variance
    % along the direction d. These can be derived with a bit of tedious
    % calculation. I haven't been able to simplify these expressions
    % further.
    denom = (vx2 * d(1)^2) + (vx1 * d(2)^2) - (2 * r * sx1 * sx2 * d(1) * d(2));
    cm = -((vx2 * x(1) * d(1)) + (vx1 * x(2) * d(2)) - (r * (x(1) * d(2) + x(2) * d(1))));
    cm = cm / denom;
    cv = (vx1 * vx2 * (1 - r^2)) / denom;
    lambda = normrnd(cm, cv);
    % update x
    x = x + (lambda * d);
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

