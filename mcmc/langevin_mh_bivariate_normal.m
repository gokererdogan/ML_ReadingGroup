% Sampling from a bivariate normal using Langevin algorithm
%  Langevin algorithm is a Metropolis-Hastings algorithm based on the
%  Langevin diffusion process. We use the gradient of the target
%  probability to bias the random-walk towards high probability regions.
%
% Here is the Langevin update equation
%   x(t+1) = x(t) + (v^2 / 2) * grad(log(f(x(t))) + v*n
%   where n ~ N(0, I)
% x(t+1) is accepted according to the standard Metropolis-Hastings rule.
%
% Goker Erdogan
% gokererdogan@gmail.com
% 15 June 2015


%% Univariate case
mu = 0;
v = 1;

% step size
h = 1;
% proposal variance
varp = h^2;

% number of samples
N = 10000;
samples = zeros(N,1);

x = 0;
for n = 1:N
    % update equation: x(t+1) = x(t) - (h^2 * (x(t) - mu)) / 2v^2 + h*n
    xmean = x - ((h^2 * (x - mu)) / (2*v^2));
    xp = normrnd(xmean, varp);
    
    % acceptance step. a = p(xp)/p(x) * q(x|xp)/q(xp|x)
    a = normpdf(xp, mu, v) / normpdf(x, mu, v);
    xpmean = xp - ((h^2 * (xp - mu)) / (2*v^2));
    q_xp_to_x = normpdf(x, xpmean, varp);
    q_x_to_xp = normpdf(xp, xmean, varp);
    a = a * (q_xp_to_x / q_x_to_xp);
    
    if rand() < a
        x = xp;
    end
    % 
    samples(n) = x;
end

histogram(samples, 'Normalization', 'pdf')
mean(samples)
var(samples)

% One observes that the quality of the sampling is quite dependent on the
% value of h (step size).

%% Bivariate case

% the update equation for the bivariate case is analogous.
%   x(t+1) = x(t) + (h^2 / 2) * inv(S) * (x - mu) + h*n
%   where n ~ N(0, I_{2x2})

% correlation coefficient
r = 0.5;
% mean
mu = [0; 0];
% variance of x1
vx1 = 1;
sx1 = sqrt(vx1);
% variance of x2
vx2 = 1;
sx2 = sqrt(vx2);
covx1x2 = r * sx1 * sx2;
% covariance matrix
S = [vx1, covx1x2; covx1x2, vx2];

% step size for the Langevin update
h = 1;
% proposal covariance
Sp = h .* eye(2);

% number of samples
N = 1000;
samples = zeros(N, 2);

% initial point
x = [0; 0];
for n = 1:N
    xmean = x - (h^2/2) * (S \ (x - mu));
    % draw proposed x from proposal
    xp = mvnrnd(xmean, Sp)';
    
    % calculate acceptance ratio
    a = mvnpdf(xp, mu, S) / mvnpdf(x, mu, S);
    xpmean = xp - (h^2/2) * (S \ (xp - mu));
    q_xp_to_x = mvnpdf(x, xpmean, Sp);
    q_x_to_xp = mvnpdf(xp, xmean, Sp);
    a = a * (q_xp_to_x / q_x_to_xp);
    % accept/reject
    if rand() < a
        x = xp;
    end
    
    samples(n, :) = x;
end

% plot samples
hold on
axis equal
scatter(samples(:,1), samples(:,2))

% plot contour lines of bivariate gaussian
draw_contours(@(t) (mvnpdf(t, mu', S)), [-5, 5], [-5, 5], 100);

% estimated mean and covariance matrices
mean(samples)
cov(samples)