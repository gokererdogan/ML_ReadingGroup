%% Hamiltonian MCMC Demonstration
% Here we demonstrate Hamiltonian MCMC with some examples.
% Sampling procedure is implemented in hamiltonian_mcmc_sample function.
%
% Goker Erdogan
% 4 June 2015
%% sample from N(0,1)
u = @(x) (x^2 / 2.0);
gU = @(x) (x);
[x, p] = hamiltonian_mcmc_sample(0, u, gU, 0.1, 20, 10000);
histogram(x, 'Normalization', 'pdf')

%% sample from bivariate independent normal
u = @(x) (x'*x / 2.0);
gU = @(x) (x);
[x, p] = hamiltonian_mcmc_sample([0; 0], u, gU, 0.1, 20, 10000);
scatter(x(:,1), x(:,2))

%% sample from bivariate correlated normal
s = [1, .8; .8, 1];
u = @(x) (x'*(s\x) / 2.0);
gU = @(x) (s\x);
[x, p] = hamiltonian_mcmc_sample([0; 0], u, gU, 0.1, 20, 10000);
scatter(x(:,1), x(:,2))