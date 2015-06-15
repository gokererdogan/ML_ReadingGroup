% Hamiltonian MCMC
% See Neal, R. M. (2011). MCMC Using Hamiltonian Dynamics. 
%   In Brooks, S., Gelman, A., Jones, G., and Meng, X., editors, 
%   Handbook of Markov Chain Monte Carlo, chapter 5, pages 113?162. 
%   Chapman and Hall / CRC Press
% for a very nice introduction to Hamiltonian MCMC
%
% Goker Erdogan
% 4 June 2015
function [x, p] = hamiltonian_mcmc_sample(x0, U, gradU, eps, L, N)
%HAMILTONIAN_MCMC_SAMPLE Sample from U using Hamiltonian MCMC
%   x0: initial x (Dx1 vector)
%   U: Potential function (i.e., -log of target prob. dist. P)
%   gradU: Function that calculates derivative of U
%   eps: Step size in Hamiltonian dynamics updates
%   L: Number of leapfrog states
%   N: Number of samples

% get the number of dimensions
D = size(x0, 1);

% samples
x = zeros(N,D);
% momentum variables
p = zeros(N,D);

cx = x0;
for i = 1:N
    % sample momentum
    cp = randn(D, 1);
    oldp = cp;
    oldx = cx;
    for l = 1:L
        % update x, p according to Hamiltonian dynamics
        phalf = cp - (eps .* gradU(cx)) ./ 2.0;
        cx = cx + (eps .* phalf);
        cp = phalf - (eps .* gradU(cx)) ./ 2.0;
    end
    
    % calculate potential energy
    old_pot = U(oldx);
    new_pot = U(cx);
    
    % calculate kinetic energy
    old_kin = oldp'*oldp / 2.0;
    new_kin = cp'*cp / 2.0;
    
    % calculate acceptance prob.
    dh = min(1, exp((old_pot + old_kin) - (new_pot + new_kin))); 
    
    % accept/reject
    if rand() > dh % reject
        cx = oldx;
        cp = oldp;
    end
    x(i, :) = cx;
    p(i, :) = cp;
end

end

