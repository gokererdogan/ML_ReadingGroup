function [ax] = draw_contours(func, xr, yr, N)
% draw_contours Draw the contours of func(x, y) over the domain xr, yr
% using N points across each dimension
%   
% Goker Erdogan
% gokererdogan@gmail.com
% 15 Jun 2015

% form the cartesian product of xr with yr with itself
tx = linspace(xr(1), xr(2), N);
ty = linspace(yr(1), yr(2), N);
tx = repmat(tx, N, 1);
ty = repmat(ty', 1, N);
pts = [tx(:), ty(:)];

% evaluate function and put them into matrix form (for contour function)
p = func(pts);
z = zeros(N, N);
z(:) = p;

ax = contour(tx, ty, z);
end

