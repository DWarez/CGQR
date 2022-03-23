% Function to compute an householder vector for a given vector
function [u, s] = householder_vector(x)
v = x;  % copy the input vector 
s = norm(x); % compute the 2-norm

% change the sign of the norm if needed, for stability purposes
if(x(1) >= 0)
    s = -s;
end

v(1) = v(1) - s;
n = norm(v);  % recompute the norm
u = v/n;  % normalize the output vector
end