% Implementation with no optimization

% function [Q, A] = qr_householders(A)
% [m, n] = size(A);
% Q = eye(m);
% 
% for j = 1:n
%     [u, ~] = householder_vector(A(j:m, j));
%     H = eye(length(u)) - 2*(u*u');
%     A(j:end, j:end) = H * A(j:end, j:end);
%     Q(:, j:end) = Q(:, j:end)*H;
% end
% end

function [Q, A] = qr_householders(A)
[m, n] = size(A);
Q = eye(m);

for j = 1:n
    [u, ~] = householder_vector(A(j:m, j));
    H = eye(length(u)) - 2*(u*u');
    A(j:end, j:end) = H * A(j:end, j:end);
    Q(:, j:end) = Q(:, j:end)*H;
end
end