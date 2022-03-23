function [x] = conjugate_gradient(A, b)
x = [];
residual = b;
direction = b;
previous_residual = residual;

for j = 1:100
    alpha = dot(residual',residual)/(direction' * A * direction);
    x = x + alpha*direction;
    residual = residual - alpha*A*direction;
    beta = dot(residual', residual)/dot(previous_residual', previous_residual);
    direction = residual + beta*direction;
end

end