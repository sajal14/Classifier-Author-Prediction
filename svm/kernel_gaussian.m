function K = kernel_gaussian(X, X2, sigma)
% Evaluates the Gaussian Kernel with specified sigma
%
% Usage:
%
%    K = KERNEL_GAUSSIAN(X, X2, SIGMA)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the Guassian kernel
% with parameter sigma=20.

n = size(X,1);
m = size(X2,1);
K = zeros(m, n);

% HINT: Transpose the sparse data matrix X, so that you can operate over columns. Sparse
% column operations in matlab are MUCH faster than row operations.

% YOUR CODE GOES HERE.

% temp = reshape(X', [1 size(X,2) n]);
% rep_X1 = repmat(temp, [m 1 1]);
% rep_X2 = repmat(X2, [1 1 n]);
% diff = rep_X2 - rep_X1;
% distMat = sqrt(sum(diff.^2, 2));
% distMat = squeeze(distMat);
% K = exp(-distMat.^2/(2*sigma.^2));

X1 = X';
X2 = X2';

K = []

for i = 1:size(X2,2)
    for j = 1:size(X1,2)
        size(X2(:,i));
        size(X1(:,j));
        diff = X2(:,i)-X1(:,j);
        K(i,j) = sum((diff.^2));        
    end    
end
   
K = exp(-K./(2*sigma^2));



