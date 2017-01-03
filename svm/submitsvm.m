trainX = dlmread('../train/words_train.txt');
trainY = dlmread('../train/genders_train.txt');
testX = dlmread('../test/words_test.txt');
addpath('libsvm')


X = sparse(trainX);
X_test = sparse(testX);
Y = trainY;
Y_2 = zeros(size(testX,1),1);
% Y_2 = testLabels;

K = kernel_intersection(X,X);
Ktest = kernel_intersection(X, X_test);

crange = 10.^[-10:2:3];
for i = 1:numel(crange)
    acc(i) = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -v 10 -c %g', crange(i)));
end

[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));


model = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -c %g', crange(bestc)));

[yhat acc vals] = svmpredict(Y_2, [(1:size(Ktest,1))' Ktest], model);

% accuracy = 1 - nnz(yhat- Y_2)/ size(testLabels,1);




