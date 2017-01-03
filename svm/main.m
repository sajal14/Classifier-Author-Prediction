trainX = dlmread('../train/words_train.txt');
trainY = dlmread('../train/genders_train.txt');

p = randperm(4998,2500);
ind = 1:4998;
p2 = setdiff(ind,p);

trainData = trainX(p,:);
testData = trainX(p2,:);

trainLabels = trainY(p,:);
testLabels = trainY(p2,:);

addpath('libsvm')


% X = sparse(trainData);
% X_2 = sparse(testData);
Y = trainLabels;
Y_2 = testLabels;

X = trainData;
X_2 = testData;

K = kernel_intersecK = kernel_intersection(X,X);
Ktest = kernel_intersection(X, X_2);tion(X,X);
Ktest = kernel_intersection(X, X_2);

crange = 10.^[-10:2:3];
for i = 1:numel(crange)
    acc(i) = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -v 10 -c %g', crange(i)));
end

[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));


model = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -c %g', crange(bestc)));

[yhat acc vals] = svmpredict(Y_2, [(1:size(Ktest,1))' Ktest], model);

accuracy = 1 - nnz(yhat- Y_2)/ size(testLabels,1);




