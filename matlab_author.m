clc
clear all

p = randperm(12113,6000);
ind = 1:12113;
p2 = setdiff(ind,p);

addpath('liblinear')
addpath('libsvm')

i0 = dlmread('train/ind_0_49.txt');
i1 = dlmread('train/ind_50_99.txt');
i2 = dlmread('train/ind_100_499.txt');
i3 = dlmread('train/ind_500_999.txt');
i4 = dlmread('train/ind_1000_1999.txt');
trainX_1 =[i0, i1, i2, i3 i4];

trainX_1 = trainX_1(2:end,:);
trainY = dlmread('train/train_labels');
trainData = trainX_1(p,:);
testData = trainX_1(p2,:);
trainLabels = trainY(p,:);
testLabels = trainY(p2,:);

SVMModel1 = fitcsvm(trainData,trainLabels,'KernelFunction','kernel_intersection','boxconstraint',0.01);
[label_train1,Score_train] = predict(SVMModel1,trainData);
[label1,Score] = predict(SVMModel1,testData);

acc1 = 1 - nnz(label1 - testLabels)/size(testLabels,1);
acctrain1 = 1 - nnz(label_train1 - trainLabels)/size(trainLabels,1);


% Top MI Bigram Model

trainX_2 = dlmread('train/train_mi_bigram.txt');
trainX_2 = trainX_2(2:end,:);

trainData = trainX_2(p,:);
testData = trainX_2(p2,:);

SVMModel2 = fitcsvm(trainData,trainLabels,'KernelFunction','kernel_intersection');
[label_train2,Score_train2] = predict(SVMModel2,trainData);

[label2,Score] = predict(SVMModel2,testData);

acc2 =1 - nnz(label2 - testLabels)/size(testLabels,1);
acctrain2 = 1 - nnz(label_train2 - trainLabels)/size(trainLabels,1);


% Top Uni With MI Model

trainX_3 = dlmread('train/uni_tr_model.txt');
trainX_3 = trainX_3(2:end,:);

trainData = trainX_3(p,:);
testData = trainX_3(p2,:);

SVMModel3 = fitcsvm(trainData,trainLabels,'KernelFunction','kernel_intersection');
[label_train3,Score_train3] = predict(SVMModel3,trainData);

[label3,Score] = predict(SVMModel3,testData);

acc3 =1 - nnz(label3 - testLabels)/size(testLabels,1);
acctrain3 = 1 - nnz(label_train3 - trainLabels)/size(trainLabels,1);


labelFinal = zeros(size(testLabels,1),1);
ind1 = find(label2==1 | label2==1 | label3==1  );
labelFinal(ind1)=1;
acc_final =1 - nnz(labelFinal - testLabels)/size(testLabels,1);











