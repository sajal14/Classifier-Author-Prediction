%% Plots/submission for SVM portion, Question 1.
addpath ./libsvm
%% Put your written answers here.
clear all
answers{1} = 'Kernal intersection is working best. This kernal uses the count of number of common words in the two topics. As the number of common words are more, greater is the similarity between documents. In this case, the classification labels our Windows PC and Mac which is predicted based on the comments of the user. There would be many words which would be common to either Windows based PC or Mac. (For e.g. "Maverick" is the OS for Mac, and someone would rarely use it in conversation of Windows PC.) Thus this kernal gives a good and sensible similarity relation.';

save('problem_1_answers.mat', 'answers');

%% Load and process the data.

load ../data/windows_vs_mac.mat;
[X Y] = make_sparse(traindata, vocab);
[Xtest Ytest] = make_sparse(testdata, vocab);
size(X)
size(Y)
%% Bar Plot - comparing error rates of different kernels

% INSTRUCTIONS: Use the KERNEL_LIBSVM function to evaluate each of the
% kernels you mentioned. Then run the line below to save the results to a
% .mat file.
% k = @(x,x2) kernel_intersection(x, x2);
% [test_err info] = kernel_libsvm(X, Y, Xtest, Ytest, k)
% test_err
results.linear = 0.1130 % ERROR RATE OF LINEAR KERNEL GOES HERE
results.quadratic = 0.1258 % ERROR RATE OF QUADRATIC KERNEL GOES HERE
results.cubic = 0.1617 % ERROR RATE OF CUBIC KERNEL GOES HERE
results.gaussian = 0.1361 % ERROR RATE OF GAUSSIAN (SIGMA=20) GOES HERE
results.intersect = 0.0988 % ERROR RATE OF INTERSECTION KERNEL GOES HERE

% Makes a bar chart showing the errors of the different algorithms.
algs = fieldnames(results);
for i = 1:numel(algs)
    y(i) = results.(algs{i});
end
bar(y);
set(gca,'XTickLabel', algs);
xlabel('Kernel');
ylabel('Test Error');
title('Kernel Comparisons');

print -djpeg -r72 plot_1.jpg;
