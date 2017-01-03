function [X Y] = make_sparse(data, vocab)
% Returns a sparse matrix representation of the data.
%
% Usage:
%
%  [X, Y] = MAKE_SPARSE(DATA, VOCAB)
%
% For a struct array of newsgroup examples DATA, and a cell array
% vocabulary VOCAB, returns a sparse matrix X where X(i,j) is the # of
% times that word j occured in example i. Note that since X is sparse, only
% non-zero entries are stored. Also returns a binary label vector Y for the
% given data.

% Strip out -1's (unknown words in test set) from the counts.
for i = 1:numel(data)
    data(i).counts = data(i).counts(~[data(i).counts(:,1)==-1], :);
end

% data_cnt = data.counts
% data_labels = data.label
% data_labels
% size(data_labels)
% size(data_cnt)

rowidx = [];
colidx = [];
values = [];

for i = 1:numel(data)
    counts = data(i).counts;
    num_ele = size(counts,1);
    rows = repmat(i,1,num_ele);
    rowidx = [rowidx rows];
    colidx = [colidx counts(:,1)'];
    values = [values counts(:,2)'];
end

size(rowidx)
size(colidx)
size(values)

% YOUR CODE GOES HERE. Your job is to determine in rowidx, colidx, and values
% for the sparse matrix. If D is the number of NON ZERO values of X, then
% these are each D x 1 vectors. The idea here is that Matlab will create a
% sparse matrix data structure such that:
%                  X(rowidx(i),colidx(i)) = values(i).
% For more information about sparse matrices, see doc sparse.
%
% P.S., if we didn't use a sparse matrix, our full X matrix would take up
% 500 MB of memory!

X = sparse(rowidx, colidx, values, numel(data), numel(vocab))
% Do not touch this: this computes the text label to a numeric 0-1 label,
% where 1 examples are mac newsgroup postings.
Y = double(cellfun(@(x)isequal(x, 'comp.sys.mac.hardware'), {data.label})');
