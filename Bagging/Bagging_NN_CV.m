

clear; 
close all;

% load the training data and test labels
instances=load('training_data.csv'); 
s=rng;
labels=load('training_labels.csv');
M = 9;
cv = cvpartition(labels, 'Kfold', 10);
cv_accuracy = zeros(1, cv.NumTestSets);
param = '-t 0 -c 1 -h 0 -w1 %.3f -w-1 %.3f';
for i = 1 : cv.NumTestSets
fprintf('Iteration #%d\n', i);

% initialize training/testing dataset
training = cv.training(i);
testing = cv.test(i);

x_training = instances(training, :); 
y_training = labels(training, :);
x_testing = instances(testing, :); 
y_testing = labels(testing, :);

% build the weak/base learners
hiddenLayerSize=10;
n = size(x_training, 1);
learners = cell(M, 1);
net=patternnet(hiddenLayerSize);
target_training=ind2vec(y_training');

for m = 1 : M
indices = randsample(n, randi([round(n/2), n]));
w = ones(size(indices, 1), 1);
%positive = numel(indices) / sum(y_training(indices, :) == 1);
%negative = numel(indices) / sum(y_training(indices, :) == -1);
[learners{m},tr] = train(net,x_training',target_training);

end

% predict on the testing data
%%
n = size(x_testing, 1);
predictions = zeros(n, M);
pcount=zeros(M,1);
for m = 1 : M
predictions(:, m) = vec2ind(learners{m}(x_testing'))';
%pcount(m)=sum(y_testing == predictions(:,m));
end
yprediction=zeros(n,1);
for ii=1:n
    yprediction(ii)=mode(predictions(ii,:));
end
%predictions = mean(predictions,2);
%I=find(pcount==max(pcount(:)));
cv_accuracy(i) = sum(y_testing == yprediction) / size(y_testing, 1);
end
fprintf('Accuracy => [%s]\nMean => %s\n', num2str(cv_accuracy), num2str(mean(cv_accuracy)));