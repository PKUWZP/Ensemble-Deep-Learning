clear; 
close all;

s=rng;
instances=load('training_data.csv'); 
labels=load('training_labels.csv');

cv = cvpartition(labels, 'Kfold', 10);
cv_accuracy = zeros(1, cv.NumTestSets);
M = 9;
%params = '-t 0 -c 1 -h 0 -w1 %.3f -w-1 %.3f';
for i = 1 : cv.NumTestSets
fprintf('Iteration #%d\n', i);
training = cv.training(i);
testing = cv.test(i);
% input data for the first level learners
x_training = instances(training, :); 
y_training = labels(training, :);
target_training=ind2vec(y_training');
hiddenLayerSize=10;
net=patternnet(hiddenLayerSize);
% train individual learners on the training data (first level)
models = cell(M, 1);
n = size(x_training, 1);
for m = 1 : M
indices = randsample(n, randi([round(n/2), n]));
%w = ones(size(indices, 1), 1);
%positive = numel(indices) / sum(y_training(indices, :) == 1);
%negative = numel(indices) / sum(y_training(indices, :) == -1);
[models{m},tr] = train(net, x_training',target_training);
end
% prepare the data for the second level learner using the outputs from
% the first level learners
x = zeros(size(instances, 1), m);
y = labels;
for m = 1 : M
x(:, m) = vec2ind(models{m}(instances'))';
end
% train the second level learner with LDA 

model = ClassificationDiscriminant.fit(x(training,:),y(training,:));
% test performance over the testing dataset
predictions = predict(model, x(testing,:));
cv_accuracy(i) = sum(predictions == y(testing, :)) / size(y(testing, :), 1);
end
fprintf('Accuracy => [%s]\nMean => %s\n', num2str(cv_accuracy), num2str(mean(cv_accuracy)));