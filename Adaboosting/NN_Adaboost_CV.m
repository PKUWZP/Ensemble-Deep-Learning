clear; 
close all;

s=rng;
instances=load('training_data.csv'); 
labels=load('training_labels.csv');
M = 9;
cv = cvpartition(labels, 'Kfold', 10);
cv_accuracy = zeros(1, cv.NumTestSets);

for i = 1 : cv.NumTestSets
training = cv.training(i);
testing = cv.test(i);
x_training = instances(training, :); 
y_training = labels(training, :);
x_testing = instances(testing, :); 
y_testing = labels(testing, :);

models = cell(M, 1);
n = size(x_training, 1);
w = repmat(1 / n, n, M);
alpha = zeros(M, 1);
eps = zeros(M, 1);

hiddenLayerSize=10;
net=patternnet(hiddenLayerSize);
target_training=ind2vec(y_training);

% Weak/base Learners 
for m = 1 : M
positive = size(y_training, 1) / sum(y_training == 1);
negative = size(y_training, 1) / sum(y_training == -1);
%models{m} = svmtrain(w(:, m) ./ min(w(:, m)), y_training, x_training, sprintf(param, positive, negative));
[models{m},tr] = train(net,x_training',target_training);
predictions = models{m}(x_training');
%predictions = svmpredict(y_training, x_training, models{m});

I = (predictions ~= y_training);
eps(m) = (w(:, m)' * I) / sum(w(:, m));
alpha(m) = log ( (1 - eps(m)) / eps(m) );

if m < M
w(:, m + 1) = w(:, m) .* exp(alpha(m) * I);
end

end
predictions = zeros(size(y_testing, 1), M);
for m = 1 : M
predictions(:, m) = models{m}();
end
predictions = sign(predictions * alpha);
cv_accuracy(i) = sum(predictions == y_testing) / size(y_testing, 1);
end
fprintf('Accuracy => [%s]\nMean => %s\n', num2str(cv_accuracy), num2str(mean(cv_accuracy) * 100));