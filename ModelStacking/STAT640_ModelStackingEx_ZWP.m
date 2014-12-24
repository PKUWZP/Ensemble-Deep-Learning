%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this function is the implementation of Model Stacking algorithms 
%with Feed-forward neural networks as the base learners 
%and Linear Discriminant Analysis as the second-level learner

%Author: Zhipeng Wang, Department of Physics and Center for Theoretical
%Biological Physics, Rice University, Houston, TX, 77005

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; 
close all;
s=rng;
instances=load('training_data.csv'); 
labels=load('training_labels.csv');
testdata=load('test_data.csv');

It=5000;
%cv = cvpartition(labels, 'Kfold', 10);
cv_accuracy = zeros(1, It);
M = 10;
predictions = zeros(size(testdata,1),M);
%params = '-t 0 -c 1 -h 0 -w1 %.3f -w-1 %.3f';
for i = 1 : It  % interations
fprintf('Iteration #%d\n', i);
training = instances;
testing = testdata;
% input data for the first level learners
x_training = instances; 
y_training = labels;
target_training=ind2vec(y_training'); %transform the labels into target matrix
hiddenLayerSize=10; % less number of neurons to avoid overfitting
net=patternnet(hiddenLayerSize);
% train individual learners on the training data (first level)
models = cell(M, 1);
n = size(x_training, 1);
for m = 1 : M
%indices = randsample(n, randi([round(n/2), n]));
[models{m},tr] = train(net, x_training',target_training);
end
% prepare the data for the second level learner using the outputs from
% the first level learners
x = zeros(size(instances, 1), m);
t = zeros(size(testdata,1), m);
y = labels;
for m = 1 : M
x(:, m) = vec2ind(models{m}(instances'))';
t(:, m) = vec2ind(models{m}(testdata'))';
end
% train the second level learner with LDA 

model = ClassificationDiscriminant.fit(x,y);
% test performance over the testing dataset

predictions (:,i) = predict(model, t);
%cv_accuracy(i) = sum(predictions == y(testing, :)) / size(y(testing, :), 1);
end
%fprintf('Accuracy => [%s]\nMean => %s\n', num2str(cv_accuracy), num2str(mean(cv_accuracy)));

FS=1:size(predictions,1);
FinalPL=zeros(size(predictions,1),2);
FinalPL(:,1)=FS;
for j = 1: size(predictions,1)
    FinalPL(j,2)=mode(predictions(j,:));  % voting for classification
end
csvwriteh('STAT_640_ZWP.csv',FinalPL,{'ID','Prediction'}); % save the final file to be STAT_640_ZWP.csv
