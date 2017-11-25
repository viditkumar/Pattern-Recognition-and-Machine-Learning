%% Ques 4
%%% Build a SVM Classifier for the classes !1 , !2 and !3 using a Radial Basis Function
%%% Kernel. Plot a graph depicting the recognition accuracy on the test data Test1.mat,
%%% Test2.mat and Test3.mat for different values of penalty factors C and precisions
%%% of the Radial Basis Function.

clc
% clear
close all
format long

% addpath to libsvm
%% Load data

load('Pattern1.mat');
load('Pattern2.mat');
load('Pattern3.mat');
load('Test1.mat');
load('Test2.mat');
load('Test3.mat');

%% Cell to matrix. Preparing training and testing dataset

[~ , m] = size(test_pattern_1);

for i = 1:m
   A = test_pattern_1{i};
   X(i , :) = A; 
   A = test_pattern_2{i};
   X(100+i , :) = A;
   A = test_pattern_3{i};
   X(200+i , :) = A;
end

[~ , n] = size(train_pattern_1);

for i = 1:n
   A = train_pattern_1{i};
   train_data(i , :) = A; 
   A = train_pattern_2{i};
   train_data(200+i , :) = A;
   A = train_pattern_3{i};
   train_data(400+i , :) = A;
end

train_output = [1*ones(200,1) ; 2*ones(200,1) ; 3*ones(200,1)];
test_output = [1*ones(100,1) ; 2*ones(100,1) ; 3*ones(100,1)];

%% SVM Train and SVM Predict

k = 1;
for i = 1:10:200
    for j = 0.05:0.1:1.1
        options = sprintf('-t  2 -g %d -c %d ' , j , i);
        svm_model = svmtrain(train_output , train_data , options);
        [~ , accuracy , ~] = svmpredict(test_output , X , svm_model , '-q');
        Accuracy(k , :) = [i , j , accuracy];
        k = k + 1;
    end    
end

%% Finding maximum accuracy

[~, m] = max(Accuracy(: , 3));

maximum_accuracy = Accuracy(m , 3)
C_at_max = Accuracy(m , 1)
gamma_at_max = Accuracy(m , 2)

%% Plotting the accuracies

gamma = 0.05:0.1:1.1;
C = 1:10:200;
[gamma , C]=meshgrid(gamma,C);
Acc3 = reshape(Accuracy(: , 3) , [length(C) , length(gamma)]);
surf(gamma,C ,Acc3);



