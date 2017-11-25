%% Ques-1

close all
clear 
clc

%% Initialization

C1 = zeros(32*32 , 200);
C2 = zeros(32*32 , 200);
C3 = zeros(32*32 , 200);
lambda1 = 0.8;
lambda2 = 0.8;
result = zeros(300, 3);

%% Reading images from folder and reizing them to 32*32

%Folder 1
srcFiles = dir('D:\6th Sem\EE 657\8042 Assignment\Assignment_list\TrainCharacters\1\*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
    filename = strcat('D:\6th Sem\EE 657\8042 Assignment\Assignment_list\TrainCharacters\1\',srcFiles(i).name);
    J = imresize(imread(filename), 0.25);
    C1(:,i) = J(:);
end
C1 = C1';

%Folder 2
srcFiles = dir('D:\6th Sem\EE 657\8042 Assignment\Assignment_list\TrainCharacters\2\*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
    filename = strcat('D:\6th Sem\EE 657\8042 Assignment\Assignment_list\TrainCharacters\2\',srcFiles(i).name);
    J = imresize(imread(filename), 0.25);
    C2(:,i) = J(:);
end
C2 = C2';

%Folder 3
srcFiles = dir('D:\6th Sem\EE 657\8042 Assignment\Assignment_list\TrainCharacters\3\*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
    filename = strcat('D:\6th Sem\EE 657\8042 Assignment\Assignment_list\TrainCharacters\3\',srcFiles(i).name);
    J = imresize(imread(filename), 0.25);
    C3(:,i) = J(:);
end
C3 = C3';

%% Finding means

mean1 = mean(C1)';
mean2 = mean(C2)';
mean3 = mean(C3)';

%% Case-1: The samples of a given character class are modelled by a separate covariance matrix

cov1 = cov(C1) + lambda1*eye(1024);
cov2 = cov(C2) + lambda1*eye(1024);
cov3 = cov(C3) + lambda1*eye(1024);

tot = [C1 ; C2 ; C3];

cov_tot = cov(tot);

for i = 1:1024
   var(i) = cov(cov_tot(: , i));
end

%% Case-2: The samples across all the characters are pooled to generate a common diagonal
%%% covariance matrix . The diagonal entries correspond to the variances of the individual
%%% features, that are considered to be independent

covar_2 = diag(var) + lambda2*eye(1024);

%% Case-3: The covariance matrix of each class is forced to be identity matrix

covar_3 = eye(1024);

%% Reading Test Folder 1

T1 = zeros(32*32 , 100);
srcFiles = dir('D:\6th Sem\EE 657\8042 Assignment\Assignment_list\TestCharacters\TestCharacters\1\*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
    filename = strcat('D:\6th Sem\EE 657\8042 Assignment\Assignment_list\TestCharacters\TestCharacters\1\',srcFiles(i).name);
    J = imresize(imread(filename), 0.25);
    T1(:,i) = J(:);
end

for i = 1:100
    dist(1) = Distance(T1(:,i), mean1, cov1);
    dist(2) = Distance(T1(:,i), mean2, cov2);
    dist(3) = Distance(T1(:,i), mean3, cov3);
    [M, u] =   min(dist);
    result(i, 1) = u;
end

for i = 1:100
    dist(1) = Distance(T1(:,i), mean1, covar_2);
    dist(2) = Distance(T1(:,i), mean2, covar_2);
    dist(3) = Distance(T1(:,i), mean3, covar_2);
    [M, u] =   min(dist);
    result(i, 2) = u;
end

for i = 1:100
    dist(1) = Distance(T1(:,i), mean1, covar_3);
    dist(2) = Distance(T1(:,i), mean2, covar_3);
    dist(3) = Distance(T1(:,i), mean3, covar_3);
    [M, u] =   min(dist);
    result(i, 3) = u;
end
%% Reading Test Folder 2

T1 = zeros(32*32 , 100);
srcFiles = dir('D:\6th Sem\EE 657\8042 Assignment\Assignment_list\TestCharacters\TestCharacters\2\*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
    filename = strcat('D:\6th Sem\EE 657\8042 Assignment\Assignment_list\TestCharacters\TestCharacters\2\',srcFiles(i).name);
    J = imresize(imread(filename), 0.25);
    T1(:,i) = J(:);
end

for i = 1:100
    dist(1) = Distance(T1(:,i), mean1, cov1);
    dist(2) = Distance(T1(:,i), mean2, cov2);
    dist(3) = Distance(T1(:,i), mean3, cov3);
    [M, u] =   min(dist);
    result(i+100, 1) = u;
end

for i = 1:100
    dist(1) = Distance(T1(:,i), mean1, covar_2);
    dist(2) = Distance(T1(:,i), mean2, covar_2);
    dist(3) = Distance(T1(:,i), mean3, covar_2);
    [M, u] =   min(dist);
    result(i+100, 2) = u;
end

for i = 1:100
    dist(1) = Distance(T1(:,i), mean1, covar_3);
    dist(2) = Distance(T1(:,i), mean2, covar_3);
    dist(3) = Distance(T1(:,i), mean3, covar_3);
    [M, u] =   min(dist);
    result(i+100, 3) = u;
end
%% Reading Test Folder 3

T1 = zeros(32*32 , 100);
srcFiles = dir('D:\6th Sem\EE 657\8042 Assignment\Assignment_list\TestCharacters\TestCharacters\3\*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
    filename = strcat('D:\6th Sem\EE 657\8042 Assignment\Assignment_list\TestCharacters\TestCharacters\3\',srcFiles(i).name);
    J = imresize(imread(filename), 0.25);
    T1(:,i) = J(:);
end

for i = 1:100
    dist(1) = Distance(T1(:,i), mean1, cov1);
    dist(2) = Distance(T1(:,i), mean2, cov2);
    dist(3) = Distance(T1(:,i), mean3, cov3);
    [M, u] =   min(dist);
    result(i+200, 1) = u;
end

for i = 1:100
    dist(1) = Distance(T1(:,i), mean1, covar_2);
    dist(2) = Distance(T1(:,i), mean2, covar_2);
    dist(3) = Distance(T1(:,i), mean3, covar_2);
    [M, u] =   min(dist);
    result(i+200, 2) = u;
end

for i = 1:100
    dist(1) = Distance(T1(:,i), mean1, covar_3);
    dist(2) = Distance(T1(:,i), mean2, covar_3);
    dist(3) = Distance(T1(:,i), mean3, covar_3);
    [M, u] =   min(dist);
    result(i+200, 3) = u;
end

%% Calculating the accuracies

n1 = sum(result(1:100, 1) == 1);
n2 = sum(result(100:200, 1) == 2);
n3 = sum(result(200:300, 1) == 3);

fprintf('Individual accuracy for first folder: %0.2f \n', n1);
fprintf('Individual accuracy for second folder: %0.2f \n', n2);
fprintf('Individual accuracy for third folder: %0.2f \n', n3);

accuracy1 = (n1+n2+n3)/3;

n1 = sum(result(1:100, 2) == 1);
n2 = sum(result(100:200, 2) == 2);
n3 = sum(result(200:300, 2) == 3);

fprintf('Individual accuracy for first folder: %0.2f \n', n1);
fprintf('Individual accuracy for second folder: %0.2f \n', n2);
fprintf('Individual accuracy for third folder: %0.2f \n', n3);

accuracy2 = (n1+n2+n3)/3;

n1 = sum(result(1:100, 3) == 1);
n2 = sum(result(100:200, 3) == 2);
n3 = sum(result(200:300, 3) == 3);

fprintf('Individual accuracy for first folder: %0.2f \n', n1);
fprintf('Individual accuracy for second folder: %0.2f \n', n2);
fprintf('Individual accuracy for third folder: %0.2f \n', n3);

accuracy3 = (n1+n2+n3)/3;

fprintf('Average accuracy for Case-1: %0.2f \n', accuracy1);
fprintf('Average accuracy for Case-2: %0.2f \n', accuracy2);
fprintf('Average accuracy for Case-3: %0.2f \n', accuracy3);
