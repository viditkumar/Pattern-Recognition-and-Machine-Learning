%% Ques 3
%%% Face Recognition using PCA

clc;
clear;
close all;
format long;

%% Initialization

h = 112;
w = 92;
d = h*w;
n = 200;

A = zeros(d, n);

%% Reading the images

k = 0;
for i = 1:40
    for j = 1:5
        image = imread(strcat('D:\6th Sem\EE 657\FInal Folder\gallery\s',int2str(i),'\',int2str(j),'.pgm'));
        image = double(image);
        image = reshape(image,[],1);
        A(:, k+j) = image;
    end
    k = k + 5;    
end

mu = mean(A')';

L = zeros(d, n);

for i = 1:n
    L(:,i)=(A(:,i) - mu);
end

L = (L'*L)/200;

[Vecs, Vals] = eig(L);
B = diag(Vals);
Vecs = A*Vecs;

colormap gray;

for i=1:5
    subplot(2, 3, i);
    imagesc(reshape(Vecs(:,201-i),112,92));
end

%% Plot a graph depicting the percentage of the total variance of the original data
%%% retained in the reduced space versus the number of dimensions

Vals1 = sort(B, 'descend');
tot_var = sum(abs(Vals1(:)));
z = 1:length(Vals1);
flag = 0;
for i = 1:length(Vals1)
    ratio1(i) = (sum(abs(Vals1(1:i))))/tot_var;
    if(ratio1(i) > 0.95 && (flag == 0))
        reqd_dim = i;
        flag = 1;
    end
end

figure(6), plot(z, ratio1)
hline = refline([0 ratio1(reqd_dim)]);
% hline.Color = 'r';
fprintf('Required dimension so that at least 0.95 of the total variance of the original data is accounted: %d \n', reqd_dim);

%% Reconstruct the image ‘face_input_1.pgm’ using the:
% (a)Eigenface corresponding to the largest eigenvalue:

m = 1;
OrigImg1 = imread(strcat('D:\6th Sem\EE 657\FInal Folder\face_input_1.pgm'));
OrigImg1 = double(OrigImg1);
OrigImg1 = reshape(OrigImg1,[],1);
OrigImg1 = OrigImg1 - mu;
 
u = (Vecs(:,201-m))/norm((Vecs(:,201-m)));
Projection = (OrigImg1'*(u))*(u);
Reconstructed = Projection + mu;
figure(7);
colormap gray;
imagesc(reshape(Reconstructed, h, w));

error1 =  sum((OrigImg1 - Reconstructed).^2)/((norm(OrigImg1))^2);
fprintf('Error in (a): %0.02f \n', error1/100);

%(b)Top 15 Eigenfaces:

m = 15;

for i = 2:m
    u=(Vecs(:,200-i))/norm((Vecs(:,200-i)));
    Projection = (OrigImg1'*(u));
    Reconstructed = Reconstructed + Projection*(u);  
end

figure(8);
colormap gray;
imagesc(reshape(Reconstructed, h, w));

error2 =  sum((OrigImg1 - Reconstructed).^2)/((norm(OrigImg1))^2);
fprintf('Error in (b): %0.02f \n', error2/100);

%(c)All the Eigenfaces:

m = 199;

for i = 16:m-1
    u = (Vecs(:,200-i))/norm((Vecs(:,200-i)));
    Projection = (OrigImg1'*(u));
    Reconstructed = Reconstructed + Projection*(u);  
end

figure(9);
colormap gray;
imagesc(reshape(Reconstructed, h, w));

error3 =  sum((OrigImg1 - Reconstructed).^2)/((norm(OrigImg1))^2);
fprintf('Error in (c): %0.02f \n', error3/100);

% Depict graphically the mean squared error obtained for different number of Eigen-faces

OrigImg1 = imread(strcat('D:\6th Sem\EE 657\FInal Folder\face_input_1.pgm'));
OrigImg1 = double(OrigImg1);
OrigImg1 = reshape(OrigImg1,[],1);
OrigImg1 = OrigImg1 - mu;

for m = 1:200
    u = (Vecs(:,201-m))/norm((Vecs(:,201-m)));
    Projection = (OrigImg1'*(u))*(u);
    Reconstructed = Projection + mu;
    error_plot1(m) =  sum((OrigImg1 - Reconstructed).^2)/((norm(OrigImg1))^2);
end
figure(14);
plot(error_plot1)

%% Reconstruct the image ‘face_input_2.pgm’ using the:
% (a)Eigenface corresponding to the largest eigenvalue:

m = 1;
OrigImg2 = imread(strcat('D:\6th Sem\EE 657\FInal Folder\face_input_2.pgm'));
OrigImg2 = double(OrigImg2);
OrigImg2 = reshape(OrigImg2,[],1);
OrigImg2 = OrigImg2 - mu;
 
u = (Vecs(:,201-m))/norm((Vecs(:,201-m)));
Projection = (OrigImg1'*(u))*(u);
Reconstructed = Projection + mu;
figure(10);
colormap gray;
imagesc(reshape(Reconstructed, h, w));

error1 =  sum((OrigImg2 - Reconstructed).^2)/((norm(OrigImg2))^2);
fprintf('Error in (a): %0.02f \n', error1/100);

%(b)Top 15 Eigenfaces:

m = 15;

for i = 2:m
    u=(Vecs(:,200-i))/norm((Vecs(:,200-i)));
    Projection = (OrigImg2'*(u));
    Reconstructed = Reconstructed + Projection*(u);  
end

figure(11);
colormap gray;
imagesc(reshape(Reconstructed, h, w));

error2 =  sum((OrigImg2 - Reconstructed).^2)/((norm(OrigImg2))^2);
fprintf('Error in (b): %0.02f \n', error2/100);

%(c)All the Eigenfaces:

m = 199;

for i = 16:m-1
    u = (Vecs(:,200-i))/norm((Vecs(:,200-i)));
    Projection = (OrigImg2'*(u));
    Reconstructed = Reconstructed + Projection*(u);  
end

figure(12);
colormap gray;
imagesc(reshape(Reconstructed, h, w));

error3 =  sum((OrigImg2 - Reconstructed).^2)/((norm(OrigImg2))^2);
fprintf('Error in (c): %0.02f \n', error3/100);

% Depict graphically the mean squared error obtained for different number of Eigen-faces

OrigImg2 = imread(strcat('D:\6th Sem\EE 657\FInal Folder\face_input_2.pgm'));
OrigImg2 = double(OrigImg2);
OrigImg2 = reshape(OrigImg2,[],1);
OrigImg2 = OrigImg2 - mu;

for m = 1:200
    u = (Vecs(:,201-m))/norm((Vecs(:,201-m)));
    Projection = (OrigImg2'*(u))*(u);
    Reconstructed = Projection + mu;
    error_plot2(m) =  sum((OrigImg2 - Reconstructed).^2)/((norm(OrigImg2))^2);
end
figure(15);
plot(error_plot2)
