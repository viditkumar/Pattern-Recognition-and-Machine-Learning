%% Ques-2
%%%%% Implement a GMM based clustering framework on the image ‘ski_image.jpg’. You may
%%%%% choose 3 Gaussian components and use the RGB values as the features.

clear
clc
close all
format long

%% Reading the image

image = imread('ski_image.jpg');
[l , w, h] = size(image);
R = image(: , : , 1);
G = image(: , : , 2);
B = image(: , : , 3);

d = l*w;

R = reshape(R , [d , 1]);
G = reshape(G , [d , 1]);
B = reshape(B , [d , 1]);
X = [R , G , B];

X = double(X)/255; % Making the values in the matrix as 'double'


%% Initializing the means

mean1 = double([120 , 120 , 120])/255;
mean2 = double([12 , 12 , 12])/255;
mean3 = double([180 , 180 , 180])/255;
mean = [mean1 ; mean2 ; mean3];

%% Initializing the covariance matrices

cov1 = eye(3 , 3);
cov2 = eye(3 , 3);
cov3 = eye(3 , 3);

%% Initializiing the weights

wt1 = 1/3;
wt2 = 1/3;
wt3 = 1/3;

%% 

m = 30;
LOG = zeros(1 , m); %Initializing the Log likelihood function
for i = 1:m
    fprint('Iteration number in process: %d\n', i);
    
    for j = 1:d
        data = X(j , :);
        n1 = (1/((2*pi)^(1.5) * (det(cov1))^0.5))*exp(-1/2*((data - mean1) * inv(cov1) * (data - mean1)'));
        n2 = (1/((2*pi)^(1.5) * (det(cov2))^0.5))*exp(-1/2*((data - mean2) * inv(cov2) * (data - mean2)'));
        n3 = (1/((2*pi)^(1.5) * (det(cov3))^0.5))*exp(-1/2*((data - mean3) * inv(cov3) * (data - mean3)'));
        sumX = wt1 * n1 + wt2 * n2 + wt3 * n3;
        
        % Calculating the responsibilities
        res(j , 1) = wt1 * n1 / sumX;
        res(j , 2) = wt2 * n2 / sumX;
        res(j , 3) = wt3 * n3 / sumX;
    end
    
    N1 = 0;
    N2 = 0;
    N3 = 0;
    
    for j = 1:d
        N1 = N1 + res(j , 1);
        N2 = N2 + res(j , 2);
        N3 = N3 + res(j , 3);
    end
    
    sum1 = 0;
    sum2 = 0;
    sum3 = 0;
    
    for j = 1:d
        sum1 = sum1 + res(j , 1)*X(j, :);
        sum2 = sum2 + res(j , 2)*X(j, :);
        sum3 = sum3 + res(j , 3)*X(j, :);
    end
    
    % Updating the means
    mean1 = sum1/N1;
    mean2 = sum2/N2;
    mean3 = sum3/N3;
    
    sig1 = zeros(3 , 3);
    sig2 = zeros(3 , 3);
    sig3 = zeros(3 , 3);
    
    for j = 1:d
        sig1=sig1+res(j ,1)*(X(j , :)-mean1)'*(X(j , :)-mean1);
        sig2=sig2+res(j ,2)*(X(j , :)-mean2)'*(X(j , :)-mean2);
        sig3=sig3+res(j ,3)*(X(j , :)-mean3)'*(X(j , :)-mean3);
    end
    
    % Updating the covariance matrices
    cov1 = sig1/N1;
    cov2 = sig2/N2;
    cov3 = sig3/N3;
    
    % Updating the weights
    wt1 = N1/d;
    wt2 = N2/d;
    wt3 = N3/d;
    
    for j = 1:d
        data = X(j , :);
        n1 = (1/((2*pi)^(1.5) * (det(cov1))^0.5))*exp(-1/2*((data - mean1) * inv(cov1) * (data - mean1)'));
        n2 = (1/((2*pi)^(1.5) * (det(cov2))^0.5))*exp(-1/2*((data - mean2) * inv(cov2) * (data - mean2)'));
        n3 = (1/((2*pi)^(1.5) * (det(cov3))^0.5))*exp(-1/2*((data - mean3) * inv(cov3) * (data - mean3)'));
        sumX = wt1 * n1 + wt2 * n2 + wt3 * n3; 
        LOG(i) = LOG(i) + log(sumX);
    end   
end

%% Segmented Image

img = res;
seg = reshape(img , 321 , 481 , 3);
seg = uint8(seg);
figure;
imshow(seg);





