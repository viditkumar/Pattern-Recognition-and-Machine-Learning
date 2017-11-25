function [ dist ] = Distance(X , mean , cov)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
dist = (X - mean)'*inv(cov)*(X-mean);
end

