function [ classprobs ] = getClassProbs( D, theta )
X_C = D(1,:);       X_E = D(3,:);
mu_Y = theta(1);    mu_0 = theta(2);    mu_1 = theta(3);
sigma_Y = 1;        sigma_0 = 1;        sigma_1 = 1;
classprobs = 1 ./ ( 1 + (sigma_1 / sigma_0) * exp( - 0.5 * ( ...
    ((X_E - mu_0)/sigma_0).^2 - ((X_E - mu_1)/sigma_1).^2 ) - ...
    sigma_Y * (X_C - mu_Y) ) );
end

