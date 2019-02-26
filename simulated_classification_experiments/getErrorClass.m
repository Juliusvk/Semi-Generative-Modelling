function E = getErrorClass(theta, D)

X_C = D(1,:);
Y = D(2,:);
X_E = D(3,:);
n = size(D,2);
mu_Y = theta(1);
mu_0 = theta(2);
mu_1 = theta(3);
sigma_Y = 1;
sigma_0 = 1; 
sigma_1 = 1;
prob = 1 ./ ( 1 + (sigma_1 / sigma_0) * exp( - 0.5 * ( ...
    ((X_E - mu_0)/sigma_0).^2 - ((X_E - mu_1)/sigma_1).^2 ) - ...
    sigma_Y * (X_C - mu_Y) ) );
fitted = (prob > 0.5);
correct = (fitted == Y);
E = 1 - mean(correct);

end

