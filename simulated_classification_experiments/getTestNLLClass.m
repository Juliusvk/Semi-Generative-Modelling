function TestNLL = getTestNLLClass(theta, D)

mu_Y = theta(1);
mu_0 = theta(2);
mu_1 = theta(3);
sigma_Y = 1;
sigma_0 = 1;
sigma_1 = 1;
X_C = D(1,:);
Y = D(2,:);
X_E = D(3,:);
l_EY = Y .* log( normpdf(X_E, mu_1, sigma_1) ) + ...
    (1 - Y) .* log( normpdf(X_E, mu_0, sigma_0) ); %l(X_E|Y)
l_YC = - log( 1 + exp(- sigma_Y * (X_C - mu_Y) ) ) - ...
    (1 - Y) * sigma_Y .* (X_C - mu_Y); %l(Y|X_C)
TestNLL = -mean( l_EY + l_YC );

end

