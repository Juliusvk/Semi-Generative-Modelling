function nll = nll_Pooled(X_CS,Y_S,X_ES,X_CT,X_ET,lambda,theta)
mu_Y = theta(1);
mu_0 = theta(2);
mu_1 = theta(3);

nll_YC = log( 1+exp(-(X_CS-mu_Y)) )  +  (1-Y_S) .* (X_CS-mu_Y);
nll_EY = Y_S .* (0.5 * (X_ES-mu_1).^2) + (1-Y_S) .* (0.5 * (X_ES-mu_0).^2);
nll_EC = log(1 + exp(-(X_CT - mu_Y))) - ...
    log( normpdf(X_ET,mu_1,1) + exp(-(X_CT-mu_Y)).*normpdf(X_ET,mu_0,1) );
nll = lambda * mean(nll_YC + nll_EY) + (1-lambda) * mean(nll_EC); 

end

