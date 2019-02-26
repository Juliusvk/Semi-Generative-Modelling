function N = pooled_nll_real_sigma(X_CS,Y_S,X_ES,X_CT,X_ET,lambda,theta)
a = theta(1);
b = theta(2);
c = theta(3);
d = theta(4);
sigma_Y = exp(theta(5));
sigma_E = exp(theta(6));

N_Y = 0.5 * log(sigma_Y) +...
    (1/(2*sigma_Y)) * mean((Y_S - a - b * X_CS).^2) ;
N_E = 0.5 * log(sigma_E) +...
(1/(2*sigma_E)) * mean((X_ES - c - d * Y_S).^2);
N_sup = N_Y + N_E;

N_unsup = 0.5 * ( log(sigma_Y)+log(sigma_E)-log(d^2*sigma_Y + sigma_E) )...
    + (0.5/(sigma_E + d^2 * sigma_Y)) * ...
    mean((X_ET - c - a * d - b * d * X_CT).^2 );

N = lambda * N_sup + (1 - lambda) * N_unsup;

end