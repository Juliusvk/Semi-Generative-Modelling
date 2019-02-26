function N = sup_nll_real(X_CS,Y_S,X_ES,theta)
a = theta(1);
b = theta(2);
c = theta(3);
d = theta(4);
sigma_Y = 1;
sigma_E = 1;

N_Y = 0.5 * log(sigma_Y) +...
    (1/(2*sigma_Y)) .* mean((Y_S - a - b * X_CS).^2) ;
N_E = 0.5 * log(sigma_E) +...
(1/(2*sigma_E)) .* mean((X_ES - c - d * Y_S).^2) ;

N = N_Y+N_E;

end

