function nll = nll_YGivenXC(X_C,Y,w,theta)
mu_Y=theta(1);
l_YC = - log( 1+exp(-(X_C-mu_Y)) )   -   (1-Y) .* (X_C-mu_Y); 
nll = -mean(w .* l_YC);
end

