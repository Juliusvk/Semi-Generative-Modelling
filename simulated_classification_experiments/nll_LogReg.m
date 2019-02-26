function nll = nll_LogReg(X_C, Y, X_E, theta)
n = size(X_C,2);
X = [ones(1,n); X_C; X_E];
nll = mean( log( 1+exp(-theta*X) ) + (1-Y) .* (theta*X) );
end

