function Y = predict_real_sigma(X_Ctest,X_Etest,theta)
a = theta(1);
b = theta(2);
c = theta(3);
d = theta(4);
sigma_Y = exp(theta(5));
sigma_E = exp(theta(6));

if d~=0
    Y = ( sigma_E * (a + b * X_Ctest) + ...
    d^2 * sigma_Y * ((X_Etest - c)./d) )...
    ./ (sigma_E + d^2 * sigma_Y);
else
    Y = (a + b * X_Ctest);
end

end