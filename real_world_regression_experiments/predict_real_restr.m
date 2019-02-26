function Y = predict_real_restr(X_Ctest,X_Etest,theta)
a = theta(1);
b = -exp(theta(2));
c = theta(3);
d = -exp(theta(4));
sigma_Y = 1;
sigma_E = 1;

if d~=0
    Y = ( sigma_E * (a + b * X_Ctest) + ...
    d^2 * sigma_Y * ((X_Etest - c)./d) )...
    ./ (sigma_E + d^2 * sigma_Y);
else
    Y = (a + b * X_Ctest);
end

end