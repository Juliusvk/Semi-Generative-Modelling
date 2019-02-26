function [mu_S,mu_T,mu_Y,mu_0,mu_1] = getParamsClass(misspecified)
mu_S = rand; %U[0,1]
mu_T = -mu_S; % -mu_T
mu_Y = 2*rand-1; %U[-1,1]
if misspecified == false
    mu_0 = -rand; %U[0,1]
    mu_1 = rand;   % - mu_0
else
    mu_0 = 0;
    mu_1 = 3;
end

end

