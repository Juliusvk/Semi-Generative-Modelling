function [ D_S, D_T ] = getDataClass( params )
% This function generates a synthetic dataset for domain adaptation 
% according to the causal DAG:    D --> X_C --> Y --> X_E
% output: source (D_S) and target (D_T) data in the format [X_C; Y; X_E],
% i.e. (3 x n_S) and (3 x n_T respectively)

% number of source and target samples
n_S = params(1); n_T = params(2);

% source and target marginals for X_C - Normal distributions
mu_S = params(3); sigma_S = params(4);
mu_T = params(5); sigma_T = params(6);

% conditional for Y given X_C - logistic( sigma_Y * ( x - mu_Y ) )
mu_Y = params(7); sigma_Y = params(8); 

% conditional for X_E given Y
mu_0 = params(9); sigma_0 = params(10);
mu_1 = params(11); sigma_1 = params(12);
mixture = params(13);

% Draw X_C (Normally distributed with domain-dependent mean and std.):
X_CS = mu_S + sigma_S * randn(1,n_S); X_CT = mu_T + sigma_T * randn(1,n_T);

% Draw Y (binary classification with logistic conditional Y|X_C):
p_YS = 1./( 1 + exp(-sigma_Y * (X_CS - mu_Y)) ); %P(Y=1|X_CS)
p_YT = 1./( 1 + exp(-sigma_Y * (X_CT - mu_Y)) ); %P(Y=1|X_CT)
Y_S = double(p_YS > rand(1,n_S)); Y_T = double(p_YT > rand(1,n_T));

% Draw X_E (Normally distributed with class-dependent mean and std.):
X_ES = zeros(1,n_S); X_ET = zeros(1,n_T);
mog = gmdistribution([-mu_1; mu_1], sigma_1, [0.5, 0.5]);
r_S = randn(1,n_S); r_T = randn(1,n_T);
for i = 1 : n_S
    if Y_S(1,i) == 0
        X_ES(1,i) = mu_0 + sigma_0 * r_S(1,i);
    elseif Y_S(1,i) == 1 && mixture == false
        X_ES(1,i) = mu_1 + sigma_1 * r_S(1,i);
    elseif Y_S(1,i) == 1 && mixture == true
        X_ES(1,i) = random(mog);
    end
end
for i = 1 : n_T
    if Y_T(1,i) == 0
        X_ET(1,i) = mu_0 + sigma_0 * r_T(1,i);
    elseif Y_T(1,i) == 1 && mixture == false
        X_ET(1,i) = mu_1 + sigma_1 * r_T(1,i);
    elseif Y_T(1,i) == 1 && mixture == true
        X_ET(1,i) = random(mog);
    end
end

D_S = [X_CS; Y_S; X_ES];
D_T = [X_CT; Y_T; X_ET];

end

