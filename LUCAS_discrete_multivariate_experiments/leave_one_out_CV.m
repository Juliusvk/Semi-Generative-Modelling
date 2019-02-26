function lambda = leave_one_out_CV(S_S, G_S, L_S, C_S, F_S,...
    S_T, G_T, C_T, F_T, n_lambda)

% performs leave one out cross validation to find the best lambda
lambda_range = exp(linspace(0,-2,n_lambda));
val_error = zeros(n_lambda,1);

n_S = size(S_S,1);
options = optimset('MaxFunEvals',10^9, 'MaxIter', 10^9);

% compute sufficient statistics of unlabelled data
n0000 = sum(ismember([S_T G_T C_T F_T],[0 0 0 0],'rows'));
n0001 = sum(ismember([S_T G_T C_T F_T],[0 0 0 1],'rows'));
n0010 = sum(ismember([S_T G_T C_T F_T],[0 0 1 0],'rows'));
n0011 = sum(ismember([S_T G_T C_T F_T],[0 0 1 1],'rows'));
n0100 = sum(ismember([S_T G_T C_T F_T],[0 1 0 0],'rows'));
n0101 = sum(ismember([S_T G_T C_T F_T],[0 1 0 1],'rows'));
n0110 = sum(ismember([S_T G_T C_T F_T],[0 1 1 0],'rows'));
n0111 = sum(ismember([S_T G_T C_T F_T],[0 1 1 1],'rows'));
n1000 = sum(ismember([S_T G_T C_T F_T],[1 0 0 0],'rows'));
n1001 = sum(ismember([S_T G_T C_T F_T],[1 0 0 1],'rows'));
n1010 = sum(ismember([S_T G_T C_T F_T],[1 0 1 0],'rows'));
n1011 = sum(ismember([S_T G_T C_T F_T],[1 0 1 1],'rows'));
n1100 = sum(ismember([S_T G_T C_T F_T],[1 1 0 0],'rows'));
n1101 = sum(ismember([S_T G_T C_T F_T],[1 1 0 1],'rows'));
n1110 = sum(ismember([S_T G_T C_T F_T],[1 1 1 0],'rows'));
n1111 = sum(ismember([S_T G_T C_T F_T],[1 1 1 1],'rows'));
if n0000+n0001+n0010+n0011+n0100+n0101+n0110+n0111+...
        n1000+n1001+n1010+n1011+n1100+n1101+n1110+n1111~=size(S_T,1)
    error('wrong checksum')
end

%iterate through train/val splits
for i=1:n_S
    theta_0 = 0.5*randn(10,1);
    % use i^th data point for validation, rest for training
    S_train = S_S;    G_train = G_S;    L_train = L_S;
    C_train = C_S;    F_train = F_S;
    S_val = S_S(i,1);    G_val = G_S(i,1);    L_val = L_S(i,1);
    C_val = C_S(i,1);    F_val = F_S(i,1);
    S_train(i) = [];    G_train(i) = [];    L_train(i) = [];
    C_train(i) = [];    F_train(i) = [];
    
    % compute sufficient statistics of labelled data
    % nLij = #(L=1,S=i,G=j); nLijc = #(L=0,S=i,G=j)
    nL00 = sum(ismember([S_train G_train L_train], [0 0 1], 'rows')); %a = 0.23
    nL00c = sum(ismember([S_train G_train L_train], [0 0 0], 'rows'));%(1-a)
    nL01 = sum(ismember([S_train G_train L_train], [0 1 1], 'rows'));%b = 0.87
    nL01c = sum(ismember([S_train G_train L_train], [0 1 0], 'rows'));%(1-b)
    nL10 = sum(ismember([S_train G_train L_train], [1 0 1], 'rows'));%c = 0.84
    nL10c = sum(ismember([S_train G_train L_train], [1 0 0], 'rows'));%(1-c)
    nL11 = sum(ismember([S_train G_train L_train], [1 1 1], 'rows'));%d = 0.99
    nL11c = sum(ismember([S_train G_train L_train], [1 1 0], 'rows'));%(1-d)
    if (nL00+nL00c+nL01+nL01c+nL10+nL10c+nL11+nL11c)~=size(L_train,1)
        error('wrong checksum')
    end
    
    % nCi = #(C=1,L=i); nCic = #(C=0,L=i)
    nC0 = sum(ismember([L_train C_train],[0 1], 'rows')); %0.13 < e < 0.65
    nC0c = sum(ismember([L_train C_train],[0 0], 'rows'));%(1-e)
    nC1 = sum(ismember([L_train C_train],[1 1], 'rows')); %0.77 < f < 1
    nC1c = sum(ismember([L_train C_train],[1 0], 'rows')); %(1-f)
    if (nC0+nC0c+nC1+nC1c)~=size(L_train,1)
        error('wrong checksum')
    end
    
    % nFij = #(F=1,L=i,C=j); nFijc = #(F=1,L=i,C=j)
    nF00 = sum(ismember([L_train C_train F_train], [0 0 1], 'rows')); %g = 0.35
    nF00c = sum(ismember([L_train C_train F_train], [0 0 0], 'rows')); %(1-g)
    nF01 = sum(ismember([L_train C_train F_train], [0 1 1], 'rows')); %h = 0.80
    nF01c = sum(ismember([L_train C_train F_train], [0 1 0], 'rows')); %(1-h)
    nF10 = sum(ismember([L_train C_train F_train], [1 0 1], 'rows')); %k = 0.57
    nF10c = sum(ismember([L_train C_train F_train], [1 0 0], 'rows')); %(1-k)
    nF11 = sum(ismember([L_train C_train F_train], [1 1 1], 'rows')); %m = 0.90
    nF11c = sum(ismember([L_train C_train F_train], [1 1 0], 'rows')); %(1-m)
    if (nF00+nF00c+nF01+nF01c+nF10+nF10c+nF11+nF11c)~=size(L_train,1)
        error('wrong checksum')
    end
    
    %iterate through lambda choices
    for j=1:n_lambda
        lambda = lambda_range(j);
        fun_P = @(theta) lambda * lucasNLL_S( theta, nL00, nL00c, nL01, nL01c,...
            nL10, nL10c, nL11, nL11c, nC0, nC0c, nC1, nC1c, nF00, nF00c, nF01,...
            nF01c, nF10, nF10c, nF11, nF11c ) +...
            (1 - lambda) * lucasNLL_T( theta, n0000, n0001, n0010, n0011, ...
            n0100, n0101, n0110, n0111, n1000, n1001, n1010, n1011, n1100, n1101,...
            n1110, n1111);
        theta_P = fminsearch(fun_P,theta_0,options);
        val_error(j) = val_error(j) + lucas_error(theta_P,...
            S_val, G_val, C_val, F_val, L_val);
    end
end

[~, idx] = min(val_error);
lambda = lambda_range(idx);

end

