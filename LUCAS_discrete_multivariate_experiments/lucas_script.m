clear all; clc; close all;

%% load data
load('lucas0_train.mat');
theta_true = [0.23;0.87;0.84;0.99;0.4;0.8;0.35;0.8;0.57;0.90]; %approx true parameters

% Features:
S = X(:,1); %smoking (causal)
G = X(:,5); %genetics (causal)
C = X(:,11); % coughing (effect)
F = X(:,9); %fatigue (effect)

% Target variable
L = load('lucas0_train.targets'); %lung cancer
L = (L + 1) ./ 2; %change labels from -1/1 to 0/1

% Domain selector
A = X(:,3); %anxiety, a direct cause of smoking

% Source domain: no anxiety, target domain: anxiety
S_0 = S(A==0);
G_0 = G(A==0);
C_0 = C(A==0);
F_0 = F(A==0);
L_0 = L(A==0);

S_1 = S(A==1);
G_1 = G(A==1);
C_1 = C(A==1);
F_1 = F(A==1);
L_1 = L(A==1);

N_S = size(S_0,1);
N_T = size(S_1,1);

%% define simulation parameters
n_iter = 1000;

n_S = 2^4;
k = 2^3;
n_T_range = 2.^(0:1:8);
n_lambda = 10;

%lambda_range = linspace(0,1,n_lambda);
e_S = zeros(n_iter,1);
e_P = zeros(n_iter,size(n_T_range,2));
options = optimset('MaxFunEvals',10^9, 'MaxIter', 10^9);

%% run simulation
for i = 1:n_iter
    disp(['Iteration: ', num2str(i), ' / ',num2str(n_iter)])
    
    %% split data into training and testing
    r = randperm(N_S,n_S)';
    S_S = S_0(r);
    G_S = G_0(r);
    L_S = L_0(r);
    C_S = C_0(r);
    F_S = F_0(r);
    
    r = randperm(N_T)';
    te = r(1:1000);
    S_te = S_1(te);
    G_te = G_1(te);
    L_te = L_1(te);
    C_te = C_1(te);
    F_te = F_1(te);
    
    
    %% compute sufficient statistics
    % nLij = #(L=1,S=i,G=j); nLijc = #(L=0,S=i,G=j)
    nL00 = sum(ismember([S_S G_S L_S], [0 0 1], 'rows')); %a = 0.23
    nL00c = sum(ismember([S_S G_S L_S], [0 0 0], 'rows'));%(1-a)
    nL01 = sum(ismember([S_S G_S L_S], [0 1 1], 'rows'));%b = 0.87
    nL01c = sum(ismember([S_S G_S L_S], [0 1 0], 'rows'));%(1-b)
    nL10 = sum(ismember([S_S G_S L_S], [1 0 1], 'rows'));%c = 0.84
    nL10c = sum(ismember([S_S G_S L_S], [1 0 0], 'rows'));%(1-c)
    nL11 = sum(ismember([S_S G_S L_S], [1 1 1], 'rows'));%d = 0.99
    nL11c = sum(ismember([S_S G_S L_S], [1 1 0], 'rows'));%(1-d)
    if (nL00+nL00c+nL01+nL01c+nL10+nL10c+nL11+nL11c)~=size(L_S,1)
        error('wrong checksum')
    end
    
    % nCi = #(C=1,L=i); nCic = #(C=0,L=i)
    nC0 = sum(ismember([L_S C_S],[0 1], 'rows')); %0.13 < e < 0.65
    nC0c = sum(ismember([L_S C_S],[0 0], 'rows'));%(1-e)
    nC1 = sum(ismember([L_S C_S],[1 1], 'rows')); %0.77 < f < 1
    nC1c = sum(ismember([L_S C_S],[1 0], 'rows')); %(1-f)
    if (nC0+nC0c+nC1+nC1c)~=size(L_S,1)
        error('wrong checksum')
    end
    
    % nFij = #(F=1,L=i,C=j); nFijc = #(F=1,L=i,C=j)
    nF00 = sum(ismember([L_S C_S F_S], [0 0 1], 'rows')); %g = 0.35
    nF00c = sum(ismember([L_S C_S F_S], [0 0 0], 'rows')); %(1-g)
    nF01 = sum(ismember([L_S C_S F_S], [0 1 1], 'rows')); %h = 0.80
    nF01c = sum(ismember([L_S C_S F_S], [0 1 0], 'rows')); %(1-h)
    nF10 = sum(ismember([L_S C_S F_S], [1 0 1], 'rows')); %k = 0.57
    nF10c = sum(ismember([L_S C_S F_S], [1 0 0], 'rows')); %(1-k)
    nF11 = sum(ismember([L_S C_S F_S], [1 1 1], 'rows')); %m = 0.90
    nF11c = sum(ismember([L_S C_S F_S], [1 1 0], 'rows')); %(1-m)
    if (nF00+nF00c+nF01+nF01c+nF10+nF10c+nF11+nF11c)~=size(L_S,1)
        error('wrong checksum')
    end
    
    %% compute log-likelihood of labelled data
    theta_0 = 0.5*randn(10,1);    
    fun_S = @(theta) lucasNLL_S( theta, nL00, nL00c, nL01, nL01c,...
        nL10, nL10c, nL11, nL11c, nC0, nC0c, nC1, nC1c, nF00, nF00c, nF01,...
        nF01c, nF10, nF10c, nF11, nF11c );
    theta_S = fminsearch(fun_S,theta_0,options);
    e_S(i,1) = lucas_error(theta_S, S_te, G_te, C_te, F_te, L_te);
    
    for j = 1:size(n_T_range,2)
        n_T = n_T_range(j)
        
        t = r(1001 : (1000 + n_T));
        
        S_T = S_1(t);
        G_T = G_1(t);
        %L_T = L_1(t);
        C_T = C_1(t);
        F_T = F_1(t);
        
        lambda_fixed = n_S/(n_S+sqrt(n_T))     
%         lambda = leave_one_out_CV(S_S, G_S, L_S, C_S, F_S,...
%             S_T, G_T, C_T, F_T, n_lambda);
        lambda = k_fold_CV(S_S, G_S, L_S, C_S, F_S,...
             S_T, G_T, C_T, F_T, n_lambda, k)

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
              
        fun_P = @(theta) lambda * lucasNLL_S( theta, nL00, nL00c, nL01, nL01c,...
            nL10, nL10c, nL11, nL11c, nC0, nC0c, nC1, nC1c, nF00, nF00c, nF01,...
            nF01c, nF10, nF10c, nF11, nF11c ) +...
            (1 - lambda) * lucasNLL_T( theta, n0000, n0001, n0010, n0011, ...
            n0100, n0101, n0110, n0111, n1000, n1001, n1010, n1011, n1100, n1101,...
            n1110, n1111);
        theta_P = fminsearch(fun_P,theta_0,options);
        e_P(i,j) = lucas_error(theta_P, S_te, G_te, C_te, F_te, L_te);
    end
end

error_S = mean(e_S);
std_S = std(e_S);
error_P = mean(e_P);
std_P = std(e_P);

%%
% figure(1);
% errorbar(lambda_range, error_P, std_P)
% hold on
% plot(lambda_range, ones(1,n_lambda)*error_S)
% hold off
% xlabel('\lambda')
% ylabel('classification error')
% legend('\theta_P','\theta_S')


figure(2);
loglog(n_T_range, ones(size(n_T_range))*error_S)
hold on
errorbar(n_T_range, error_P, std_P,'-xr','LineWidth',1)
hold off
xlabel('n_T')
ylabel('classification error')
legend('\theta_S','\theta_P')
title(['n_S = ',num2str(n_S)])

%[sigm(theta_S) sigm(theta_P) theta_true]