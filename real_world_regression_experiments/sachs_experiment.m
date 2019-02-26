clear all; close all; clc;

%% LOAD DATA
%D1: MEK->ERK->AKT with intervention on MEK
source = readtable('Sachs.SOM.Datasets/Data Files/2. cd3cd28icam2.xls');
target = readtable('Sachs.SOM.Datasets/Data Files/13. cd3cd28icam2+u0126.xls');
D1S = [source.pmek, source.p44_42, source.pakts473]';
D1T = [target.pmek, target.p44_42, target.pakts473]';
% D2: PKC->PKA->AKT with intervention on PKC
source = readtable('Sachs.SOM.Datasets/Data Files/1. cd3cd28.xls');
target = readtable('Sachs.SOM.Datasets/Data Files/4. cd3cd28+g0076.xls');
D2S = [source.PKC, source.PKA, source.pakts473]';
D2T = [target.PKC, target.PKA, target.pakts473]';

%% ASSIGN CAUSE X_C, OUTCOME Y, AND EFFECT X_E 
dataset = 2; %1: MEK->ERK->AKT, 3: PKC->PKA->AKT
switch dataset
    case 1
        X_CS = D1S(1,:); Y_S = D1S(2,:); X_ES = D1S(3,:);
        X_CT = D1T(1,:); Y_T = D1T(2,:); X_ET = D1T(3,:);
    case 2
        X_CS = D2S(1,:); Y_S = D2S(2,:); X_ES = D2S(3,:);
        X_CT = D2T(1,:); Y_T = D2T(2,:); X_ET = D2T(3,:);
end
n_source = size(X_CS,2); n_target = size(X_CT,2);
        
%% TRANSFORM TO LOG-SCALE
X_CS = log(X_CS);   Y_S = log(Y_S); X_ES = log(X_ES);
X_CT = log(X_CT);   Y_T = log(Y_T); X_ET = log(X_ET);
        
%% SET SIMULATION PARAMETERS
n_iter = 10000; %no of runs
n_S = 2^4; % amount of labelled source data
n_T_range = 2.^(1:1:9); N_T = size(n_T_range,2); % unlabelled target data
n_test = 200; %test set size
lambda = 0.8; %interpolation: weight of sup. component
% lambda = 1-1/n_S;
options = optimset('MaxFunEvals',10^9, 'MaxIter',10^9);

model_sigma = 1; %0 means sigma's are not modelled, 1 means they are
if model_sigma == 0,    theta_0 = zeros(4,1);
else,   theta_0 = zeros(6,1); end
model_restr = 1; %0: no restriction, 1: neg. slope for inverse relation
model_ftTrans = 1; %0: no comparison, 1: compare with TCA, MIDA, ...

par_TCA.isRegress = 1; par_TCA.kerName = 'lin'; par_TCA.bSstca = true;      
par_TCA.m = 2; par_TCA.mu = 1; par_TCA.gamma = .1; par_TCA.lambda = 0;
par_MIDA.isRegress = 1; par_MIDA.kerName = 'lin'; par_MIDA.bSmida = true;
par_MIDA.m = 2; par_MIDA.kerSigma = 1e-1;
par_SA.pcaCoef = 2;
par_GFK.dr = 1;

RMSE_S=zeros(n_iter,1); RMSE_P=zeros(n_iter,N_T);RMSE_LR=zeros(n_iter,1);
RMSE_TCA=zeros(n_iter,N_T); RMSE_MIDA = zeros(n_iter,N_T);
RMSE_GFK = zeros(n_iter,N_T); RMSE_SA = zeros(n_iter,N_T);
NegLogLik_S = zeros(n_iter,1); NegLogLik_P = zeros(n_iter,N_T);

%% SIMULATE
for i=1:n_iter
    disp(['Iteration: ', num2str(i), ' / ',num2str(n_iter)])
    s_idx = randperm(n_source,n_S); %draw n_S labelled source samples
    x_CS = X_CS(s_idx);     y_S = Y_S(s_idx);   x_ES = X_ES(s_idx);
    test_idx = randperm(n_target, n_test); %draw test set from target dom.
    x_Ctest = X_CT(test_idx);y_test = Y_T(test_idx);x_Etest = X_ET(test_idx);
    targ_idx = setdiff(1:n_target,test_idx); %remaining target data for SSL
    
    % theta_S
    if model_sigma == 0 || model_restr == 0
        sup_nll = @(theta) sup_nll_real(x_CS,y_S,x_ES,theta);
        theta_S = fminsearch(sup_nll, theta_0, options);
        predicted_S = predict_real(x_Ctest, x_Etest, theta_S);
        NegLogLik_S(i) = sup_nll_real(x_Ctest, y_test, x_Etest, theta_S);      
    elseif model_sigma == 0 || model_restr == 1
        sup_nll = @(theta) sup_nll_real_restr(x_CS,y_S,x_ES,theta);
        theta_S = fminsearch(sup_nll, theta_0, options);
        predicted_S = predict_real_restr(x_Ctest, x_Etest, theta_S);
        NegLogLik_S(i)=sup_nll_real_restr(x_Ctest,y_test,x_Etest,theta_S);      
    elseif model_sigma == 1 || model_restr == 0
        sup_nll = @(theta) sup_nll_real_sigma(x_CS,y_S,x_ES,theta);
        theta_S = fminsearch(sup_nll, theta_0, options); 
        predicted_S = predict_real_sigma(x_Ctest, x_Etest, theta_S); 
        NegLogLik_S(i)=sup_nll_real_sigma(x_Ctest,y_test,x_Etest,theta_S);      
    elseif model_sigma == 1 || model_restr == 1
        sup_nll = @(theta) sup_nll_real_restr_sigma(x_CS,y_S,x_ES,theta);
        theta_S = fminsearch(sup_nll, theta_0, options); 
        predicted_S = predict_real_restr_sigma(x_Ctest, x_Etest, theta_S);
        NegLogLik_S(i) = sup_nll_real_restr_sigma(...
            x_Ctest,y_test,x_Etest,theta_S);      
    end
    RMSE_S(i) = sqrt( mean( (y_test - predicted_S).^2 ) );
    
    % theta_LR
    X_LR = [ones(n_S,1), x_CS', x_ES'];
    theta_LR = (X_LR' * X_LR) \ X_LR' * y_S'; %least sq. sol.
    X_LR_test = [ones(n_test,1), x_Ctest', x_Etest'];
    predicted_LR = (X_LR_test * theta_LR)';
    RMSE_LR(i) = sqrt( mean( (y_test - predicted_LR).^2 ) );
    
    for j = 1:N_T
        n_T = n_T_range(j);
        lambda = n_S/(n_S+n_T);
        t_idx = targ_idx(randperm(size(targ_idx,2),n_T)); %draw targ. data
        x_CT = X_CT(t_idx);     y_T = Y_T(t_idx);   x_ET = X_ET(t_idx);
        
        % theta_P: our proposed method
        if model_sigma == 0 || model_restr == 0
            pooled_nll = @(theta) pooled_nll_real(...
                x_CS,y_S,x_ES,x_CT,x_ET,lambda,theta); 
            theta_P = fminsearch(pooled_nll, theta_0, options); 
            predicted_P = predict_real(x_Ctest, x_Etest, theta_P); 
            NegLogLik_P(i,j) = sup_nll_real(...
                x_Ctest, y_test, x_Etest, theta_P);
        elseif model_sigma == 0 || model_restr == 1
            pooled_nll = @(theta) pooled_nll_real_restr(...
                x_CS,y_S,x_ES,x_CT,x_ET,lambda,theta); 
            theta_P = fminsearch(pooled_nll, theta_0, options); 
            predicted_P = predict_real_restr(x_Ctest, x_Etest, theta_P); 
            NegLogLik_P(i,j) = sup_nll_real_restr(...
                x_Ctest, y_test, x_Etest, theta_P);
        elseif model_sigma == 1 || model_restr == 0
            pooled_nll = @(theta) pooled_nll_real_sigma(...
                x_CS,y_S,x_ES,x_CT,x_ET,lambda,theta); 
            theta_P = fminsearch(pooled_nll, theta_0, options); 
            predicted_P = predict_real_sigma(x_Ctest, x_Etest, theta_P); 
            NegLogLik_P(i,j) = sup_nll_real_sigma(...
                x_Ctest, y_test, x_Etest, theta_P);
        elseif model_sigma == 1 || model_restr == 1
            pooled_nll = @(theta) pooled_nll_real_restr_sigma(...
                x_CS,y_S,x_ES,x_CT,x_ET,lambda,theta); 
            theta_P = fminsearch(pooled_nll, theta_0, options); 
            predicted_P = predict_real_restr_sigma(x_Ctest,x_Etest,theta_P); 
            NegLogLik_P(i,j) = sup_nll_real_restr_sigma(...
                x_Ctest, y_test, x_Etest, theta_P);
        end
        RMSE_P(i,j) = sqrt( mean( (y_test - predicted_P).^2 ) );% evaluate
        
        %TCA, MIDA, SA, GFK
        if model_ftTrans == 1
            ft = [x_CS', x_ES'; x_CT', x_ET'];
            d_s = [true(n_S,1); false(n_T,1)];
            
            % TCA with linear kernel
            [ft_TCA,model_TCA] = ftTrans_tca(ft,d_s,y_S',d_s,par_TCA);
            X_TCA = [ones(n_S,1), ft_TCA(1:n_S,:)]; %transf source feat (n_Sx3)
            theta_TCA = (X_TCA' * X_TCA) \ X_TCA' * y_S'; %least sq. sol.
            W_TCA = ft' * model_TCA.W; %
            X_TCA_test = [ones(n_test,1), [x_Ctest', x_Etest'] * W_TCA];
            predicted_TCA = (X_TCA_test * theta_TCA)';
            RMSE_TCA(i,j) = sqrt( mean( (y_test - predicted_TCA).^2 ) );
            
            %MIDA with linear kernel
            [ft_MIDA,model_MIDA] = ftTrans_mida(ft,double(d_s),y_S',...
                d_s,par_MIDA);
            X_MIDA = [ones(n_S,1), ft_MIDA(1:n_S,:)]; %transf source feat (n_Sx3)
            theta_MIDA = (X_MIDA' * X_MIDA) \ X_MIDA' * y_S'; %least sq. sol.
            W_MIDA = ft' * model_MIDA.W; %
            X_MIDA_test = [ones(n_test,1), [x_Ctest', x_Etest'] * W_MIDA];
            predicted_MIDA = (X_MIDA_test * theta_MIDA)';
            RMSE_MIDA(i,j) = sqrt( mean( (y_test - predicted_MIDA).^2 ) );
            
            %Subspace Alignment (SA)
            [ft_SA, model_SA] = ftTrans_sa(ft,d_s,y_S',double(d_s),par_SA);
            X_SA = [ones(n_S,1), ft_SA(1:n_S,:)]; %transf source feat (n_Sx3)
            theta_SA = (X_SA' * X_SA) \ X_SA' * y_S'; %least sq. sol.
            W_SA = model_SA.WT; muT_SA = model_SA.muT;
            X_SA_test = [ones(n_test,1), ([x_Ctest',x_Etest']-muT_SA) * W_SA];
            predicted_SA = (X_SA_test * theta_SA)';
            RMSE_SA(i,j) = sqrt( mean( (y_test - predicted_SA).^2 ) );
            
            %Geodesic Flow Kernel (GFK)
            [ft_GFK,model_GFK]=ftTrans_gfk(ft,d_s,y_S',double(d_s),par_GFK);
            X_GFK = [ones(n_S,1), ft_GFK(1:n_S,:)]; %transf source feat (n_Sx3)
            theta_GFK = (X_GFK' * X_GFK) \ X_GFK' * y_S'; %least sq. sol.
            W_GFK = model_GFK.W;
            muT_GFK = mean([x_CT', x_ET']); sigmaT = std([x_CT', x_ET']);
            X_GFK_test = [ones(n_test,1),...
                (([x_Ctest', x_Etest']-muT_GFK))./sigmaT * W_GFK];
            predicted_GFK = (X_GFK_test * theta_GFK)';
            RMSE_GFK(i,j) = sqrt( mean( (y_test - predicted_GFK).^2 ) );
        end
    end    
end

%% REMOVE NaNs
nan_S = sum(isnan(RMSE_S)); RMSE_S(isnan(RMSE_S))=[];
nan_P = sum( isnan(RMSE_P),2 ); RMSE_P(logical(nan_P),: )=[];
nan_GFK = sum( isnan(RMSE_GFK),2 ); RMSE_GFK(logical(nan_GFK),: )=[];

%% PLOT
figure(1);
if model_ftTrans == 0
    subplot(1,2,1)
        loglog(n_T_range, mean(NegLogLik_P),'-rx',...
            n_T_range, mean(NegLogLik_S)*ones(1,N_T),'b',...
            'LineWidth',1)
        xlabel('n_T')
        ylabel('NLL')
        title(['n_S = ',num2str(n_S),', n_{iter} = ', num2str(n_iter)])
        legend('\theta_P','\theta_S')  
    subplot(1,2,2)
            loglog(n_T_range, mean(RMSE_P),'-rx',...
                n_T_range, mean(RMSE_S)*ones(1,N_T),'b',...
                n_T_range, mean(RMSE_LR)*ones(1,N_T),'g',...
                'LineWidth',1)
            xlabel('n_T')
            ylabel('RMSE')
            title(['n_S = ',num2str(n_S),', n_{iter} = ', num2str(n_iter)])
            legend('\theta_P', '\theta_S', '\theta_{LR}')
else   
    loglog(n_T_range, mean(RMSE_P),'-rx',...
        n_T_range, mean(RMSE_S)*ones(1,N_T),'b',...
        n_T_range, mean(RMSE_LR)*ones(1,N_T),'g',...
        n_T_range, mean(RMSE_TCA),'-mo',...
        n_T_range, mean(RMSE_MIDA),'-kd',...
        n_T_range, mean(RMSE_GFK),'-c^',...
        n_T_range, mean(RMSE_SA),'-ys',...        
        'LineWidth',1)
    xlabel('n_T')
    ylabel('RMSE')
    title(['n_S = ',num2str(n_S),', n_{iter} = ', num2str(n_iter),...
        ', \lambda = ', num2str(lambda), ', model \sigma? - ',...
        num2str(model_sigma), ', restricted? - ', num2str(model_restr)])
    legend('\theta_P', '\theta_S', '\theta_{LR}',...
        'TCA', 'MIDA', 'GFK', 'SA')
end

%% PLOT DATA
figure(2);
    subplot(1,3,1)
        plot(X_CS, Y_S, 'bo', X_CT, Y_T, 'rx')
        xlabel('X_C')
        ylabel('Y')
        title('Y vs X_C')
        legend('Source', 'Target')
    subplot(1,3,2)
        plot(Y_S, X_ES, 'bo', Y_T, X_ET, 'rx')
        xlabel('Y')
        ylabel('X_E')
        title('X_E vs Y')
        legend('Source', 'Target')
    subplot(1,3,3)
        plot(X_CS, X_ES, 'bo', X_CT, X_ET, 'rx')
        xlabel('X_C')
        ylabel('X_E') 
        title('X_E vs X_C')
        legend('Source', 'Target')      
figure(3);
    subplot(1,3,1)
        loglog(X_CS, Y_S, 'bo', X_CT, Y_T, 'rx')
        xlabel('X_C')
        ylabel('Y')
        title('Y vs X_C')
        legend('Source', 'Target')
    subplot(1,3,2)
        loglog(Y_S, X_ES, 'bo', Y_T, X_ET, 'rx')
        xlabel('Y')
        ylabel('X_E')
        title('X_E vs Y')
        legend('Source', 'Target')
    subplot(1,3,3)
        loglog(X_CS, X_ES, 'bo', X_CT, X_ET, 'rx')
        xlabel('X_C')
        ylabel('X_E') 
        title('X_E vs X_C')
        legend('Source', 'Target')