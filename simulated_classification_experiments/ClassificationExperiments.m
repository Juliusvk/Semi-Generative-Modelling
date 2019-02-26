%% CLASSIFICATION EXPERIMENTS ON SYNTHETIC DATA
clear all; close all; clc;

%% SET SIMULATION PARAMETERS
n_iter = 10^3;
n_S = 2^10;
n_T_range = 2^0;
N_T = size(n_T_range,2);
n_test = 1000;
model.misspecified = false;
model.bayesian = false;
model.ftTrans = false;
show.errors = true;
show.data = true;
show.posterior = false;
show.boundaries = false;
sigma_T = 1; sigma_S = 1; sigma_Y = 1; sigma_0 = 1; sigma_1 = 1;
options = optimset('MaxFunEvals',10^6, 'MaxIter', 10^6);

%% MCMC parameters
% initial theta (3 x 1): (m, mu_0, mu_1)
delta = 0.1; % step-size
nchains = 10;
nsamples = 1100;
burnin = 100;
sigma_prior = 10;

%% LOAD DEFAULT PARAMETERS FOR FEAT. TRANS. METHODS
par_TCA.kerName = 'lin';par_TCA.bSstca = true;      par_TCA.lambda = 0;
par_TCA.m = 2;          par_TCA.mu = 1;             par_TCA.gamma = .1;
par_MIDA.isRegress = 0; par_MIDA.kerName = 'lin';   par_MIDA.bSmida = true;
par_MIDA.m = 2;         par_MIDA.kerSigma = 1e-1;   par_SA.pcaCoef = 2; 
par_GFK.dr = 1;         par_ITL.pcaCoef = 2;        par_ITL.lambda = 10;

%% INITIALISE RESULT-MATRICES
error_S = zeros(n_iter,1);      error_WS = zeros(n_iter,1);
error_LR = zeros(n_iter,1);     error_P = zeros(n_iter,N_T);
error_TCA = zeros(n_iter,N_T);  error_MIDA = zeros(n_iter,N_T);
error_SA = zeros(n_iter,N_T);   error_GFK = zeros(n_iter,N_T);
error_ITL = zeros(n_iter,N_T);  
NLL_S = zeros(n_iter,1);        NLL_WS = zeros(n_iter,1);
NLL_P = zeros(n_iter,N_T);      NLL_LR = zeros(n_iter,1);

%% SIMULATE
t = 0;
for i=1:n_iter
    tic;
    disp(['Iteration: ', num2str(i), ' / ',num2str(n_iter)])
    %% draw model parameters
    [mu_S,mu_T,mu_Y,mu_0,mu_1] = getParamsClass(model.misspecified);
    % remove the following lines for randomly drawn parameters:
    mu_S = -1;
    mu_T = 1;
    mu_Y = 0;
    mu_0 = -0.5;
    mu_1 = -mu_0;
    
    params = [mu_S sigma_S mu_T sigma_T mu_Y sigma_Y mu_0 sigma_0...
        mu_1 sigma_1 model.misspecified];
    
    %% draw source and test data
    s_pos=0; s_neg=0; % to ensure at least two samples per class
    while s_pos < 2 || s_neg < 2
        [D_S, D_test] = getDataClass([n_S, n_test, params]);
        s_pos = sum(D_S(2,:) == 1); s_neg = sum(D_S(2,:) == 0);
    end
    X_CS = D_S(1,:);        Y_S = D_S(2,:);         X_ES = D_S(3,:);
    X_Ctest = D_test(1,:);  Y_test = D_test(2,:);   X_Etest = D_test(3,:);
    w_CS = normpdf(X_CS, mu_T, sigma_T) ./ normpdf(X_CS, mu_S, sigma_S);
    
    if model.bayesian == false
        %% theta_S
        nll_S = @(theta) nll_YGivenXC(X_CS, Y_S, ones(1,n_S), theta);
        th_S = fminsearch(nll_S, 0, options);
        m_0 = sum(X_ES(1,Y_S == 0)) / length( X_ES(1,Y_S == 0));
        m_1 = sum(X_ES(1,Y_S == 1)) / length( X_ES(1,Y_S == 1));
        theta_S = [th_S, m_0, m_1];
        error_S(i,1) = getErrorClass(theta_S, D_test);
        NLL_S(i,1) = getTestNLLClass(theta_S, D_test);
        
        %% theta_WS
        nll_WS = @(theta) nll_YGivenXC(X_CS, Y_S, w_CS, theta);
        th_WS = fminsearch(nll_WS, 0, options);
        m_0W = sum(w_CS(1,Y_S==0).*X_ES(1,Y_S == 0))/sum(w_CS(1,Y_S == 0));
        m_1W = sum(w_CS(1,Y_S==1).*X_ES(1,Y_S == 1))/sum(w_CS(1,Y_S == 1));
        theta_WS = [th_WS, m_0W, m_1W];
        error_WS(i,1) = getErrorClass(theta_WS, D_test);
        NLL_WS(i,1) = getTestNLLClass(theta_WS, D_test);
        
        %% theta_LR
        nll_LR = @(theta) nll_LogReg(X_CS, Y_S, X_ES, theta);
        theta_LR = fminsearch(nll_LR, zeros(1,3), options);
        error_LR(i,1) = getErrorLRClass(theta_LR, D_test);
        
        %% methods using unlabelled data
        for j=1:N_T
            %% draw target data
            n_T = n_T_range(j);
            lambda = n_S/(n_S+n_T);
            [~,D_T] = getDataClass([0, n_T, params]);
            X_CT = D_T(1,:);     Y_T = D_T(2,:);     X_ET = D_T(3,:);
            
            %% theta_P
            nll_P=@(theta) nll_Pooled(X_CS,Y_S,X_ES,X_CT,X_ET,lambda,theta);
            theta_P = fminsearch(nll_P, zeros(1,3), options);
            error_P(i,j) = getErrorClass(theta_P, D_test);
            NLL_P(i,j) = getTestNLLClass(theta_P, D_test);
            
            %% ft. trans. methods
            if model.ftTrans == true
                ft = [X_CS', X_ES'; X_CT', X_ET'];
                d_s = [true(n_S,1); false(n_T,1)];
                
                %% TCA
                [ft_TCA,model_TCA] = ftTrans_tca(ft,d_s,Y_S',d_s,par_TCA);
                X_CS_TCA = ft_TCA(1:n_S,1)'; X_ES_TCA = ft_TCA(1:n_S,2)';
                nll_TCA = @(theta) nll_LogReg(X_CS_TCA,Y_S,X_ES_TCA,theta);
                theta_TCA = fminsearch(nll_TCA, zeros(1,3), options);
                W_TCA = ft' * model_TCA.W; %
                X_test_TCA = (D_test([1,3],:))' * W_TCA;
                D_test_TCA = [(X_test_TCA(:,1))'; Y_test; ...
                    (X_test_TCA(:,2))'];
                error_TCA(i,j) = getErrorLRClass(theta_TCA, D_test_TCA);
                
                %% MIDA
                [ft_MIDA,model_MIDA]=ftTrans_tca(ft,d_s,Y_S',d_s,par_MIDA);
                X_CS_MIDA=ft_MIDA(1:n_S,1)'; X_ES_MIDA = ft_MIDA(1:n_S,2)';
                nll_MIDA=@(theta) nll_LogReg(X_CS_MIDA,Y_S,X_ES_MIDA,theta);
                theta_MIDA = fminsearch(nll_MIDA, zeros(1,3), options);
                W_MIDA = ft' * model_MIDA.W; %
                X_test_MIDA = (D_test([1,3],:))' * W_MIDA;
                D_test_MIDA = [(X_test_MIDA(:,1))'; Y_test; ...
                    (X_test_MIDA(:,2))'];
                error_MIDA(i,j) = getErrorLRClass(theta_MIDA, D_test_MIDA);
                
                %% Subspace Alignment (SA)
                [ft_SA,model_SA]=ftTrans_sa(ft,d_s,Y_S',double(d_s),par_SA);
                X_CS_SA=ft_SA(1:n_S,1)'; X_ES_SA = ft_MIDA(1:n_S,2)';
                nll_SA=@(theta) nll_LogReg(X_CS_SA,Y_S,X_ES_SA,theta);
                theta_SA = fminsearch(nll_SA, zeros(1,3), options);
                W_SA = model_SA.WT; muT_SA = model_SA.muT;
                X_test_SA = ((D_test([1,3],:))'-muT_SA) * W_SA;
                D_test_SA = [(X_test_SA(:,1))'; Y_test; ...
                    (X_test_SA(:,2))'];
                error_SA(i,j) = getErrorLRClass(theta_SA, D_test_SA);
                
                %% Geodesic Flow Kernel (GFK)
                [ft_GFK, model_GFK] = ftTrans_sa(ft,d_s,Y_S',...
                    double(d_s),par_GFK);
                X_CS_GFK=ft_GFK(1:n_S,1)'; X_ES_GFK = ft_MIDA(1:n_S,2)';
                nll_GFK=@(theta) nll_LogReg(X_CS_GFK,Y_S,X_ES_GFK,theta);
                theta_GFK = fminsearch(nll_GFK, zeros(1,3), options);
                W_GFK = model_GFK.WT;
                muT_GFK = mean([X_CT', X_ET']);
                sigmaT_GFK = std([X_CT', X_ET']);
                X_test_GFK=((D_test([1,3],:))'-muT_GFK)./sigmaT_GFK*W_GFK;
                D_test_GFK = [(X_test_GFK(:,1))'; Y_test; ...
                    (X_test_GFK(:,2))'];
                error_GFK(i,j) = getErrorLRClass(theta_GFK, D_test_GFK);
                
                %% ITL
                [ft_ITL, model_ITL] = ftTrans_itl(ft,d_s,Y_S',...
                    double(d_s),par_ITL);
                X_CS_ITL=ft_ITL(1:n_S,1)'; X_ES_ITL = ft_ITL(1:n_S,2)';
                X_CT_ITL=ft_ITL(1+n_S:end,1)';X_ET_ITL=ft_ITL(1+n_S:end,2)';
                nll_ITL=@(theta) nll_LogReg(X_CS_ITL,Y_S,X_ES_ITL,theta);
                theta_ITL = fminsearch(nll_ITL, zeros(1,3), options);
                W_ITL = model_ITL.L;
                X_test_ITL=(D_test([1,3],:))' * W_ITL;
                D_test_ITL = [(X_test_ITL(:,1))'; Y_test; ...
                    (X_test_ITL(:,2))'];
                error_ITL(i,j) = getErrorLRClass(theta_ITL, D_test_ITL);
            end
        end    
    elseif model.bayesian == true
        %% define source log-posterior
        logprior = @(theta) -sum((theta.^2)./(2*sigma_prior^2));
        logpost_EYS = @(theta) -Y_S .* (0.5 * (X_ES-theta(3)).^2) - ...
            (1-Y_S) .* (0.5 * (X_ES-theta(2)).^2);
        logpost_S = @(theta) logprior(theta) -...
                n_S * nll_YGivenXC(X_CS,Y_S,ones(1,n_S),theta) + ...
                sum(logpost_EYS(theta));
        logpost_WS = @(theta) logprior(theta) - ...
                n_S * nll_YGivenXC(X_CS,Y_S,w_CS,theta) + ...
                sum(w_CS .* logpost_EYS(theta));            
        %% Metropolis-Hastings sampling
        n_keep = nsamples - burnin;
        nsamples_tot = nchains * n_keep;
        sample_S=zeros(3,nsamples_tot); sample_WS=zeros(3,nsamples_tot);
        for k=1:nchains                
            theta_S = zeros(3,nsamples); theta_WS=zeros(3,nsamples);
            theta_0 = randn(3,1); theta_tS=theta_0; theta_tWS=theta_0; 
            for s=1:nsamples
                cand_S = mvnrnd(theta_tS,delta*eye(3))';
                cand_WS = mvnrnd(theta_tWS,delta*eye(3))';
                if exp(logpost_S(cand_S)-logpost_S(theta_tS))> rand
                    theta_tS = cand_S;    theta_S(:,s)=cand_S;
                else
                    theta_S(:,s)=theta_tS;
                end
                if exp(logpost_WS(cand_WS)-logpost_WS(theta_tWS))> rand
                    theta_tWS = cand_WS;    theta_WS(:,s)=cand_WS;
                else
                    theta_WS(:,s)=theta_tWS;
                end
            end
            sample_S(:, ((k-1)*n_keep+1) : k*n_keep ) = ...
                theta_S(:,(burnin+1):nsamples);
            sample_WS(:, ((k-1)*n_keep+1) : k*n_keep ) =...
                theta_WS(:,(burnin+1):nsamples);
        end
        %% Predict with samples
        predictions_S = zeros(size(sample_S,2), n_test);
        predictions_WS = zeros(size(sample_WS,2), n_test);
        negloglik_S = zeros(size(sample_S,2), 1);
        negloglik_WS = zeros(size(sample_WS,2), 1);
        for s=1:size(sample_S,2)
            predictions_S(s,:) = getClassProbs(D_test,sample_S(:,s));
            predictions_WS(s,:) = getClassProbs(D_test,sample_WS(:,s));
            negloglik_S(s,1) = getTestNLLClass(sample_S(:,s), D_test);
            negloglik_WS(s,1) = getTestNLLClass(sample_WS(:,s),D_test);
        end
        fitted_S = (mean(predictions_S,1) > 0.5);
        fitted_WS = (mean(predictions_WS,1) > 0.5);
        error_S(i,1) = 1- mean(fitted_S==D_test(2,:));
        error_WS(i,1) = 1 - mean(fitted_WS==D_test(2,:));
        NLL_S(i,1)=mean(negloglik_S);   NLL_WS(i,1)=mean(negloglik_WS);
        %% draw target data    
        for j=1:N_T
            n_T = n_T_range(j);
            lambda = n_S/(n_S+n_T);
            [~,D_T] = getDataClass([0, n_T, params]);
            X_CT = D_T(1,:);     Y_T = D_T(2,:);     X_ET = D_T(3,:);            
            %% define target log-posterior
            logpost_ECT = @(theta) -log(1 + exp(-(X_CT - theta(1)))) + ...
                log( normpdf(X_ET,theta(3),1)+exp(-(X_CT - theta(1))) .*...
                normpdf(X_ET,theta(2),1) );
            logpost_P = @(theta) logprior(theta) + ...
                sum(logpost_EYS(theta)) -...
                n_S * nll_YGivenXC(X_CS,Y_S,ones(1,n_S),theta) + ...
                sum(logpost_ECT(theta));
            %% Metropolis-Hastings sampling 
            sample_P=zeros(3,1);
            for k=1:nchains                
                theta_P = zeros(3,nsamples);
                theta_0 = randn(3,1);
                theta_tP=theta_0;
                for s=1:nsamples
                    cand_P = mvnrnd(theta_tP,delta*eye(3))';
                    if exp(logpost_P(cand_P)-logpost_P(theta_tP))> rand
                        theta_tP = cand_P;    theta_P(:,s)=cand_P;
                    else
                        theta_P(:,s)=theta_tP;
                    end
                end
                sample_P( : , ((k-1)*n_keep+1) : k*n_keep) =...
                    theta_P(:,(burnin+1):nsamples);
            end
            %% Predict with samples
            predictions_S = zeros(size(sample_S,2), n_test);
            predictions_WS = zeros(size(sample_WS,2), n_test);
            predictions_P = zeros(size(sample_P,2), n_test);
            negloglik_S = zeros(size(sample_S,2), 1);
            negloglik_WS = zeros(size(sample_WS,2), 1);
            negloglik_P = zeros(size(sample_P,2), 1);
            for s=1:size(sample_S,2)
                predictions_P(s,:) = getClassProbs(D_test,sample_P(:,s));
                negloglik_P(s,1) = getTestNLLClass(sample_P(:,s), D_test);
            end
            fitted_P = (mean(predictions_P,1) > 0.5);  
            error_P(i,j) = 1 - mean(fitted_P==D_test(2,:));
            NLL_P(i,j) = mean(negloglik_P);
        end
    end
    t = t+toc;  avg_t = t/i;
    disp(['ETA: ', num2str( round(avg_t*(n_iter-i)/3600, 1) ) , ' hours'])
end
avg_t = t/n_iter;
disp(['Average time per iteration: ', num2str(avg_t)])

%% CALCULATE & DISPLAY RESULTS
avg_error_S = mean(error_S);        avg_error_WS = mean(error_WS);
avg_error_P = mean(error_P);        avg_error_LR = mean(error_LR);
avg_error_TCA = mean(error_TCA);    avg_error_MIDA = mean(error_MIDA);
avg_error_SA = mean(error_SA);      avg_error_GFK = mean(error_GFK);
avg_error_ITL = mean(error_ITL);
avg_NLL_S = mean(NLL_S);            avg_NLL_WS = mean(NLL_WS);
avg_NLL_P = mean(NLL_P);

if show.errors == true
    disp(['theta_S: error = ',num2str(avg_error_S),...
        ', NLL = ',num2str(avg_NLL_S)])
    disp(['theta_WS: error = ',num2str(avg_error_WS),...
        ', NLL = ',num2str(avg_NLL_WS)])
    disp(['theta_P: error = ',num2str(avg_error_P),...
        ', NLL = ',num2str(avg_NLL_P)])
    disp(['theta_LR: error = ',num2str(avg_error_LR)])
end

%% TEST FOR SIGNIFICANCE USING A PAIRED T-TEST
h = zeros(1,N_T);
p = zeros(1,N_T);

for j=1:N_T
    [h(1,j), p(1,j)] = ttest( error_P(:,j), error_S(:,1) );
end


%% PLOT RESULTS
figure(1); % log-log plots
if model.ftTrans == false
    subplot(1,2,1)
    loglog(n_T_range, avg_NLL_P,'-rx',...
        n_T_range, avg_NLL_S*ones(1,N_T),'b',...
        n_T_range, avg_NLL_WS*ones(1,N_T),'--b',...
        'LineWidth',1)
    xlabel('n_T')
    ylabel('NLL')
    title(['n_S = ',num2str(n_S),', n_{iter} = ', num2str(n_iter)])
    legend('\theta_P','\theta_S','\theta_{WS}')
    subplot(1,2,2)
    loglog(n_T_range, avg_error_P,'-rx',...
        n_T_range, avg_error_S*ones(1,N_T),'b',...
        n_T_range, avg_error_WS*ones(1,N_T),'--b',...
        n_T_range, avg_error_LR*ones(1,N_T),'-.g',...
        'LineWidth',1)
    xlabel('n_T')
    ylabel('error rate')
    title(['n_S = ',num2str(n_S),', n_{iter} = ', num2str(n_iter)])
    legend('\theta_P','\theta_S','\theta_{WS}','\theta_{LR}')
else
    loglog(n_T_range, avg_error_P,'-rx',...
        n_T_range, avg_error_S*ones(1,N_T),'b',...
        n_T_range, avg_error_WS*ones(1,N_T),'--b',...
        n_T_range, avg_error_LR*ones(1,N_T),'-.g',...
        n_T_range, avg_error_TCA,'-mo',...
        n_T_range, avg_error_MIDA,'-kd',...
        n_T_range, avg_error_GFK,'-c^',...
        n_T_range, avg_error_SA,'-ys',...     
        'LineWidth',1)
    hold on
    loglog(n_T_range, avg_error_ITL,'*-','color', [.67 .31 .02],...
        'LineWidth',1)
    hold off
    xlabel('n_T')
    ylabel('error rate')
    title(['n_S = ',num2str(n_S),', n_{iter} = ', num2str(n_iter)])
    legend('\theta_P','\theta_S','\theta_{WS}','\theta_{LR}',...
        'TCA','MIDA','GFK', 'SA', 'ITL')
end

%% PLOT DATA
if show.data == true
    figure(2);
    hold on
    plot(X_CT(Y_T==0), X_ET(Y_T==0), 'k+', 'LineWidth',.5)
    plot(X_CT(Y_T==1), X_ET(Y_T==1), 'ks', 'LineWidth',.5)
    plot(X_CS(Y_S==0), X_ES(Y_S==0), 'rx', 'LineWidth',2)
    plot(X_CS(Y_S==1), X_ES(Y_S==1), 'bo', 'LineWidth',2)
    hold off
    xlabel('X_C')
    ylabel('X_E')
    legend('Y=0 (target)','Y=1 (target)','Y=0 (source)','Y=1 (source)')
end

%% PLOT POSTERIOR DISTRIBUTIONS
if model.bayesian == true && show.posterior == true
    figure(3);
    subplot(3,3,1)
        histogram(sample_S(1,:),'Normalization','pdf')
        hold on; plot([mu_Y mu_Y], ylim, 'LineWidth',3); hold off;
        ylabel('\theta_S')
    subplot(3,3,2)
        histogram(sample_S(2,:),'Normalization','pdf')
        hold on; plot([mu_0 mu_0], ylim, 'LineWidth',3); hold off;
    subplot(3,3,3)
        histogram(sample_S(3,:),'Normalization','pdf')
        hold on; plot([mu_1 mu_1], ylim, 'LineWidth',3); hold off;
    subplot(3,3,4)
        histogram(sample_WS(1,:),'Normalization','pdf')
        hold on; plot([mu_Y mu_Y], ylim, 'LineWidth',3); hold off;
        ylabel('\theta_{WS}')
    subplot(3,3,5)
        histogram(sample_WS(2,:),'Normalization','pdf')
        hold on; plot([mu_0 mu_0], ylim, 'LineWidth',3); hold off;
    subplot(3,3,6)
        histogram(sample_WS(3,:),'Normalization','pdf')
        hold on; plot([mu_1 mu_1], ylim, 'LineWidth',3); hold off;
    subplot(3,3,7)
        histogram(sample_P(1,:),'Normalization','pdf')
        hold on; plot([mu_Y mu_Y], ylim, 'LineWidth',3); hold off;
        ylabel('\theta_{P}')
        xlabel('m')
    subplot(3,3,8)
        histogram(sample_P(2,:),'Normalization','pdf')
        hold on; plot([mu_0 mu_0], ylim, 'LineWidth',3); hold off;
        xlabel('\mu_0')
    subplot(3,3,9)
        histogram(sample_P(3,:),'Normalization','pdf')
        hold on; plot([mu_1 mu_1], ylim, 'LineWidth',3); hold off;
        xlabel('\mu_1')
end

%% PLOT DECISION BOUNDARY
if model.misspecified == true && show.boundaries == true
    mog1 = gmdistribution([-mu_1; mu_1], 1, [0.5, 0.5]);
    range = -10:.1:10; res = length(range);
    boundary_true = zeros(res);
    for i = 1 : res
        for j = 1 : res
            boundary_true(i,j) = yGivenXC(range(i),mu_Y,1).*pdf(mog1,range(j))./ ...
                ( yGivenXC(range(i),mu_Y,1).*pdf(mog1,range(j)) + ...
                (1-yGivenXC(range(i),mu_Y,1)).*normpdf(range(j), mu_0, 1) );
        end
    end
    [X_C, X_E] = meshgrid(range);
    boundary_S = decisionBoundary(sample_S, X_C, X_E);
    boundary_P = decisionBoundary(sample_P, X_C, X_E);    
    figure(4);
        imagesc(boundary_true'); colorbar;
        xlabel('X_C')
        ylabel('X_E')
    figure(5);
        imagesc(boundary_S); colorbar;
        xlabel('X_C')
        ylabel('X_E')        
    figure(6);
        imagesc(boundary_P); colorbar;
        xlabel('X_C')
        ylabel('X_E')    
end
%% Errorbar plots
figure(1); % log-log plots
    subplot(1,2,1)
    loglog(n_T_range, avg_NLL_S*ones(1,N_T),'b',...
        n_T_range, avg_NLL_WS*ones(1,N_T),'--b',...
        'LineWidth',1)
    hold on
    errorbar(n_T_range, avg_NLL_P, std(NLL_P),'-xr','LineWidth',1)
    hold off
    xlabel('n_T')
    ylabel('NLL')
    title(['n_S = ',num2str(n_S),', n_{iter} = ', num2str(n_iter)])
    legend('\theta_S','\theta_{WS}','\theta_P')
    subplot(1,2,2)
    loglog(n_T_range, avg_error_S*ones(1,N_T),'b',...
        n_T_range, avg_error_WS*ones(1,N_T),'--b',...
        n_T_range, avg_error_LR*ones(1,N_T),'-.g',...
        'LineWidth',1)
    hold on
    errorbar(n_T_range, avg_error_P, std(error_P),'-xr','LineWidth',1)
    hold off
    xlabel('n_T')
    ylabel('error rate')
    title(['n_S = ',num2str(n_S),', n_{iter} = ', num2str(n_iter)])
    legend('\theta_S','\theta_{WS}','\theta_{LR}','\theta_P')


%% PLOT DATA
if show.data == true
    figure(2);
    hold on
    plot(X_CT(Y_T==0), X_ET(Y_T==0), 'k+', 'LineWidth',.5)
    plot(X_CT(Y_T==1), X_ET(Y_T==1), 'ks', 'LineWidth',.5)
    plot(X_CS(Y_S==0), X_ES(Y_S==0), 'rx', 'LineWidth',2)
    plot(X_CS(Y_S==1), X_ES(Y_S==1), 'bo', 'LineWidth',2)
    hold off
    xlabel('X_C')
    ylabel('X_E')
    legend('Y=0 (target)','Y=1 (target)','Y=0 (source)','Y=1 (source)')
end

%% 
function [classprobs ] = decisionBoundary( theta, X_C, X_E )
    % theta: matrix of parameter samples (3 x N)
    % [X_C, X_E] = meshgrid(a:b:c, d:e:f)
    sigma_0 = 1;    sigma_1 = 1;    sigma_Y = 1;
    classprobs = zeros(size(X_C));
        for k=1:size(theta,2)
            mu_Y = theta(1,k);
            mu_0 = theta(2,k);
            mu_1 = theta(3,k);
            classprobs = classprobs +...
                1 ./ ( 1 + (sigma_1 / sigma_0) * exp( - 0.5 * ( ...
                ((X_E - mu_0)./sigma_0).^2 - ((X_E - mu_1)./sigma_1).^2 ) - ...
                sigma_Y * (X_C - mu_Y) ) );
        end
    classprobs = classprobs./size(theta,2);
end