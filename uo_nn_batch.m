  clear;
%
% Parameters.
%
tr_seed = 123456; te_seed = 789101;
tr_p = 250; te_q = 250; tr_freq = 0.5;                        % Datasets generation
epsG = 10^-6; kmax = 1000;                                    % Stopping condition.
ils=3; ialmax = 1; kmaxBLS=10; epsal=10^-3; c1=0.01; c2=0.45; % Linesearch.
icg = 2; irc = 2 ; nu = 1.0;                                  % Search direction.
sg_seed = 565544; sg_al0 = 2; sg_be = 0.3; sg_ga = 0.01;      % SGM iteration.
sg_emax = kmax; sg_ebest = floor(0.01*sg_emax);               % SGM stopping condition.
%
% Optimization
%
global iheader; iheader = 1;
csvfile = strcat('uo_nn_batch_',num2str(tr_seed),'-',num2str(te_seed),'.csv');
fileID = fopen(csvfile ,'w');
t1=clock;
for num_target = [1:10]
    for la = [0.0, 0.01, 0.1]
        for isd = [1,3,7]
            [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu);
            if iheader == 1
                fprintf(fileID,'num_target;      la; isd;  niter;     tex; tr_acc; te_acc;        L*;\n');
            end
            fprintf(fileID,'         %1i; %7.4f;   %1i; %6i; %7.4f;  %5.1f;  %5.1f;  %8.2e;\n', mod(num_target,10), la, isd, niter, tex, tr_acc, te_acc, fo);
            iheader=0;
        end
    end
end
t2=clock;
total_t = etime(t2,t1);
fprintf(' wall time = %6.1d s.\n', total_t);
fclose(fileID);
uo_nn_batch_BP_log(tr_seed,te_seed,sg_seed, total_t, csvfile);

function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu)
tsolve1=clock;
%create the training data set
[Xtr,ytr] = uo_nn_dataset(tr_seed , tr_p , num_target , tr_freq);

%create the evaluater data set
[Xte,yte] =uo_nn_dataset(te_seed,te_q,num_target,tr_freq);
%Loss function and it gradient
sig = @(Xds) 1./(1+ exp(-Xds));
y = @(Xds,w) sig (w'*sig(Xds));

% L = @(w,Xds,yds ) (norm(y(Xds,w)-yds)^2)/size (yds,2)+ (la*norm(w)^2)/2;
% gL = @(w,Xds,yds) (2*sig(Xds)*((y(Xds,w)-yds).*y(Xds,w).*(1-y(Xds,w))')/size(yds,2))+la*w;

L  = @(w) (norm(y(Xtr,w)-ytr)^2)/size (ytr,2) + (la*norm(w)^2)/2;                      % Loss function.
gL = @(w) (2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))')/size(ytr,2)+la*w;    % Gradient.
Le =  @(w) (norm(y(Xte,w)-yte)^2)/size (yte,2) + (la*norm(w)^2)/2; 
acc = @(Xds,yds,wo) 100*sum(yds==round(y(Xds,wo)))/size(Xds,2);
%initialization of weights
wo=ones(1,35)'*0;
%Gradient method
if isd == 1
   [wo,niter] = GM(epsG,kmax,ialmax,L,gL,wo,c1,c2,kmaxBLS,epsal);
   fo = L(wo);
   tr_acc = acc(Xtr,ytr,wo);
   te_acc = acc(Xte,yte,wo);
%    uo_nn_Xyplot(Xtr,ytr,wo);
%BFGS-quasi Newton Method
elseif isd == 3
       [wo,niter] =  BFGS (epsG,kmax,ialmax,L,gL,wo,c1,c2,kmaxBLS,epsal);
       fo = L(wo);
       tr_acc = acc(Xtr,ytr,wo);
       te_acc = acc(Xte,yte,wo);
%        disp("niter = "+niter);
%        uo_nn_Xyplot(Xtr,ytr,wo);
       
elseif isd == 7
    [wo] =  SGM (wo,la,L,Le,gL,Xtr,ytr,Xte,yte,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest);
    fo = L(wo);
    tr_acc = acc(Xtr,ytr,wo);
    te_acc = acc(Xte,yte,wo);
    
    niter = 2021;
%        disp("training");
%        uo_nn_Xyplot(Xtr,ytr,wo);
%        disp("Testing");
%        uo_nn_Xyplot(Xte,yte,wo);
end

tsolve2=clock;
tex = etime(tsolve2,tsolve1);




end
