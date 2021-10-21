clear;
%
% Parameters for dataset generation
%
num_target =[8];
tr_freq    = .5;        
tr_p       = 250;       
te_q       = 250;       
tr_seed    = 590152;    
te_seed    = 590100;    
%
% Parameters for optimization
%
la = .1;                                                     % L2 regularization.
epsG = 10^-6; kmax = 10000;                                   % Stopping criterium.
ils=3; ialmax = 2; kmaxBLS=30; epsal=10^-3;c1=0.01; c2=0.45;  % Linesearch.
isd = 7; icg = 2; irc = 2 ; nu = 1.0;                         % Search direction.
sg_seed = 565544; sg_al0 = 2; sg_be = 0.3; sg_ga = 0.01;      % SGM iteration.
sg_emax = kmax; sg_ebest = floor(0.01*sg_emax);               % SGM stopping condition.
%
% Optimization
%
t1=clock;
[Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu);
t2=clock;
fprintf(' wall time = %6.1d s.\n', etime(t2,t1));
%

function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu)
    %create the training data set
    [Xtr,ytr] = uo_nn_dataset(tr_seed , tr_p , num_target , tr_freq);
    %create the evaluater data set
    [Xte,yte] =uo_nn_dataset(te_seed,te_q,num_target,tr_freq);
    %Loss function and it gradient
    sig = @(Xds) 1./(1+ exp(-Xds));
    y = @(Xds,w) sig (w'*sig(Xds));
    L = @(w,Xds,yds ) (norm(y(Xds,w )-yds)^2)/size (yds,2)+ (la*norm(w)^2)/2;
    gL = @(w,Xds,yds) (2*sig(Xds)*((y(Xds,w)-yds).*y(Xds,w).*(1-y(Xds,w))')/size(yds,2))+la*w;

    L2=@(w) L(w,Xtr,ytr);
    gL2=@(w) reshape(gL(w,Xtr,ytr),8750,1);
    
    
    w0=zeros(35,1);
  
    
    %Gradient method
     d=-gL2(w0);
     d2 = reshape(d,8750,1);
%      disp(d2);
     
       while(norm(gL2(w0)))>epsG 
            alsave=ialmax;
            dsave=d;
            gsave=gL2(w0);
            d=-gL2(w0);
            [ialmax,iWout] = uo_BLSNW32(L2,gL2,w0,d2,ialmax,c1,c2,kmax,epsG);
            ialmax=alsave*((gsave'*dsave)/(gL2(w0)'*d));
            w0 = w0 + ialmax.*d;
            k = k+1;  xk = [xk,x1]; alk = [alk,al];dk=[dk,d];iWk=[iWk,iWout];
            %% 
        end
end

