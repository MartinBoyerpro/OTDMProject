clear;
%
% Parameters for dataset generation
%
num_target =1;
tr_freq    = 0.0;
tr_p       = 20000;
te_q       = tr_p/10;
tr_seed    = 123456;
te_seed    = 789101;
%
% Parameters for optimization
%
% isd=search direction (1=GM; 2=CGM; 3=BFGS);
% icg=CGM variant(1=FR; 2=PR+);
% irc= Restart for the CGM (0= no restart; 1=𝑹𝑹𝑹𝑹𝑹𝑹 ; 2=( ));nu=𝝂𝝂.
la = 0.01;                                                     % L2 regularization.
epsG = 10^-6; kmax = 10000;                                   % Stopping criterium.
ils=3; ialmax = 1;
kmaxBLS=30; epsal=10^-3;c1=0.01; c2=0.45;  % Linesearch.
isd = 7; icg = 1; irc = 2 ; nu = 1.0;                         % Search direction.
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
   disp("niter = "+niter);
   disp("accuracy"+ acc(Xte,yte,wo));
%BFGS-quasi Newton Method
elseif isd == 3
       [wo,niter] =  BFGS (epsG,kmax,ialmax,L,gL,wo,c1,c2,kmaxBLS,epsal);
       disp("niter = "+niter);
       disp("accuracy"+ acc(Xte,yte,wo));

elseif isd == 7
    [wo] =  SGM (wo,la,L,Le,gL,Xtr,ytr,Xte,yte,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest);
     disp("accuracy"+ acc(Xte,yte,wo));
end

 
%      disp("w star");
%      disp(w0);

%      %BFGS
%
%       H=eye(35,35);
%       I=eye(35,35);
%       k=0;
%       for i=1:size(ytr,2)
%        L2=@(w) L(w,Xtr(:,i),ytr(:,i));
%        gL2=@(w)gL(w,Xtr(:,i),ytr(:,i));
%           while(norm(gL2(w0)))>epsG
%
%               d=-H*gL2(w0);
%               w0save=w0;
%               [al,iWout] = uo_BLSNW32(L2,gL2,w0,d,ialmax,c1,c2,kmaxBLS,epsal);
%               w0=w0+al*d;
%               s=w0-w0save;
%               y=gL2(w0)-gL2(w0save);
%               rhoBFGS=1/(y'*s);
%               H=(I-rhoBFGS*s*y')*H*(I-rhoBFGS*y*s')+rhoBFGS*s*s';
%               k=k+1;
% %               xk = [xk,x1]; alk = [alk,al];dk=[dk,d];iWk=[iWk,iWout];
%           end
%       end
%      % calcul de l'accuracy sur le dataset de training cf diapo n°6
%
%uo_nn_Xyplot(Xte,yte,w0);
%      sumTr = 0;
%      for j=1:sizeOfData(2)
%             if round(y(Xtr(:,j),w0))== ytr(:,j)
%                 sumTr = sumTr +1;
%             end
%      end
%      accuracyTr = (100/size (ytr,2 ))*sumTr;
%      disp("accuracy training : "+accuracyTr);
%      % calcul de l'accuracy sur le dataset de test cf diapo n°7
%       sumTe = 0;
%      for h=1:sizeOfData(2)
%             if round(y(Xte(:,h),w0))== yte(:,h)
%                 sumTe = sumTe +1;
%             end
%      end
%      accuracyTe = (100/size (yte,2 ))*sumTe;
%      disp("accuracy testing : "+accuracyTe);


% SGD method



end

