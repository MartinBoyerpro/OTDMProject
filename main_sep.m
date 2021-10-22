clear;
%
% Parameters for dataset generation
%
num_target =[1 3 5 7];
tr_freq    = .5;        
tr_p       = 250;       
te_q       = 250;       
tr_seed    = 123456;    
te_seed    = 789101;    
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

function [Xtr,ytr,w0,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu)
    %create the training data set
    [Xtr,ytr] = uo_nn_dataset(tr_seed , tr_p , num_target , tr_freq);
    
    %create the evaluater data set
    [Xte,yte] =uo_nn_dataset(te_seed,te_q,num_target,tr_freq);
    %Loss function and it gradient
    sig = @(Xds) 1./(1+ exp(-Xds));
    y = @(Xds,w) sig (w'*sig(Xds));
    L = @(w,Xds,yds ) (norm(y(Xds,w )-yds)^2)/size (yds,2)+ (la*norm(w)^2)/2;
    gL = @(w,Xds,yds) (2*sig(Xds)*((y(Xds,w)-yds).*y(Xds,w).*(1-y(Xds,w))')/size(yds,2))+la*w;

%     L2=@(w) L(w,Xtr(:,end),ytr(:,end));
%     gL2=@(w)gL(w,Xtr(:,end),ytr(:,end));
    
  w0=ones(1,35)'*0;
  % w0= rand(35,1);
% uo_nn_Xyplot(Xtr,ytr,w0)
%     sg = sig(Xtr);
%     disp("sig : "+size(sg));
%     ytt = y(Xtr,w0);
%     disp("y : "+size(ytt));
%     
%     test  = (2*sig(Xtr)*((y(Xtr,w0)-ytr).*y(Xtr,w0).*(1-y(Xtr,w0))')/size(ytr,2));
%     disp(size(test));
%     disp(size(ytr));
%     
    %Gradient method
     sizeOfData = size(Xtr);
     %boucle qui va passer chaque colonne des dataset Xtr et ytr dans la
     %fonction Loss, chaque colonne = un chiffre qui va nous permettre
     %d'entrainer le réseaux
     %entrainer le réseaux = afiner les poids pour minimiser la fonction
     %Loss
     for i=1:size(ytr,2)
       L2=@(w) L(w,Xtr(:,i),ytr(:,i));
       gL2=@(w)gL(w,Xtr(:,i),ytr(:,i));
       d=-gL2(w0);
       k=0;
       while(norm(gL2(w0)))>epsG && k<kmax
%             alsave=ialmax;
%             dsave=d;
%             gsave=gL2(w0);
            d=-gL2(w0);
            [ialmax,iWout] = uo_BLSNW32(L2,gL2,w0,d,ialmax,c1,c2,kmaxBLS,epsG);
%             ialmax=alsave*((gsave'*dsave)/(gL2(w0)'*d));
            w0 = w0 + ialmax.*d;
            k = k+1; 
%             disp("k = "+k);
           
%             xk = [xk,x1]; alk = [alk,al];dk=[dk,d];iWk=[iWk,iWout];
            %% 
       end
     end
     disp("w0 = "+w0);
%      % calcul de l'accuracy sur le dataset de training cf diapo n°6
      uo_nn_Xyplot(Xtr,ytr,w0);
      uo_nn_Xyplot(Xte,yte,w0);
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
       
end

