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
almin= 10^-3;
la = .1;                                                     % L2 regularization.
epsG = 10^-6; kmax = 10000;                                   % Stopping criterium.
ils=3; ialmax = 2; kmaxBLS=30; epsal=10^-3;c1=0.01; c2=0.45;  % Linesearch.
isd = 1; icg = 2; irc = 2 ; nu = 1.0;                         % Search direction.
sg_seed = 565544; sg_al0 = 2; sg_be = 0.3; sg_ga = 0.01;      % SGM iteration.
sg_emax = kmax; sg_ebest = floor(0.01*sg_emax);               % SGM stopping condition.
%
% Optimization
%
sig = @(Xds) 1./(1+ exp(-Xds));
y = @(Xds,w) sig (w'*sig(Xds));
L = @(w,Xds,yds ) (norm(y(Xds,w )-yds)^2)/size (yds,2)+ (la*norm(w)^2)/2;
gL = @(w,Xds,yds) (2*sig(Xds)*(( y( Xds,w)-yds).*y(Xds,w).*(1-y(Xds,w))')/size(yds,2))+la*w;


t1=clock;
[Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu);
t2=clock;
fprintf(' wall time = %6.1d s.\n', etime(t2,t1));
%
function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu);
[Xtr,ytr] =uo_nn_dataset(1234,10,[4],0.5);
[Xte,yte] =uo_nn_dataset(1234,10,[4],0);
sig = @(Xds) 1./(1+ exp(-Xds));
y = @(Xds,w) sig (w'*sig(Xds));
L = @(w,Xds,yds ) (norm(y(Xds,w )-yds)^2)/size (yds,2)+ (la*norm(w)^2)/2;
gL = @(w,Xds,yds) (2*sig(Xds)*((y(Xds,w)-yds).*y(Xds,w).*(1-y(Xds,w))')/size(yds,2))+la*w;

w= zeros(size(Xtr));
%disp(w);
sigtest=sig(Xtr);
gltr=gL(w,Xtr,ytr);
disp(gltr);
[wk, dk, alk, iWk,betak,Hk,tauk]=uo_solve(w,L,gL,1,epsG,kmax,ialmax,0,0,c1,c2,0,isd,icg,irc,nu,0)


end



function [xk, dk, alk, iWk,betak,Hk,tauk] = uo_solve(x1,f,g,h,epsG,kmax,almax,almin,rho,c1,c2,iW,isd,icg,irc,nu,delta)
        k=1;
        tauk=1;
        xk = [x1];
        dk=[];
        al=almax;
        alk=[]   ; 
        betak=[];
        Hk=[];
        iWk=[]
        
   %GM     
   if isd==1 
        d=-g(x1);
        while(norm(g(x1)))>epsG 
            alsave=al;
            dsave=d;
            gsave=g(x1);
            d=-g(x1);
            [al,iWout] = uo_BLSNW32(f,g,x1,d,almax,c1,c2,kmax,epsG)
            almax=alsave*((gsave'*dsave)/(g(x1)'*d));
            x1 = x1 + al*d;
            k = k+1;  xk = [xk,x1]; alk = [alk,al];dk=[dk,d];iWk=[iWk,iWout];
            %% 
        end
        
        alk = [alk,NaN];
        for k=1:size(xk,2)
            %fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
        end 
  %CGM      
  elseif isd== 2
      %CGM-FR
      if icg ==1
          %CGM-FR,RC0
          if irc == 0
              d=-g(x1);
              while(norm(g(x1)))>epsG 
                dsave=d;
                [al,iWout] = uo_BLS(x1,d,f,g,almax,almin,rho,c1,c2,iW);
                xsave=x1;
                x1 = x1 + al*d;
                k = k+1;  xk = [xk,x1]; alk = [alk,al];dk=[dk,d];iWk=[iWk,iWout];
                Beta=(g(x1)'*g(x1))/(norm(g(xsave))*norm(g(xsave)));
                betak=[betak,Beta];
                d=-g(x1)+Beta*dsave;
                %% 
              end
              
              
              for k=1:size(xk,2)-1
                %fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
              end 
          %CGM-FR,RC1    
          elseif irc == 1
               d=-g(x1);
              while(norm(g(x1)))>epsG 
                dsave=d;
                [al,iWout] = uo_BLS(x1,d,f,g,almax,almin,rho,c1,c2,iW);
                xsave=x1;
                x1 = x1 + al*d;
                k = k+1;  xk = [xk,x1]; alk = [alk,al];dk=[dk,d];iWk=[iWk,iWout];
                Beta=(g(x1)'*g(x1))/(norm(g(xsave))*norm(g(xsave)));
                betak=[betak,Beta];
                %restart condition
                if mod(k,12)==0
                   d=-g(x1); 
                else
                  d=-g(x1)+Beta*dsave;
                end

              end
              
             
              for k=1:size(xk,2)-1
               % fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
              end
          %CGM-FR,RC2    
          elseif irc == 2
                 d=-g(x1);
              while(norm(g(x1)))>epsG 
                dsave=d;
                [al,iWout] = uo_BLS(x1,d,f,g,almax,almin,rho,c1,c2,iW);
                xsave=x1;
                x1 = x1 + al*d;
                k = k+1;  xk = [xk,x1]; alk = [alk,al];dk=[dk,d];iWk=[iWk,iWout];
                Beta=(g(x1)'*g(x1))/(norm(g(xsave))*norm(g(xsave)));
                betak=[betak,Beta];
                %restart condition
                if (abs(g(x1)-g(xsave))/norm(g(xsave))*norm(g(xsave)))>= nu
                   d=-g(x1); 
                else
                  d=-g(x1)+Beta*dsave;
                end

              end
              
              
              for k=1:size(xk,2)-1
                %fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
              end
          end
      %CGM-PR+
      elseif icg==2
              %CGM-PR+,RC0
              if irc==0
            d=-g(x1);
                  while(norm(g(x1)))>epsG 
                    dsave=d;
                    [al,iWout] = uo_BLS(x1,d,f,g,almax,almin,rho,c1,c2,iW);
                    xsave=x1;
                    x1 = x1 + al*d;
                    k = k+1;  xk = [xk,x1]; alk = [alk,al];dk=[dk,d];iWk=[iWk,iWout];
                    Beta= max(0,(g(x1)'*(g(x1)-g(xsave)))/(norm(g(xsave))*norm(g(xsave))));
                    betak=[betak,Beta];
                    d=-g(x1)+Beta*dsave;
                    %% 
                  end
                  
                  
                  for k=1:size(xk,2)-1
                    %fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
                  end 
              %CGM-PR+,RC1    
              elseif irc==1
                  d=-g(x1);
                  while(norm(g(x1)))>epsG 
                    dsave=d;
                    [al,iWout] = uo_BLS(x1,d,f,g,almax,almin,rho,c1,c2,iW);
                    xsave=x1;
                    x1 = x1 + al*d;
                    k = k+1;  xk = [xk,x1]; alk = [alk,al];dk=[dk,d];iWk=[iWk,iWout];
                    Beta= max(0,(g(x1)'*(g(x1)-g(xsave)))/(norm(g(xsave))*norm(g(xsave))));
                    betak=[betak,Beta];
                    %restart condition
                    if mod(k,12)==0
                       d=-g(x1); 
                    else
                      d=-g(x1)+Beta*dsave;
                    end
                    %% 
                  end
                  
                  
                  for k=1:size(xk,2)-1
                    %fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
                  end
              %CGM-PR+,RC2    
              elseif irc==2
                  d=-g(x1);
                  while(norm(g(x1)))>epsG 
                    dsave=d;
                    [al,iWout] = uo_BLS(x1,d,f,g,almax,almin,rho,c1,c2,iW);
                    xsave=x1;
                    x1 = x1 + al*d;
                    k = k+1;  xk = [xk,x1]; alk = [alk,al];dk=[dk,d];iWk=[iWk,iWout];
                    Beta= max(0,(g(x1)'*(g(x1)-g(xsave)))/(norm(g(xsave))*norm(g(xsave))));
                    betak=[betak,Beta];
                    if (abs(g(x1)-g(xsave))/norm(g(xsave))*norm(g(xsave)))>= nu
                        d=-g(x1); 
                    else
                        d=-g(x1)+Beta*dsave;
                    end
                    %% 
                  end
                  
                  
                  for k=1:size(xk,2)-1
                    %fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
                  end 
              end            

      end

  %BFGS
  elseif isd==3
      H=eye(2,2);
      I=eye(2,2);
      k=0;
      while(norm(g(x1)))>epsG
          Hk=[Hk;H];
          d=-H*g(x1);
          xsave=x1;
          [al,iWout] = uo_BLS(x1,d,f,g,almax,almin,rho,c1,c2,iW);
          x1=x1+al*d;
          s=x1-xsave;
          y=g(x1)-g(xsave);
          rhoBFGS=1/(y'*s);
          H=(I-rhoBFGS*s*y')*H*(I-rhoBFGS*y*s')+rhoBFGS*s*s';
          k=k+1;
          xk = [xk,x1]; alk = [alk,al];dk=[dk,d];iWk=[iWk,iWout];
      end
      
      
                  for k=1:size(xk,2)-1
                   % fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
                  % disp(Hk(2*k-1:2*k,:))
                  end 
                  

  end


    
end

%BLS

function [X,y] = uo_nn_dataset(seed, ncol, target, freq)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
% target: if <>[] : target numbers.
%         if = [] : the function is asked to randomly generate 'ncol'
%         digits.
% freq : 
% Output:
% y     : if <>[] : y(i)=1 if X(:,i) is one of the target digits in 'target'; y(i)=0 otherwise.
%         if = [] : y(i) integer stored in X(:,i).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = [
    0 0 1 0 0, 0 1 1 0 0, 0 0 1 0 0, 0 0 1 0 0, 0 0 1 0 0, 0 0 1 0 0, 0 1 1 1 0; % 1
    0 1 1 1 0, 1 0 0 0 1, 0 0 0 0 1, 0 0 0 1 0, 0 0 1 0 0, 0 1 0 0 0, 1 1 1 1 1; % 2
    0 1 1 1 0, 1 0 0 0 1, 0 0 0 0 1, 0 0 1 1 0, 0 0 0 0 1, 1 0 0 0 1, 0 1 1 1 0; % 3
    0 0 1 1 0, 0 1 0 1 0, 1 0 0 1 0, 1 0 0 1 0, 1 1 1 1 1, 0 0 0 1 0, 0 0 0 1 0; % 4
    1 1 1 1 1, 1 0 0 0 0, 1 1 1 1 0, 0 0 0 0 1, 0 0 0 0 1, 1 0 0 0 1, 0 1 1 1 0; % 5
    0 0 1 1 1, 0 1 0 0 0, 1 0 0 0 0, 1 1 1 1 0, 1 0 0 0 1, 1 0 0 0 1, 0 1 1 1 0; % 6
    1 1 1 1 1, 0 0 0 0 1, 0 0 0 0 1, 0 0 0 1 0, 0 0 1 0 0, 0 1 0 0 0, 1 0 0 0 0; % 7
    0 1 1 1 0, 1 0 0 0 1, 1 0 0 0 1, 0 1 1 1 0, 1 0 0 0 1, 1 0 0 0 1, 0 1 1 1 0; % 8
    0 1 1 1 0, 1 0 0 0 1, 1 0 0 0 1, 0 1 1 1 1, 0 0 0 0 1, 0 0 0 0 1, 0 1 1 1 0; % 9
    0 1 1 1 0, 1 0 0 0 1, 1 0 0 0 1, 1 0 0 0 1, 1 0 0 0 1, 1 0 0 0 1, 0 1 1 1 0; % 0
    ];
N = N';
%%
xval     = 10;
maxsigma = 1.;
minsigma = 0.25;
if xval > 0
    N = xval*((N-1)+N);
end
if ~isempty(seed) rng(seed); end;
nT = size(target,2);
nPixels = size(N,1);
%% Generation
for j=1:ncol
    if nT  > 0 & rand() < (freq-0.1*nT)/(1-0.1*nT)
        i = target(randi([1,nT]));
    else
        i = randi(10);
    end
    X(:,j) = N(:,i);
    if isempty(target)
        y(j) = i;
    else
        
        if any(i == target)
            y(j) = 1;
        else
            y(j) = 0;
        end
    end
end
%% Blur
for j=1:ncol
    l = zeros(1,nPixels);
    sigma = minsigma + (maxsigma-minsigma)*rand();
    for k = 1:nPixels
        ii = randi([1 nPixels]);
        while l(ii)==1
            ii = randi([1 nPixels]);
        end
        l(ii) = 1;
        if X(ii,j) > 0
            X(ii,j) =  xval + sigma*xval*randn();
        else
            X(ii,j) = -(xval + sigma*xval*randn());
        end
    end
end
end


function [alphas,iout] = uo_BLSNW32(f,g0,x0,d,alpham,c1,c2,maxiter,eps)
% function alphas = strongwolfe(f,d,x0,alpham)
% Line search algorithm satisfying strong Wolfe conditions
% Algorithms 3.5 on pages 60-61 in Nocedal and Wright
% MATLAB code by Kartik Sivaramakrishnan
% Last modified: January 27, 2008
%
% F.-Javier Heredia, September 2018 <fjh...>
% g,c1,c2,maxiter, eps
% iout = 1 : too many iterations
% iout = 2 : stacked, alpha_[i]=alpha^[i-1]

alpha0 = 0;
alphap = alpha0;
g = @(x) g0(x)';
%<fjh
iout = 0;
if c1 == 0
    c1 = 1e-4;
end
if c2==0
    c2 = 0.5;
end
%alphax = alpham*rand(1);
alphax = alpham;
%[fx0,gx0] = feval(f,x0,d);
fx0 = f(x0);
gx0 = g(x0)*d;
%>
fxp = fx0;
gxp = gx0;
i=1;
% alphap is alpha_{i-1}
% alphax is alpha_i
while (1 ~= 2 && i < maxiter)
  %<fjh
    if abs(alphap-alphax) < eps
      iout = 2;
      alphas = alphax;
      return
    end
  %>
    xx = x0 + alphax*d;
  %<fjh
  %[fxx,gxx] = feval(f,xx,d);
    fxx = f(xx);
    gxx = g(xx)*d;
  %>
    if (fxx > fx0 + c1*alphax*gx0) || ((i > 1) && (fxx >= fxp)),
    [alphas,iout_zoom] = zoom(f,g,x0,d,alphap,alphax,c1,c2,eps);
    %<fjh
    if iout_zoom == 2
        iout = 2;
    end
    %>
    return;
  end
  if abs(gxx) <= -c2*gx0,
    alphas = alphax;
    return;
  end
  if gxx >= 0,
    [alphas,iout_zoom] = zoom(f,g,x0,d,alphax,alphap,c1,c2,eps);
    %<fjh
    if iout_zoom == 2
        iout = 2;
    end
    %>
    return;
  end
  alphap = alphax;
  fxp = fxx;
  gxp = gxx;
  alphax = alphax + (alpham-alphax)*rand(1);
  i = i+1;
end
if i==maxiter
    iout = 1;
    alphas = alphax;
end
end


function [alphas,iout] = zoom(f,g,x0,d,alphal,alphah,c1,c2,eps)
% function alphas = zoom(f,g,x0,d,alphal,alphah)
% Algorithm 3.6 on page 61 in Nocedal and Wright
% MATLAB code by Kartik Sivaramakrishnan
% Last modified: January 27, 2008
% F.-Javier Heredia, September 2018 <fjh...>

%<fjh
%[fx0,gx0] = feval(f,x0,d);
fx0 = f(x0);
gx0 = g(x0)*d;
iout = 0;
%>
while (1~=2),
    %<fjh
    if abs(alphal-alphah) < eps
      iout = 2;
      alphas = alphax;
      return
    end
    %>

   alphax = 1/2*(alphal+alphah);
   xx = x0 + alphax*d;
   %<fjh
   %[fxx,gxx] = feval(f,xx,d);
   fxx = f(xx);
   gxx = g(xx)*d;
   %>
   xl = x0 + alphal*d;
   %<fjh
   %fxl = feval(f,xl,d);
   fxl = f(xl);
   %>
   if ((fxx > fx0 + c1*alphax*gx0) || (fxx >= fxl)),
      alphah = alphax;
   else
      if abs(gxx) <= -c2*gx0,
        alphas = alphax;
        return;
      end
      if gxx*(alphah-alphal) >= 0,
        alphah = alphal;
      end
      alphal = alphax;
   end
end 
end





