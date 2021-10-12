fprintf('[uo_FDM_CE21]\n'); clear;
f = @(x) x(1)^2 + x(2)^3 + 3*x(1)*x(2); g = @(x) [ 2*x(1)+3*x(2) ; 3*x(2)^2 + 3*x(1)]; h = @(x) [];
% Input parameters.
x1 = [-3;-1]; % starting solution.
epsG= 10^-6; kmax= 1500; % Stopping criterium:
almax= 2; almin= 10^-3; rho=0.5;c1=0.01;c2=0.45; iW= 1;% Linesearch:
isd= 3; icg= 2; irc= 1 ; nu = 0.1; % Search direction
delta = 0; % this parameter is only useful with SDM
[xk,dk,alk,iWk,betak,Hk,tauk] = uo_solve(x1,f,g,h,epsG,kmax,almax,almin,rho,c1,c2,iW,isd,icg,irc,nu,delta); save('uo_FDM_CE21.mat','f','g','h','epsG','kmax','almax','almin','rho','c1','c2','iW','isd','icg','irc','nu','xk','dk','alk','iWk','betak');
 % Problem
% Optimization: tauk is only useful with SDM.
% Optimization's log
xo=[-2.25;1.5]; xylim = []; logfreq = 1;
[la1k,kappak,rk,Mk] = uo_solve_log(x1,f,g,h,epsG,kmax,almax,almin,rho,c1,c2,iW,isd,icg,irc,nu,delta,xk,dk,alk,iWk,betak,Hk, tauk,xo,xylim,logfreq);
fprintf('[uo_FDM_CE21]\n');


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
    if isd==1 
        d=-g(x1);
        while(norm(g(x1)))>epsG 
            dsave=d;
            d=-g(x1);
            alsave=al;
            xsave=x1;
            [al,iWout] = uo_BLSNW32(x1,d,f,g,almax,almin,rho,c1,c2,iW);
            x1 = x1 + al*d;
            k = k+1;  xk = [xk,x1]; alk = [alk,al];dk=[dk,d];iWk=[iWk,iWout];
            almax=alsave*((g(xsave)'*dsave)/g(x1)'*d);
            %% 
        end
        
        alk = [alk,NaN];
        for k=1:size(xk,2)
            fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
        end 
  elseif isd== 2
      if icg ==1
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
                fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
              end 
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
                fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
              end
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
                fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
              end
          end

      elseif icg==2
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
                    fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
                  end 
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
                    fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
                  end
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
                    fprintf('k =  %3i, xk =[%+5.2e, %+5.2e],  ||gk|| = %+5.2e, alk =  %5.3f \n',k, xk(1,k), xk(2,k),norm(g(xk(:,k))),alk(k));
                  end 
              end            

      end

  
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

function [al,iWout] = uo_BLS(x,d,f,g,almax,almin,rho,c1,c2,iW)
    al = almax;
    checkStrong = iW==2;  % input asks for strong
     % update conditions
    wc1 = f(x+al*d) <= f(x) + c1*(g(x).'*d)*al;
    wc2 = g(x+al*d).'*d >= c2*(g(x).'*d);
    swc2 = abs(g(x+al*d).'*d) <= c2*abs(g(x).'*d);
    if (wc1)
        iWout = 1;
        if (wc2); iWout = 2; end
        if (and(checkStrong, swc2)); iWout = 3; end
    else
        iWout = 0;
    end
    while and(al > almin, not(iWout == iW + 1))
        [al,iWout] = uo_BLS(x,d,f,g,almax*rho,almin,rho,c1,c2,iW);
        % update condition
    end

% iWout = 0: al does not satisfy any WC
% iWout = 1: al satisfies (WC1)
% iWout = 2: al satisfies WC
% iWout = 3: al satisfies SWC
end
function [gk,la1k,kappak,rk,Mk] = uo_solve_log(x1,f,g,h,epsG,kmax,almax,almin,rho,c1,c2,iW,isd,icg,irc,nu,delta,xk,dk,alk,iWk,betak,Hk,tauk,xo,xylim,logfreq)
diary ('uo_solve_log.out'); diary on;
niter = size(xk,2); n= size(x1,1); nxo = size(xo,1);
if nxo==0 xo = xk(:,niter); end
fk  = []; gk = []; rk  = []; gdk = []; rk=[]; Mk=[];
la1k=zeros(1,niter);     % if NM or MNM: lowest vap of h(xk:,k)); if QNM: lowest vap of Hk(:,:,k).
kappak = zeros(1,niter); % if NM : cond. number of h(xk(:,k))), if +def; if QNM or MNM: cond. number of Hk or Bk resp.
for k = 1:niter
    x = xk(:,k); fk = [fk,f(x)]; gk = [gk,g(x)]; 
    if isd >3 la1k(k) = min(eig(h(x))); end
    if isd == 4 & all(eig(h(x))>0) kappak(k)= cond(h(xk(:,k))); end
    if k < niter
        if isd < 4
            rk = [rk, (f(xk(:,k+1))-f(xo))/(f(xk(:,k))-f(xo))   ];
            Mk = [Mk, (f(xk(:,k+1))-f(xo))/(f(xk(:,k))-f(xo))^2 ];
        else
            rk = [ rk, norm(g(xk(:,k+1))) / norm(g(xk(:,k)))   ];
            Mk = [ Mk, norm(g(xk(:,k+1))) / norm(g(xk(:,k)))^2 ];
        end
        gdk = [gdk,gk(:,k)'*dk(:,k)];
        if  isd == 3 la1k(k) = min(eig(Hk(2*k-1:2*k,:))); end
        if  isd >= 3 & isd ~=4  kappak(k) = cond(Hk(2*k-1:2*k,:)); end
    end
end
if niter > 1
    if (isd  < 4 & nxo==0) rk(niter-1)=rk(niter-2); Mk(niter-1)= Mk(niter-2); end
    if isd <= 4  tauk(1:niter-1) = 0; end
    if isd == 5  tauk(1:niter-1) = delta; end
end
if isd == 3  la1k(niter)     = 0; end

fprintf('   [uo_solve_log]\n');
fprintf('   f= %s\n', func2str(f));
fprintf('   epsG= %3.1e, kmax= %4d\n', epsG,kmax);
if isd ~= 4 fprintf('   almax= %2d, almin= %3.1e, rho= %4.2f, c1= %3.2f, c2= %3.2f, iW= %1d\n',almax,almin,rho,c1,c2,iW); end
fprintf('   isd= %1d\n',isd);
if isd == 2 fprintf('   icg= %1d, irc= %1d, nu= %3.1f\n',icg,irc,nu); end
if isd == 5 fprintf('   delta= %3.1d\n',delta); end
if n==2 fprintf('   x1 = [ %+3.1e , %+3.1e ]\n', x1(1), x1(2)); end
if isd ==1
    fprintf('      k     g''*d       al iW    ||g||        f        r        M\n');
elseif isd ==2
    fprintf('      k     g''*d       al iW     beta    ||g||        f        r        M\n');
elseif isd == 3
    fprintf('      k     g''*d       al iW    la(1)    kappa    ||g||        f        r        M\n');   
else
    fprintf('      k     g''*d       al iW    la(1) del./tau    kappa    ||g||        f        r        M\n');   
end
for k = 1:logfreq:niter-1
    if isd == 1 
        fprintf(' %6d %+3.1e %+3.1e  %1d %+3.1e %+3.1e %+3.1e %+3.1e\n', k, gdk(k), alk(k), iWk(k), norm(gk(:,k)), fk(k), rk(k),Mk(k));
    elseif isd == 2
        fprintf(' %6d %+3.1e %+3.1e  %1d %+3.1e %+3.1e %+3.1e %+3.1e %+3.1e\n', k, gdk(k), alk(k), iWk(k),  betak(k), norm(gk(:,k)), fk(k), rk(k),Mk(k));
    elseif isd == 3
        fprintf(' %6d %+3.1e %+3.1e  %1d %+3.1e %+3.1e %+3.1e %+3.1e %+3.1e %+3.1e\n', k, gdk(k), alk(k), iWk(k), la1k(k), kappak(k), norm(gk(:,k)), fk(k), rk(k),Mk(k));      
    else        
        fprintf(' %6d %+3.1e %+3.1e  %1d %+3.1e %+3.1e %+3.1e %+3.1e %+3.1e %+3.1e %+3.1e\n', k, gdk(k), alk(k), iWk(k), la1k(k), tauk(k), kappak(k), norm(gk(:,k)), fk(k), rk(k),Mk(k));      
    end
end
if isd == 1
    fprintf(' %6d                      %+3.1e %+3.1e\n', niter, norm(gk(:,niter)), fk(niter));
    fprintf('      k     g''*d       al iW    ||g||        f        r        M\n');
elseif isd ==2
    fprintf(' %6d                                        %+3.1e %+3.1e\n', niter, norm(gk(:,niter)), fk(niter));
    fprintf('      k     g''*d       al iW     beta    ||g||        f        r        M\n');
elseif isd == 3
    fprintf(' %6d                                        %+3.1e %+3.1e\n', niter, norm(gk(:,niter)), fk(niter));
    fprintf('      k     g''*d       al iW    la(1)    kappa    ||g||        f        r        M\n');
else
    fprintf(' %6d                      %+3.1e                   %+3.1e %+3.1e\n', niter, la1k(niter), norm(gk(:,niter)), fk(niter));
    fprintf('      k     g''*d       al iW    la(1) del./tau    kappa    ||g||        f        r        M\n');
end

if n==2
    if isd==1
        fprintf('   x* = [ %+3.1e , %+3.1e ]; beta = %+3.1e \n', xk(1,niter), xk(2,niter),(max(eig(h(xk(:,niter))))-min(eig(h(xk(:,niter)))))^2/(max(eig(h(xk(:,niter))))+min(eig(h(xk(:,niter)))))^2);
    else fprintf('   x* = [ %+3.1e , %+3.1e ]\n', xk(1,niter), xk(2,niter));
    end
end
fprintf('   [uo_solve_log]\n');
fs = 0;
if n == 2
    if size(xylim) == [0 0] xylim = [0 0 0 0]; end
    subplot(2,2,1); uo_solve_plot(f, xk, gk, xylim, 1, fs); subplot(2,2,2); uo_solve_plot(f, xk, gk, xylim, 2,fs);
    subplot(2,2,3);plot(rk(1:niter-1),'-o');xlabel('Iterations k');title('r^k'); subplot(2,2,4); plot(Mk(1:niter-1),'-x');xlabel('Iterations k'); title('M^k');
else
    subplot(1,2,1);plot(rk(1:niter-1),'-o');xlabel('Iterations k');title('r^k'); subplot(1,2,2); plot(Mk(1:niter-1),'-x');xlabel('Iterations k'); title('M^k');
end
diary off;
end
