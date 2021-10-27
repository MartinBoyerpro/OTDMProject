function[wo,niter,fo] = GM (epsG,kmax,ialmax,L,gL,wo,c1,c2,kmaxBLS,epsal)
%initialization of the iteration's number
k=1;
% Vector of solution points.
wok  = [wo];
% Vector of descent directions.
dk  = [];
wo = wo;
almax = 1;
% Vector of step lengths.
alk = [almax];
while(norm(gL(wo)))>epsG && k<kmax
    %gradient direction
    d=-gL(wo);
    if k>1
        if (ialmax)==1
            almax = alk(:,k-1)*((gL(wok(:,k-1))'*dk(:,k-1))/(gL(wo)'*d));
        else
            almax = 2*(L(wo)-L(wok(:,k-1)))/(gL(wo)'*d);
        end
    end
    %call BLS
    [almax,iw ] = uo_BLSNW32(L,gL,wo,d,almax,c1,c2,kmaxBLS,epsal);
    %update the solution
    wo = wo + almax.*d;
    k = k+1;
    %disp(k);
    %fprintf('%3d %10.6f  %10.6f\n',k, almax, L(wo));
    % update in order to calculate almax for the next iteration.
    wok  = [wok wo];
    dk  = [dk d];
    alk = [alk almax];
    fo=L(wo);
    
end
niter = k;
end
