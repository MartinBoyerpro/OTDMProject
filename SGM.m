function[wstar,niter,fo] = SGM (wo,la,L,Le,gL,Xtr,ytr,Xte,yte,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest);
p = size(Xtr,2);
m = sg_ga*p;
sg_Ke = p/m;
sg_Kmax = sg_emax*sg_Ke;
e = 0;
s = 0;
%best value
te_Lbest = +inf;
wstar = ones(1,35)'*0;
k=1;
count=0;
%%%%init wstar && te_Lbest
while e <= sg_emax && s< sg_ebest
    count=count+1;
    %Take a random permutation of the training data set:
    %Xtr + ytr
    dataSet = [ytr;Xtr];
    %   disp( dataSet(1,:));
    %permutation of the dataSet
    P = dataSet(:,randperm(p));
    %minibatch
    for i=0 : p/m-1
        %count=count+1;
        S = P(:,i*round(m)+1:min((i+1)*round(m),p));
        XtrS = S(2:size(S,1),:);
        ytrS = S(1,:);
        d  = -gL(wo);
        sg_k = sg_be*sg_Kmax;
        sg_al = 0.01*sg_al0;
        if k<=  sg_k
            al = (1-k/ sg_k)*sg_al0 + (k/sg_k)* sg_al;
        elseif k >  sg_k
            al = sg_al;
        end
        %update solution
        wo = wo + al*d;
        k=k+1;
    end
    e=e+1;
    te_L = Le(wo);
    %update loss function
    if te_L < te_Lbest
        te_Lbest = te_L;
        wstar = wo;
        s = 0;
    else
        s=s+1;
    end
    fo=L(wo);
end
    niter=count;
end
