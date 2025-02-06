clear all
close all

manifun()
function manifun()
    %
    K  = 2;
    delta1 = .125;
    delta2 = .065;
    sigma  = 2.e-1;
    r =3.e-2 ;
    
    % Mesh 
    k = 6;
    t0 = 0; T = 1;
    dt = 2^(-k); t = (t0:dt:T)'; nbt = length(t); 
    dx = dt*sqrt(dt); x = (0:dx:1)'; nbx = length(x);
    dp = dt; p = (0:dp:1)'; nbp = length(p);
    
    ti = find(t==0.25);
    xj = find(x==0.5);
    pk = find(p==0.);
    p0 = find(p==0);
    p05 = find(p==0.5);
    p1 = find(p==1);
    
    Xtrain = repmat(x, 1, nbp);
    Xright= (Xtrain(:) + bfun(Xtrain(:))*dt + sigmafun(Xtrain(:))*sqrt(dt))';
    Xleft = (Xtrain(:) + bfun(Xtrain(:))*dt - sigmafun(Xtrain(:))*sqrt(dt))';

    net = feedforwardnet(90);
    rng('default');
    % net.trainParam.lr	=0.0005;
    net.trainFcn = 'trainbr';
     % net.trainParam.mu = .5;
    % net.trainParam.mu_dec = .1;
    % net.trainParam.mu_inc = 10;
    % net.trainParam.min_grad = 1.e-9;
    net.trainParam.goal = dx;
    % net.trainParam.epochs = 2000;

    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    % net.performParam.normalization = 'percent';
    % net.performParam.regularization = 1.e-12; 

    V = zeros(nbt, nbx, nbp);  nV = zeros(nbt, nbx, nbp);  
    V(end, :, :) = p'.*g(x) + (1-p').*g(x);
    % nV(end, :, :) = p'.*g(x) + (1-p').*g(x);
    
    U = squeeze(V(end, :, :));

    hh1 = h1(x); hh2 = h2(x);
    ff1 = f1(x); ff2 = f2(x);
    F =  p'.*ff1 + (1-p)'.*ff2;
    H =  p'.*hh1 + (1-p)'.*hh2;

    tic
    er = zeros(1, nbt-1);
    for i=nbt:-1:2
        disp(['Time ', num2str(i),'/', num2str(nbt)])

        Ytrain = squeeze(V(i, :, :));
        fun = train(net, Xtrain(:)', Ytrain(:)', "useParallel", "yes");  
        er(i-1)= perform(fun, Ytrain(:)', fun(Xtrain(:)') );
        U(:) = 0.5*(fun(Xleft) + fun(Xright));
        U(:) = min(max(U(:)', F(:)'), H(:)');
        V(i-1, :, :) = convex_envelope(U, p);
    end
    toc
    yline(dt,'r--')
    hold on
    yline(dx,'g-')
    plot(er,'ko-')
    ylim([0,1.1*dt])
    Vff3 = squeeze(V(:,:,:));
    % nVff3 = squeeze(nV(:,:,:));

    
    
    % mymap1 = [1 0 0; 1 1 1;   0 1 0];
    % ETAfff2 = zeros(size(Vff2(:,:,pk)'));
    % ETAhff2 = ETAfff2;
    % 
    % 
    % ETAfff2 =abs( Vff2(:,:,pk)' - p(pk)*ff1 - (1-p(pk))*ff2)<= 0;
    % ETAhff2 = Vff2(:,:,pk)'  >= p(pk)*hh1+(1-p(pk))*hh2;
    % pcolor(x,t, -(ETAhff2 & ~ETAfff2)'+(~ETAhff2 & ETAfff2)')
    % surf(x,t ,squeeze(Vff2(:,:,pk)), -(ETAhff2 & ~ETAfff2)'+(~ETAhff2 & ETAfff2)')

    save('txp.mat', 't', 'x', 'p','-v7.3');
    save('F.mat', 'F','-v7.3');
    save('H.mat', 'H','-v7.3');
    save("Vff3.mat","Vff3",'-v7.3');
    % save("nVff3.mat","nVff3");

    % plot(x, ff1)
    % hold on
    % plot(x, hh1)
    % 
    % plot(x, V(ti,:,p0))

    function output = bfun(x)     % Drift
            output  = (r - sigma^2/2) *ones(size(x));
    end
    %
    function output = sigmafun(x)   % Diffusion
        output  = sigma*ones(size(x));
    end
    function output = g(x)
        output = max(K - exp(x), 0);
    end
    function output = f1(x)
        output = g(x) - 0*delta1;
    end
    function output = f2(x)
        output = g(x) - 0*delta2;
    end
    function output = h1(x)
        output = g(x) + delta1;
    end
    function output = h2(x)
        output = g(x) + delta2;
    end
end