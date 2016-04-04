function [U, out] = ftvcs_al_TVL2_3D(A,b,p,q,r,opts)
%
% This 3D version modified by: Dongjin Kwon
% Mar. 29, 2016
%
% Original version written by: Chengbo Li
% Advisor: Prof. Yin Zhang and Wotao Yin
% Computational and Applied Mathematics department, Rice University
% May. 12, 2009
%
% Goal: solve
%
%         min sum ||D_i u|| + mu/2||Au-b||_2^2    (with or without the constraint u>=0)
%
%       to recover image/signal u from encoded b,
%
%       which is equivalent to solve
%
%         min sum ||w_i|| + mu/2||Au-b||_2^2      s.t. D_i u = w_i
%
% ftvcs_al solves Augmented Lagrangian function:
% 
%         min_{u,w} sum ||w_i|| - sigma'(Du-w) + beta/2 ||Du-w||_2^2 + mu/2||Au-b||_2^2 ,
%
% by an alternating algorithm:
% i)  while norm(up-u)/norm(up) > tol_inn
%     1) Fix w^k, do Gradient Descent to 
%            - sigma'(Du-w^k) + beta/2||Du-w^k||^2 + mu/2||Au-f||^2;
%            u^k+1 is determined in the following way:
%         a) compute step length tau > 0 by BB formula
%         b) determine u^k+1 by
%                  u^k+1 = u^k - alpha*g^k,
%            where g^k = -D'sigma + beta D'(Du^k - w^k) + mu A'(Au^k-f), 
%            and alpha is determined by Amijo-like nonmonotone line search;
%     2) Given u^k+1, compute w^k+1 by shrinkage
%                 w^k+1 = shrink(Du^k+1-sigma/beta, 1/beta);
%     end
% ii) update the Lagrangian multiplier by
%             sigma^k+1 = sigma^k - beta(Du^k+1 - w^k+1).
% iii)accept current u as the initial guess to run the loop again
%
% Inputs:
%       A       : either an matrix representing the measurement or a struct 
%                   with 2 function handles:
%                 A(x,1) defines @(x) A*x;
%                 A(x,2) defines @(x) A'*x;
%       b       : either real or complex input vector representing the noisy observation of a
%                   grayscale image
%       p, q, r : size of original image
%       opts    : structure to restore parameters
%
% variables in this code:
%
% lam1 = sum ||wi||
% lam2 = ||Du-w||^2 (at current w).
% lam3 = ||Au-f||^2
% lam4 = sigma'(Du-w)
%
% f  = lam1 + beta/2 lam2 + mu/2 lam3 - lam4
%
% g  = A'(Au-f)
% g2 = D'(Du-w) (coefficients beta and mu are not included)
% 
% Numerical tests illustrate that this solver requirs large beta.
%

global D Dt

[D, Dt] = defDDt3D;

% unify implementation of A
if ~isa(A,'function_handle')
  A = @(u,mode) f_handleA(A,u,mode);
end

% get or check opts
opts = ftvcs_al_opts(opts); 

% problem dimension
n = p*q*r;

% mark important constants
mu = opts.mu;
beta = opts.beta;
tol_inn = opts.tol_inn;
tol_out = opts.tol;
gam = opts.gam;

% check if A*A'=I
tmp = rand(length(b),1);
if norm(A(A(tmp,2),1)-tmp,1)/norm(tmp,1) < 1e-3
  opts.scale_A = false;
end
clear tmp;

% check scaling A
if opts.scale_A
  [mu,A,b] = ScaleA(n,mu,A,b,opts.consist_mu); 
end 

% check scaling b
if opts.scale_b
  [mu,b,scl] = Scaleb(mu,b,opts.consist_mu);
end

% calculate A'*b
Atb = A(b,2);

% initialize U, beta
muf = mu;
betaf = beta;                                     % final beta
[U,mu,beta] = ftvcs_al_init(p,q,r,Atb,scl,opts);  % U: p*q*r
if mu > muf; mu = muf; end
if beta > betaf; beta = betaf; end
muDbeta = mu/beta;                                % muDbeta: constant
rcdU = U;

% initialize multiplers
sigmax = zeros(p,q,r);                            % sigmax, sigmay, sigmaz: p*q*r
sigmay = zeros(p,q,r);
sigmaz = zeros(p,q,r);

% initialize D^T sigma
DtsAtd = zeros(n,1); 

% initialize out.n2re
if isfield(opts,'Ut')
  Ut = opts.Ut*scl;                               % true U, just for computing the error
  nrmUt = norm_fro_3D(Ut);
else
  Ut = []; 
end
if ~isempty(Ut)
  out.n2re = norm_fro_3D(U - Ut)/nrmUt; 
end

% prepare for iterations
out.mus = mu; out.betas = beta;
out.res = []; out.itrs = []; out.f = []; out.obj = []; out.reer = [];
out.lam1 = []; out.lam2 = []; out.lam3 = []; out.lam4 = [];
out.itr = Inf;
out.tau = []; out.alpha = []; out.C = []; gp = [];
out.cnt = [];

[Ux,Uy,Uz] = D(U);                                % Ux, Uy, Uz: p*q*r
% initialize gradient W by shrinke method (depending on TVnorm)
if opts.TVnorm == 1
  Wx = max(abs(Ux) - 1/beta, 0).*sign(Ux);
  Wy = max(abs(Uy) - 1/beta, 0).*sign(Uy);
  Wz = max(abs(Uz) - 1/beta, 0).*sign(Uz);
  lam1 = sum(sum(sum(abs(Wx) + abs(Wy) + abs(Wz))));
else
  V = sqrt(Ux.*conj(Ux) + Uy.*conj(Uy) + Uz.*conj(Uz)); % V: p*q*r
  V(V==0) = 1;
  S = max(V - 1/beta, 0)./V;                      % S: p*q*r
  Wx = S.*Ux;                                     % Wx, Wy, Wz: p*q*r
  Wy = S.*Uy;
  Wz = S.*Uz;
  lam1 = sum(sum(sum(sqrt(Wx.*conj(Wx) + Wy.*conj(Wy) + Wz.*conj(Wz)))));
end

[lam2,lam3,lam4,f,g2,Au,g] = get_g(U,Ux,Uy,Uz,Wx,Wy,Wz,lam1,beta,mu,A,b,Atb,sigmax,sigmay,sigmaz);
% lam,f: constant, g2: p*q*r (=n), Au: m, g: p*q*r

% compute gradient
d = g2 + muDbeta*g - DtsAtd;

count = 1;
Q = 1; C = f;                                     % Q, C: costant
out.f = [out.f; f]; out.C = [out.C; C];
out.lam1 = [out.lam1; lam1];
out.lam2 = [out.lam2; lam2];
out.lam3 = [out.lam3; lam3];
out.lam4 = [out.lam4; lam4];

for ii = 1:opts.maxit
  if opts.disp
    %fprintf('outer iter = %d, total iter = %d, f = %f\n', count, ii, f);
    fprintf('outer iter = %d, total iter = %d, f = %f, C = %f, lam1 = %f, lam2= %f, lam3 = %f, lam4 = %f\n', count, ii, f, C, lam1, lam2, lam3, lam4);
  end

  % compute tau first
  if ~isempty(gp)
    dg = g - gp;                                  % dg: p*q*r
    dg2 = g2 - g2p;                               % dg2: p*q*r
    ss = uup'*uup;                                % ss: constant
    sy = uup'*(dg2 + muDbeta*dg);                 % sy: constant
    % sy = uup'*((dg2 + g2) + muDbeta*(dg + g));
    % compute BB step length
    tau = abs(ss/max(sy,eps));                    % tau: constant
    
    fst_itr = false;
  else
    % do Steepest Descent at the 1st ieration
    % d = g2 + muDbeta*g - DtsAtd;
    [dx,dy,dz] = D(reshape(d,p,q,r));             % dx, dy, dz: p*q*r
    dDd = norm_fro_3D(dx)^2 + norm_fro_3D(dy)^2 + norm_fro_3D(dz)^2;  % dDd: cosntant
    Ad = A(d,1);                                  % Ad: m
    % compute Steepest Descent step length
    tau = abs((d'*d)/(dDd + muDbeta*(Ad)'*Ad));
    
    % mark the first iteration 
    fst_itr = true;
  end    
  
  % keep the previous values
  Up = U; gp = g; g2p = g2; Aup = Au; 
  Uxp = Ux; Uyp = Uy; Uzp = Uz;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % ONE-STEP GRADIENT DESCENT %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  taud = tau*d;
  U = U(:) - taud;
  % projected gradient method for nonnegtivity
  if opts.nonneg
    U = max(real(U),0);
  end
  U = reshape(U,p,q,r);                           % U: p*q*r (still)
  [Ux,Uy,Uz] = D(U);                              % Ux, Uy, Uz: p*q*r
  
  [lam2,lam3,lam4,f,g2,Au,g] = get_g(U,Ux,Uy,Uz,Wx,Wy,Wz,lam1,beta,mu,A,b,Atb,sigmax,sigmay,sigmaz);
 
  % Nonmonotone Line Search
  alpha = 1;
  du = U - Up;                                    % du: p*q*r
  const = opts.c*beta*(d'*taud);

  % Unew = Up + alpha*(U - Up)
  cnt = 0; flag = true;
  
  while f > C - alpha*const
    %if cnt == 5
    if cnt == 100
      % give up and take Steepest Descent step

      % shrink gam
      gam = opts.rate_gam*gam;

      %d = g2p + muDbeta*gp - DtsAtd;
      [dx,dy,dz] = D(reshape(d,p,q,r));
      dDd = norm_fro_3D(dx)^2 + norm_fro_3D(dy)^2 + norm_fro_3D(dz)^2;
      Ad = A(d,1);
      tau = abs((d'*d)/(dDd + muDbeta*(Ad)'*Ad));
      U = Up(:) - tau*d;
      % projected gradient method for nonnegtivity
      if opts.nonneg
        U = max(real(U),0);
      end
      U = reshape(U,p,q,r);
      [Ux, Uy, Uz] = D(U);
      Uxbar = Ux - sigmax/beta;
      Uybar = Uy - sigmay/beta;
      Uzbar = Uz - sigmaz/beta;
      if opts.TVnorm == 1
        % ONE-DIMENSIONAL SHRINKAGE STEP
        Wx = max(abs(Uxbar) - 1/beta, 0).*sign(Uxbar);
        Wy = max(abs(Uybar) - 1/beta, 0).*sign(Uybar);
        Wz = max(abs(Uzbar) - 1/beta, 0).*sign(Uzbar);
        lam1 = sum(sum(sum(abs(Wx) + abs(Wy) + abs(Wz))));
      else
        % TWO-DIMENSIONAL SHRINKAGE STEP
        V = sqrt(Uxbar.*conj(Uxbar) + Uybar.*conj(Uybar) + Uzbar.*conj(Uzbar)); % V: p*q*r
        V(V==0) = 1;
        S = max(V - 1/beta, 0)./V;                % S: p*q*r
        Wx = S.*Uxbar;
        Wy = S.*Uybar;
        Wz = S.*Uzbar;
        lam1 = sum(sum(sum(sqrt(Wx.*conj(Wx) + Wy.*conj(Wy) + Wz.*conj(Wz)))));
      end
      [lam2,lam3,lam4,f,g2,Au,g] = get_g(U,Ux,Uy,Uz,Wx,Wy,Wz,lam1,beta,mu,A,b,Atb,sigmax,sigmay,sigmaz);
      alpha = 0; % remark the failure of back tracking
      break;
    end
    if flag
      dg = g - gp;
      dg2 = g2 - g2p;
      dAu = Au - Aup;                             % dAu: m
      dUx = Ux - Uxp;
      dUy = Uy - Uyp;
      dUz = Uz - Uzp;
      flag = false;
    end
    alpha = alpha*opts.gamma;
    [U,lam2,lam3,lam4,f,Ux,Uy,Uz,Au,g,g2] = update_g(p,q,r,lam1,alpha,beta,mu,...
      Up,du,gp,dg,g2p,dg2,Aup,dAu,Wx,Wy,Wz,Uxp,dUx,Uyp,dUy,Uzp,dUz,b,sigmax,sigmay,sigmaz);
    cnt = cnt + 1;
  end
  if opts.disp
    fprintf('  count of back tracking: %d, alpha = %f\n', cnt, alpha);
  end
  
  % if back tracking is successful, then recompute
  if alpha ~= 0
    Uxbar = Ux - sigmax/beta;
    Uybar = Uy - sigmay/beta;
    Uzbar = Uz - sigmaz/beta;
    if opts.TVnorm == 1
      % ONE-DIMENSIONAL SHRINKAGE STEP
      Wx = max(abs(Uxbar) - 1/beta, 0).*sign(Uxbar);
      Wy = max(abs(Uybar) - 1/beta, 0).*sign(Uybar);
      Wz = max(abs(Uzbar) - 1/beta, 0).*sign(Uzbar);
    else
      % TWO-DIMENSIONAL SHRINKAGE STEP
      V = sqrt(Uxbar.*conj(Uxbar) + Uybar.*conj(Uybar) + Uzbar.*conj(Uzbar));
      V(V==0) = 1;
      S = max(V - 1/beta, 0)./V;
      Wx = S.*Uxbar;
      Wy = S.*Uybar;
      Wz = S.*Uzbar;
    end
    
    % update parameters related to Wx, Wy, Wz
    [lam1,lam2,lam4,f,g2] = update_W(beta,Wx,Wy,Wz,Ux,Uy,Uz,sigmax,sigmay,sigmaz,...
      lam1,lam2,lam4,f,opts.TVnorm);
  end
  
  % update reference value
  Qp = Q; Q = gam*Qp + 1; C = (gam*Qp*C + f)/Q;
  uup = U - Up; uup = uup(:);                     % uup: p*q*r
  nrmuup = norm_fro_3D(uup);                      % nrmuup: constant
  
  out.res = [out.res; nrmuup];
  out.f = [out.f; f]; out.C = [out.C; C]; out.cnt = [out.cnt;cnt];
  out.lam1 = [out.lam1; lam1]; out.lam2 = [out.lam2; lam2]; 
  out.lam3 = [out.lam3; lam3];out.lam4 = [out.lam4; lam4];
  out.tau = [out.tau; tau]; out.alpha = [out.alpha; alpha];

  if ~isempty(Ut), out.n2re = [out.n2re; norm_fro_3D(U - Ut)/norm_fro_3D(Ut)]; end

  nrmup = norm_fro_3D(Up);
  RelChg = nrmuup/nrmup;

  % recompute gradient
  d = g2 + muDbeta*g - DtsAtd;
  
  if RelChg < tol_inn  && ~fst_itr
    count = count + 1;
    RelChgOut = norm_fro_3D(U-rcdU)/nrmup;
    out.reer = [out.reer; RelChgOut];
    rcdU = U;
    out.obj = [out.obj; f + lam4];
    if isempty(out.itrs)
      out.itrs = ii;
    else
      out.itrs = [out.itrs; ii - sum(out.itrs)];
    end

    % stop if already reached final multipliers
    if RelChgOut < tol_out || count > opts.maxcnt
      if opts.isreal
        U = real(U);
      end
      if exist('scl','var')
        U = U/scl;
      end
      out.itr = ii;
      fprintf('Number of total iterations is %d. \n',out.itr);
      return
    end
      
    % update multipliers
    [sigmax,sigmay,sigmaz,lam4,~] = update_mlp(beta,Wx,Wy,Wz,Ux,Uy,Uz,sigmax,sigmay,sigmaz,lam4,f);
    
    % update penality parameters for continuation scheme
    beta0 = beta;
    beta = beta*opts.rate_ctn;
    mu = mu*opts.rate_ctn;
    if beta > betaf; beta = betaf; end
    if mu > muf; mu = muf; end
    muDbeta = mu/beta;
    out.mus = [out.mus; mu]; out.betas = [out.betas; beta];

    % update function value, gradient, and relavent constant
    f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4;
    DtsAtd = -(beta0/beta)*d;     % DtsAtd should be divded by new beta instead of the old one for consistency!  
    d = g2 + muDbeta*g - DtsAtd;

    %initialize the constants
    gp = [];
    gam = opts.gam; Q = 1; C = f;
  end
end

if opts.isreal
  U = real(U);
end
if exist('scl','var')
  fprintf('Attain the maximum of iterations %d. \n',opts.maxit);
  U = U/scl;
end


function [lam2,lam3,lam4,f,g2,Au,g] = get_g(U,Ux,Uy,Uz,Wx,Wy,Wz,lam1,beta,mu,A,b,Atb,...
  sigmax,sigmay,sigmaz)

global Dt

% A*u 
Au = A(U(:),1);
g = A(Au,2) - Atb;
Vx = Ux - Wx;
Vy = Uy - Wy;
Vz = Uz - Wz;
lam2 = sum(sum(sum(Vx.*conj(Vx) + Vy.*conj(Vy)+ Vz.*conj(Vz))));
% g2 = D'(Du-w)
g2 = Dt(Vx,Vy,Vz);
Aub = Au-b;
lam3 = norm_fro_3D(Aub)^2;
lam4 = sum(sum(sum(conj(sigmax).*Vx + conj(sigmay).*Vy + conj(sigmaz).*Vz)));
f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4;


function [U,lam2,lam3,lam4,f,Ux,Uy,Uz,Au,g,g2] = update_g(p,q,r,lam1,alpha,beta,mu,...
  Up,du,gp,dg,g2p,dg2,Aup,dAu,Wx,Wy,Wz,Uxp,dUx,Uyp,dUy,Uzp,dUz,b,sigmax,sigmay,sigmaz)

g  = gp  + alpha*dg;
g2 = g2p + alpha*dg2;
U  = Up  + alpha*reshape(du,p,q,r);
Au = Aup + alpha*dAu;
Ux = Uxp + alpha*dUx;
Uy = Uyp + alpha*dUy;
Uz = Uzp + alpha*dUz;

Vx = Ux - Wx;
Vy = Uy - Wy;
Vz = Uz - Wz;
lam2 = sum(sum(sum(Vx.*conj(Vx) + Vy.*conj(Vy) + Vz.*conj(Vz))));
Aub = Au-b;
lam3 = norm_fro_3D(Aub)^2;
lam4 = sum(sum(sum(conj(sigmax).*Vx + conj(sigmay).*Vy + conj(sigmaz).*Vz)));
f = lam1 + beta/2*lam2 + mu/2*lam3 - lam4;


function [lam1,lam2,lam4,f,g2] = update_W(beta,Wx,Wy,Wz,Ux,Uy,Uz,sigmax,sigmay,sigmaz,...
  lam1,lam2,lam4,f,option)

global Dt

% update parameters because Wx, Wy, Wz were updated
tmpf = f - lam1 - beta/2*lam2 + lam4;
if option == 1
  lam1 = sum(sum(sum(abs(Wx) + abs(Wy) + abs(Wz))));
else
  lam1 = sum(sum(sum(sqrt(Wx.^2 + Wy.^2 + Wz.^2))));
end
Vx = Ux - Wx;
Vy = Uy - Wy;
Vz = Uz - Wz;
g2 = Dt(Vx,Vy,Vz);
lam2 = sum(sum(sum(Vx.*conj(Vx) + Vy.*conj(Vy) + Vz.*conj(Vz))));
lam4 = sum(sum(sum(conj(sigmax).*Vx + conj(sigmay).*Vy + conj(sigmaz).*Vz)));
f = tmpf + lam1 + beta/2*lam2 - lam4;


function [sigmax,sigmay,sigmaz,lam4,f] = update_mlp(beta,Wx,Wy,Wz,Ux,Uy,Uz,...
  sigmax,sigmay,sigmaz,lam4,f)

Vx = Ux - Wx;
Vy = Uy - Wy;
Vz = Uz - Wz;
sigmax = sigmax - beta*Vx;
sigmay = sigmay - beta*Vy;
sigmaz = sigmaz - beta*Vz;

tmpf = f + lam4;
lam4 = sum(sum(sum(conj(sigmax).*Vx + conj(sigmay).*Vy + conj(sigmaz).*Vz)));
f = tmpf - lam4;


function [U,mu,beta] = ftvcs_al_init(p,q,r,Atb,scl,opts)

% initialize mu beta
if isfield(opts,'mu0')
  mu = opts.mu0;
else
  error('Initial mu is not provided.');
end
if isfield(opts,'beta0')
  beta = opts.beta0;
else
  error('Initial beta is not provided.');
end

% initialize U
[mm,nn,oo] = size(opts.init);
if max(max(mm,nn),oo) == 1
  switch opts.init
    case 0, U = zeros(p,q,r);
    case 1, U = reshape(Atb,p,q,r);
  end
else
  U = opts.init*scl;
  if mm ~= p || nn ~= q || oo ~= r
    fprintf('Input initial guess has incompatible size! Switch to the default initial guess.\n');
    U = reshape(Atb,p,q,r);
  end
end


function val = norm_fro_3D(A)

val = sqrt(sum(sum(sum(abs(A).^2))));

