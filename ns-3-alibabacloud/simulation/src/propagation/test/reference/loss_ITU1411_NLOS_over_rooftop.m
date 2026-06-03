function g = loss_ITU1411_NLOS_over_rooftop (d, hb, hm, hr, f, l, b, st_w, phi, big)

  %%
  %% function g = loss_ITU1411_NLOS_over_rooftop(d, hb, hm, hr, f, l, b, st_w, phi, big)
  %%
  %% returns the loss at d meters for f frequency and mobile height m and
  %% base station height of hb
  
  assert(isscalar(f));
  assert(f > 0);


  lambda = 299792458 / f;
  fmhz = f/1e6;
  Dhb = hb - hr;
  ds = (lambda*d*d)/(Dhb*Dhb);
  if (l>ds)
    if (fmhz>2000)
      kf = -8;
    else
      if (big==1)
        kf = -4+1.5*((fmhz/925)-1);
      else
        kf = -4+0.7*((fmhz/925)-1);
      endif
    endif
    if (hb>hr)
      kd = 18;
      if (fmhz>2000)
         ka = 71.4;
      else 
         ka = 54;
      endif
      Lbsh = -18*log10(1+Dhb);
    else
      kd = 18 - 15*(Dhb/hr);
      Lbsh = 0;
      if (d>=500)
        ka = 54-0.8*Dhb;
      else
        ka = 54-1.6*Dhb;
      endif
    endif
    Lmsd = Lbsh + ka + kd.*log10(d./1000) + kf*log10(fmhz) - 9*log10(b);
  else
    theta = atan (Dhb /b);
    rho = sqrt(Dhb^2 + b^2);
    if (abs(hb-hr)<1)
       Qm = b./d;
    elseif (hb>hr)
       Qm = 2.35*((Dhb./d)*sqrt(b/lambda))^0.9;
    else
       Qm = (b/(2*pi.*d))*sqrt(lambda/rho)*((1/theta)-(1/(2*pi+theta)));
       b
    endif
    Lmsd = -10*log10(Qm*Qm);
  endif
  Dhm = hr-hm;
  if (phi<35)
    Lori = -10+(0.354*phi);
  elseif (phi<55)
    Lori = 2.5+0.075*(phi-35);
  else
    Lori = 4- 0.114*(phi-55);
  endif
  Lrts = -8.2 -10*log10(st_w) + 10*log10(fmhz) + 20*log10(Dhm) + Lori;
  
  Lbf = 32.4 +20*log10(d/1000) + 20*log10(fmhz);
  g = zeros(size(d));
  if (Lrts+Lmsd>0)
    L = Lbf + Lrts + Lmsd;
  else
    L = Lbf
  endif
  g(find(d > 0)) = L;

  g(find(d <= 0)) = 1;
  