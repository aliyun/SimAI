function g = loss_ITU1411_LOS (d, hb, hm, f)

  %%
  %% function g = loss_ITU1411_LOS(d, hb, hm, f)
  %%
  %% returns the loss at d meters for f frequency and mobile height m and
  %% base station height of hb
  
  assert(isscalar(f));
  assert(f > 0);


  lambda = 299792458 / f;
  Lbp = abs(20*log10(lambda^2/(8*pi*hb*hm)));
  Rbp = (4*hb*hm) / lambda;
  if (d<=Rbp)
    Ll = Lbp + 20.*log10(d./Rbp);
    Lu = Lbp + 20 + 25.*log10(d./Rbp);
  else
    Ll = Lbp + 40.*log10(d./Rbp);
    Lu = Lbp + 20 + 40.*log10(d./Rbp);
  endif

  g = zeros(size(d));
  g(find(d > 0)) = (Ll.+Lu)./2;

  g(find(d <= 0)) = 1;
  