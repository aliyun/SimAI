function g = loss_COST231_large_cities_urban(d, hb, hm, f)

  %%
  %% function g = loss_COST231_large_cities_urban(d, c)
  %%
  %% returns the loss at d meters for f frequency and mobile height m and
  %% base station height of hb
  
  assert(isscalar(f));
  assert(f > 0);

  Fhm = 3.2*(log10(11.75*hm))^2;
  C = 3;
  g = zeros(size(d));
  g(find(d > 0)) = 46.3 + (33.9*log10(f)) - (13.82*log10(hb)) + (44.9-(6.55*log10(hb))).*log10(d) - Fhm + C;

  g(find(d <= 0)) = 1;
  
       
  