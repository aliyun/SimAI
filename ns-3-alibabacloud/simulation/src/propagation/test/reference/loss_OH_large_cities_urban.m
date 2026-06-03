function g = loss_OH_large_cities_urban(d, hb, hm, f)

  %%
  %% function g = loss_OH_large_cities_urban(d, hb, hm, f)
  %%
  %% returns the loss at d meters for f frequency and mobile height m and
  %% base station height of hb
  
  assert(isscalar(f));
  assert(f > 0);

  if (f<200)
    Ch = 8.29*(log10(1.54*hm))^2-1.1;
  else
    Ch = 3.2*(log10(11.75*hm))^2-4.97;
  endif

  g = zeros(size(d));
  g(find(d > 0)) = 69.55 + (26.16*log10(f)) - (13.82*log10(hb)) + (44.9-(6.55*log10(hb))).*log10(d) - Ch;

  g(find(d <= 0)) = 1;
  
       
  