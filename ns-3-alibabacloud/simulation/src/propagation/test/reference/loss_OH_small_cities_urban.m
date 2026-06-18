function g = loss_OH_small_cities_urban(d, hb, hm, f)

  %%
  %% function g = loss_OH_small_cities_urban(d, hb, hm, f)
  %%
  %% returns the loss at d meters for f frequency and mobile height m and
  %% base station height of hb
  
  assert(isscalar(f));
  assert(f > 0);

  Ch = 0.8 + (1.1*log10(f)-0.7)*hm -(1.56*log10(f));

  g = zeros(size(d));
  g(find(d > 0)) = 69.55 + (26.16*log10(f)) - (13.82*log10(hb)) + (44.9-(6.55*log10(hb))).*log10(d) - Ch;

  g(find(d <= 0)) = 1;
  
       
  