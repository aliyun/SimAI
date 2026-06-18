function g = loss_Kun_2_6GHz (d)

  %%
  %% function g = loss_Kun_2_6GHz (d, c)
  %%
  %% returns the loss at d meters for f frequency and mobile height m and
  %% base station height of hb
  
  %assert(isscalar(f));
  %assert(f > 0);


  g = zeros(size(d));
  g(find(d > 0)) = 36 + 26.*log10(d);

  g(find(d <= 0)) = 1;
  