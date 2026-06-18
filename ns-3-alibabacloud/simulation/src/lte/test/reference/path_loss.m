function g = path_loss(d, f)

  %%
  %% function g = path_loss(d, c)
  %%
  %% returns the path-loss at d meters for channel c
  
  assert(isscalar(f));
  assert(f > 0);

  g = zeros(size(d));
  c = 299792458.0;
  g(find(d > 0)) = (c./(4.0.*pi.*f.*d(find(d > 0)))).^2;

  g(find(d <= 0)) = 1;
  
       
  
