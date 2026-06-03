function g = gain_freespace(d, f)

  %%
  %% function g = gain_freespace(d, c)
  %%
  %% returns the gain at d meters for channel c
  
  assert(isscalar(f));
  assert(f > 0);

  g = zeros(size(d));
  g(find(d > 0)) = (3e8./(4.0.*pi.*f.*d(find(d > 0)))).^2;

  g(find(d <= 0)) = 1;
  
       
  