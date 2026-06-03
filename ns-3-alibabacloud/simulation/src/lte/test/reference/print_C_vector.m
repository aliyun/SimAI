function print_C_vector (x, name)
  %%
  %% print_C_vector (x)
  %%
  %% prints to screen the C code that is needed to initialize vector x
  %% x is the data
  %% name is the name of the C variable


  assert (isvector(x));
  for jj = 1:length(x)
    printf("%s[%d] = %15.12e;\n",  name, jj-1, x(jj));
  endfor

endfunction

	  
	  

  