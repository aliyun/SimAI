
clear all;
close all;

for nrbs = [6 15 25 50 75 100]

%%  earfcn = 500;
%%  fcMHz = 2160;

  earfcn = 19400;
  fcMHz = 1730;
    
  fc = fcMHz * 1e6;
  
  rbbw = 180e3;
  bw = rbbw * nrbs;
  
  fcv = linspace (fc - bw/2 + rbbw/2, fc + bw/2 - rbbw/2, nrbs);
  name = ["fc" num2str(fcMHz, "%d") "nrb" num2str(nrbs, "%d")];
  printf("std::vector<double> %s (%s);\n",  name, num2str (length(fcv)));
  print_C_vector (fcv, name);
  printf("AddTestCase (new LteSpectrumModelTestCase (\"%s\", %d, %d, %s));\n\n", name, earfcn, nrbs, name);
  
  
endfor



  






    