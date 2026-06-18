
clear all;
close all;


earfcn = 500;



for run = 1:2
  
  for nrbs = [6 25 100]

    bw = nrbs * 180000;
    
    for txpowdBm = [10 30]

      basename = ["txpowdB" num2str(txpowdBm, "%d") "nrb" num2str(nrbs, "%d") "run" num2str(run, "%d") "earfcn" num2str(earfcn,"%d")];
      svname = [ "spectrumValue_" basename];
      arname = [ "activeRbs_" basename];

      activeRbsMask = round (rand (1,nrbs));
      activeRbs = find (activeRbsMask) - 1;
      txpowW = (10.^(txpowdBm/10))/1000;
      txpsd = txpowW.*activeRbsMask./bw;

      printf("std::vector<int> %s (%d);\n", arname, length(activeRbs));
      print_C_vector (activeRbs, arname);
            
      printf("SpectrumValue %s (LteSpectrumValueHelper::GetSpectrumModel (%d, %d));\n",
	     svname, earfcn, nrbs);
      print_C_vector (txpsd, svname);
      
      printf("AddTestCase (new LteTxPsdTestCase (\"%s\", %d, %d, %f, %s, %s));\n\n",
	     basename, earfcn, nrbs, txpowdBm, arname, svname); 
      
    endfor

  endfor

endfor


  






    