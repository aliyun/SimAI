
clear all;
close all;


earfcn = 500;

noise_psd = 1.38e-23 * 290;

for nrbs = [6 15]
  
  for noise_figure_dB = [0 5 10 ]

    name = ["nfdB" num2str(noise_figure_dB, "%d") "earfcn" num2str(earfcn,"%d") "nrb" num2str(nrbs, "%d")];

    npsdv = ones (1,nrbs) * (noise_psd * 10.^(noise_figure_dB/10));
    
    printf("SpectrumValue %s (LteSpectrumValueHelper::GetSpectrumModel (%d, %d));\n",
	   name, earfcn, nrbs);
    
    print_C_vector (npsdv, name);
    printf("AddTestCase (new LteNoisePsdTestCase (\"%s\", %d, %d, %f, %s));\n\n", name, earfcn, nrbs, noise_figure_dB, name); 
        
  endfor

endfor




  






    