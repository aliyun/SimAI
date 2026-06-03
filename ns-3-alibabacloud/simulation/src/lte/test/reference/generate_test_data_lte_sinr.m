
clear all;
close all;

format long;


% Bandwidth in Hz
BW = [20e6 22e6];

% Powers of the noise and interferences in dBm
ndBm  = [-80 -80];
i1dBm = [-70 -75];
i2dBm = [-50 -49];
i3dBm = [-55 -58];
i4dBm = [-58 -59];

% Powers in W
n  = 1e-3 *10.^(ndBm./10);
i1 = 1e-3 *10.^(i1dBm./10);
i2 = 1e-3 *10.^(i2dBm./10);
i3 = 1e-3 *10.^(i3dBm./10);
i4 = 1e-3 *10.^(i4dBm./10);

% print Power Spectral Density
for jj = 1:2
  printf("(*noisePsd)[%d] = %15.12e;\n",  jj-1, n(jj)./BW(jj));
endfor

for jj = 1:2
  printf("(*i1)[%d] = %15.12e;\n", jj-1, i1(jj)./BW(jj));
  printf("(*i2)[%d] = %15.12e;\n", jj-1, i2(jj)./BW(jj));
  printf("(*i3)[%d] = %15.12e;\n", jj-1, i3(jj)./BW(jj));
  printf("(*i4)[%d] = %15.12e;\n", jj-1, i4(jj)./BW(jj));
endfor

% starting times and durations
ts  = 1;
ds  = 1;
ti1 = 0;
di1 = 3;
ti2 = 0.7;
di2 = 1;
ti3 = 1.2;
di3 = 1;
ti4 = 1.5;
di4 = 0.1;


% Signal powers in dBm (in columns)
for sdBm_all = [ -46 -63 ; -48 -61 ]

  sdBm = sdBm_all';

  % Power in W
  s  = 1e-3 *10.^(sdBm./10);

  for jj = 1:2
    printf("(*rxPsd)[%d] = %15.12e;\n",  jj-1, s(jj)./BW(jj));
  endfor

  % the above results in five intervals A, B, C, D, E in which the SINR remains constant
  sinrA = s./(n + i1 + i2);
  dA = 0.2;
  sinrB = s./(n + i1 + i2 + i3);
  dB = 0.3;
  sinrC = s./(n + i1 +i2 + i3 + i4);
  dC = 0.1;
  sinrD = s./(n + i1 + i2 + i3);
  dD = 0.1;
  sinrE = s./(n + i1 + i3);
  dE = 0.3;

  dT = dA + dB + dC + dD + dE;
  sinrT = ( sinrA * dA + sinrB * dB + sinrC * dC + sinrD * dD + sinrE * dE ) / dT;

  for jj = 1:2
    printf("(*theoreticalSinr)[%d] = %16.16f;\n",  jj-1, sinrT(jj));
  endfor

endfor
