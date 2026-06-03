clear all;
close all;

%% LTE from theory to practice
%% Table 22.7 Reference sensitivity.

fdl = 2120e6;  % DL carrier freq Hz, EARFCN = 100
ful = 1930e6;  % UL carrier freq Hz, EARFCN = 18100 
nrbs = 25; % tx bandwidth configuration in number of RBs
bw = nrbs * 180000; % bandwidth in Hz, note that this is smaller than
		    % the nominal Channel Bandwdith, see TS 36.101 fig 5.6-1
kT = -174; % noise PSD in dBm / Hz
ndBm = kT + 10*log10(bw);  % noise power dBm

dlpdBm = 30;  % tx power dBm in DL
dlp = 10.^((dlpdBm - 30)/10); %% tx pow in W in DL
dlnf = 9; % receiver noise figure in dB in DL
dln = 10.^((ndBm-30+dlnf)/10); %% noise in W in DL

ulpdBm = 10;  % tx power dBm in UL
ulp = 10.^((ulpdBm - 30)/10); %% tx pow in W in UL
ulnf = 5; % receiver noise figure in dB in UL
uln = 10.^((ndBm-30+ulnf)/10); %% noise in W in UL

ber = 0.00005;
gamma = -log (5*ber)./1.5;



%% distances
%%         d2  
%%  UE1-----------eNB2
%%   |             |
%% d1|             |d1
%%   |     d2      |
%%  eNB1----------UE2
%%

d1 = 50;
for d2 = [20 50 200 500]

  %% propagation gains (linear)
  %%             g21dl  
  %%      UE1<----------eNB2
  %%      ^ |              
  %%g11dl1| |g11ul          
  %%      | v            
  %%      eNB1<---------UE2
  %%             g21ul

  g11dl = gain_freespace (d1, fdl);
  g11ul = gain_freespace (d1, ful);
  g21dl = gain_freespace (d2, fdl);
  g21ul = gain_freespace (d2, ful);

  %% SINR (linear)
  privateRbDlsinr = dlp*g11dl / (dln);
  privateRbUlsinr = ulp*g11ul / (uln);
  commonRbDlsinr = dlp*g11dl / (dlp*g21dl + dln);
  commonRbUlsinr = ulp*g11ul / (ulp*g21ul + uln);

  %% SINR (dB)
  privateRbDlsinrdB = 10.*log10(privateRbDlsinr);
  privateRbUlsinrdB = 10.*log10(privateRbUlsinr);
  commonRbDlsinrdB = 10.*log10(commonRbDlsinr);
  commonRbUlsinrdB = 10.*log10(commonRbUlsinr);


  %%printf("AddTestCase (new LteInterferenceHardFrTestCase (\"d1=%d, d2=%d\", % f, %f, %f, %f , ));\n", d1, d2, d1, d2, privateRbDlsinr, privateRbUlsinr)

  printf("AddTestCase (new LteInterferenceStrictFrTestCase (\"d1=%d, d2=%d\", % f, %f, %f, %f , %f, %f , ));\n", d1, d2, d1, d2, commonRbDlsinr, commonRbUlsinr, privateRbDlsinr, privateRbUlsinr)
  
endfor
