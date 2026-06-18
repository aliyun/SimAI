clear all;
close all;

%% LTE from theory to practice
%% Table 22.7 Reference sensitivity.

fdl = 2120e6;  % DL carrier freq Hz, EARFCN = 100
ful = 1930e6;  % UL carrier freq Hz, EARFCN = 18100
nrbs = 25; % tx bandwdith configuration in number of RBs
nres = nrbs * 12;
bw = 180000 / 12; % bandwidth per RE in Hz
%%bwtot = xxx %%, note that this is smaller than the nominal Channel Bandwdith, see TS 36.101 fig 5.6-1
kT = -174; % noise PSD in dBm / Hz
ndBm = kT + 10*log10(bw);  % noise power dBm for a RE

dlpdBm = 30;  % tx power in dBm in DL
dlp = (10.^((dlpdBm - 30)/10))/nres %% tx pow per RE in W in DL
dlnf = 9; % receiver noise figure in dB in DL
dln = 10.^((ndBm-30+dlnf)/10); %% noise per RE in W in DL

ulpdBm = 10;  % tx power dBm in UL
ulp = 10.^((ulpdBm - 30)/10); %% tx pow in W in UL
ulnf = 5; % receiver noise figure in dB in UL
uln = (10.^((ndBm-30+ulnf)/10)); %% noise in W per RE in UL

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

%d1 = 5400;
%%for d1 = 50;


rsrpdBmv1 = [];
rsrqdBv1 = [];
sinrdBv1 = [];

%for d1 = [10 20 50 100 200 500 1000 10000 100000 1000000]
for d1 = [10 20 50 100 200 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000]

  %%for d1 = [10]
%% for d2 = [10 100 1000 10000 100000 1000000]

%for d2 = [10 20 50 100 200 500 1000 10000 100000 1000000]
for d2 = [10000]

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


  %% RSRP (linear)
  rsrp1 = g11dl.*dlp;
  rsrp2 = g21dl.*dlp;

  %% RSRP (dBm)
  rsrp1dBm = 10.*log10(1000*rsrp1);
  rsrp2dBm = 10.*log10(1000*rsrp2);

  %% RSSI (linear)
  rssi1 = 2*(dlp*g11dl + dlp*g21dl + dln)*nrbs;
  rssi2 = 2*(dlp*g11dl + dlp*g21dl + dln)*nrbs;

  %% RSRQ (linear)
  rsrq1 = (nrbs * rsrp1) / rssi1;
  rsrq2 = (nrbs * rsrp2) / rssi2;

  %% RSRQ (dB)
  rsrq1dB = 10.*log10(rsrq1);
  rsrq2dB = 10.*log10(rsrq2);

  %% SINR (linear)
  dlsinr = dlp*g11dl / (dlp*g21dl + dln);
%    ulsinr = ulp*g11ul / (ulp*g21ul + uln);

  %% SINR (dB)
  dlsinrdB = 10.*log10(dlsinr);
%    ulsinrdB = 10.*log10(ulsinr);
  
  printf("AddTestCase (new LteUeMeasurementsTestCase (\"d1=%d, d2=%d\", % f, %f, % f, %f, % f, %f));\n", \
   d1, d2, d1, d2, rsrp1dBm, rsrp2dBm, rsrq1dB, rsrq2dB)


  rsrpdBmv1 = [rsrpdBmv1 rsrp1dBm];
  rsrqdBv1 = [rsrqdBv1 rsrq1dB];
  sinrdBv1 = [sinrdBv1 dlsinrdB];

  
endfor


endfor

plot (sinrdBv1, rsrpdBmv1, ";mapping;", "linewidth", 3,
      [min(sinrdBv1) max(sinrdBv1)], [-140 -140], ";min RSRP;", "linewidth", 3,
      [min(sinrdBv1) max(sinrdBv1)], [-44 -44], ";max RSRP;", "linewidth", 3);
xlabel("SINR (dB)");
ylabel("RSRP (dBm)");
axis([-40 60 -150 -30]);
grid on
print -dpng "rsrp.png"

figure;


plot (sinrdBv1, rsrqdBv1, ";mapping;", "linewidth", 3,
      [min(sinrdBv1) max(sinrdBv1)], [-19.5 -19.5], ";min RSRQ;", "linewidth", 3,
      [min(sinrdBv1) max(sinrdBv1)], [-3 -3], ";max RSRQ;", "linewidth", 3);
xlabel("SINR (dB)");
ylabel("RSRQ (dB)");
axis([-40 60 -30 5]);
grid on
print -dpng "rsrq.png"