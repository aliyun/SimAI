clear all;
close all;

fdl = 2120e6;  % DL carrier freq Hz, EARFCN = 100
ful = 1930e6;  % UL carrier freq Hz, EARFCN = 18100 
nrbs = 25; % tx bandwidth configuration in number of RBs
dlpdBm = 30;  % tx power dBm in DL, referenceSignalPower
dlp = 10.^((dlpdBm - 30)/10); %% tx pow in W in DL


%%DOWNLINK
P_A = 0;   %% << change this value to simulate Downlink Power Allocation
PdschPower = 10.^((dlpdBm+ P_A - 30)/10);
printf("Expexted Pdsch Power: %f \n", PdschPower)


%%UPLINK 
%% Uplink Power Control Parameters
Pcmax = 23;
Pcmin = -40;
MPusch = 6; %%number of RB used for UL transmission
PoNominalPusch = -80;
PoUePusch = 0;
PsrsOffset = 7;
alpha = 1;
deltaTF = 0;
PoPusch = PoNominalPusch + PoUePusch;

%% Closed Loop Mode
%% change fc value to simulate closed loop mode
%%TS36.213 Table 5.1.1.1-2
%%   TPC   |   Accumulated Mode  |  Absolute Mode
%%------------------------------------------------
%%    0    |         -1          |      -4
%%    1    |          0          |      -1
%%    2    |          1          |       1
%%    3    |          3          |       4
%%------------------------------------------------


%% distances
%%                d1  
%%  eNB1---------------------UE1
%%
d1 = 200;
fc = -1;

%% different FR schemes have different minimal Uplink sub-bandwidth
MPusch = 6;

%% pathloss
%%             pathloss  
%%  eNB1---------------------UE1
%%         

pathloss = path_loss (d1, fdl);
pathlossdB = 10.*log10(pathloss);
pathlossdB = -pathlossdB;

%% PUSCH TX POWER 
PuschPowerdB = 10*log10(MPusch) + PoPusch + alpha * pathlossdB + deltaTF + fc;

if (PuschPowerdB > Pcmax)
  PuschPowerdB = Pcmax;
endif

if (PuschPowerdB < Pcmin) 
  PuschPowerdB = Pcmin;
endif

PuschPower = 10.^((PuschPowerdB - 30 )/10);
printf("Expexted Pusch Power: %f \n", PuschPower)

