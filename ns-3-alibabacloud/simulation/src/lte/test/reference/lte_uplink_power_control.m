clear all;
close all;

fdl = 2120e6;  % DL carrier freq Hz, EARFCN = 100
nrbs = 25; % tx bandwidth configuration in number of RBs
dlpdBm = 30;  % tx power dBm in DL, referenceSignalPower
dlp = 10.^((dlpdBm - 30)/10); %% tx pow in W in DL ,referenceSignalPower

%% Uplink Power Control Parameters
Pcmax = 23;
Pcmin = -40;
MPusch = 25; %%number of RB used for UL transmission
PoNominalPusch = -90;
PoUePusch = 0;
PsrsOffset = 9;
alpha = 1;
deltaTF = 0;
PoPusch = PoNominalPusch + PoUePusch;
accumulatedMode = false;

%% Open Loop Power Control
%%d1 = 1000;
%%fc = 0;


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

%%Absolute Mode 
d1 = 100;
fc = -4;


%%Accumulated Mode, steps to simulate fc current value
%%d1 = 10;
%%fc = 0;
%%accumulatedMode = true;

stepNumToExecute = 12;
fcValueSequence = [
		[0, 0]; 
		[-1, 20]; 
		[-1, 20];
		[1, 1];
		[3, 10];
		[1, 15];
		[3, 1];
		[1, 10];
		[1, 10];
		[-1, 1];
		[-1, 20];
		[1, 5]
                ];

%% distances
%%                d1  
%%  eNB1---------------------UE1
%%

%% pathloss
%%             pathloss  
%%  eNB1---------------------UE1
%%         
%%

pathloss = path_loss (d1, fdl);
pathlossdB = 10.*log10(pathloss);
pathlossdB = -pathlossdB;

%% RSRP (dB)
rsrqdB = dlpdBm - pathlossdB;
rsrqdBm = rsrqdB;

if (accumulatedMode) 
  for i = 1:stepNumToExecute
    curFcValue = fcValueSequence(i,1);
    curFcNum = fcValueSequence(i,2);

    for i = 1:curFcNum
      PuschPower = 10*log10(MPusch) + PoPusch + alpha * pathlossdB + deltaTF + fc;
      if ((PuschPower > Pcmax && curFcValue> 0) || (PuschPower < Pcmin && curFcValue < 0))
        %% do not accumulate
      else
        fc = fc + curFcValue;
      endif
    endfor
  endfor
endif

printf("TxPower: %f\n", dlpdBm)
printf("Distance: %f\n", d1)
printf("PathLoss: %f \n",pathlossdB)
printf("RSRP [dBm]: %f \n", rsrqdBm)
printf("fc: %f \n", fc)

%% PUSCH TX POWER 
PuschPower = 10*log10(MPusch) + PoPusch + alpha * pathlossdB + deltaTF + fc;

if (PuschPower > Pcmax)
  PuschPower = Pcmax;
endif

if (PuschPower < Pcmin) 
  PuschPower = Pcmin;
endif

printf("PuschPower: %f \n", PuschPower)


%% PUCCH TX POWER 
PucchPower = PuschPower;
printf("PucchPower: %f \n", PucchPower)


%% SRS TX POWER 
PsrsOffsetValue = -10.5 + PsrsOffset * 1.5;
SrsPower = PsrsOffsetValue + 10*log10(nrbs) + PoPusch + alpha * pathlossdB + fc;

if (SrsPower > Pcmax)
  SrsPower = Pcmax;
endif

if (SrsPower < Pcmin) 
  SrsPower = Pcmin;
endif
printf("SrsPower: %f \n", SrsPower)

