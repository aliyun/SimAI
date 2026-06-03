clear all;
close all;

%% LTE from theory to practice
%% Table 22.7 Reference sensitivity.

fdl = 2120e6;  % DL carrier freq Hz, EARFCN = 100
ful = 1930e6;  % UL carrier freq Hz, EARFCN = 18100 

nrbsControl = 25; % tx bandwidth configuration in number of RBs
bwControl = nrbsControl * 180000; % bandwidth in Hz, note that this is smaller than
		    % the nominal Channel Bandwdith, see TS 36.101 fig 5.6-1

nrbsData = 12; % tx bandwidth configuration in number of RBs
bwData = nrbsData * 180000; % bandwidth in Hz, note that this is smaller than
		    % the nominal Channel Bandwdith, see TS 36.101 fig 5.6-1

kT = -174; % noise PSD in dBm / Hz
ndBmData = kT + 10*log10(bwData);  % noise power dBm
ndBmControl = kT + 10*log10(bwControl);  % noise power dBm

dlpdBm = 30;  % tx power dBm in DL
dlp = 10.^((dlpdBm - 30)/10); %% tx pow in W in DL
dlnf = 9; % receiver noise figure in dB in DL
dlnControl = 10.^((ndBmControl-30+dlnf)/10); %% noise in W in DL
dlnData = 10.^((ndBmData-30+dlnf)/10); %% noise in W in DL

ulpdBm = 10;  % tx power dBm in UL
ulp = 10.^((ulpdBm - 30)/10); %% tx pow in W in UL
ulnf = 5; % receiver noise figure in dB in UL
uln = 10.^((ndBmControl-30+ulnf)/10); %% noise in W in UL

ber = 0.00005;
gamma = -log (5*ber)./1.5;


%% distances
%%           d1                   d2  
%%  eNB1-------------UE1-UE2-----------eNB2
%%

d1 = 500;
d2 = 500;

%% propagation gains (linear)
%%           g11dl             g21dl 
%%  eNB1-------------UE1-UE2-----------eNB2
%%           g11ul             g21ul 
%%              

g11dl = gain_freespace (d1, fdl);
g11ul = gain_freespace (d1, ful);
g21dl = gain_freespace (d2, fdl);
g21ul = gain_freespace (d2, ful);

%% SINR (linear)
privateRbDlsinr = dlp*g11dl / (dlnControl);
privateRbUlsinr = ulp*g11ul / (ulp*g21ul + uln); %% new CQI generation approach is only for DL, so in UL CQI is the same
commonRbDlsinr = dlp*g11dl*24 / (dlp*g21dl*12 + dlnData);
commonRbUlsinr = ulp*g11ul / (ulp*g21ul + uln);

%% SINR (dB)
privateRbDlsinrdB = 10.*log10(privateRbDlsinr);
privateRbUlsinrdB = 10.*log10(privateRbUlsinr);
commonRbDlsinrdB = 10.*log10(commonRbDlsinr);
commonRbUlsinrdB = 10.*log10(commonRbUlsinr);

%% Spectal Efficiency
privateRbDlse = log2(1 + privateRbDlsinr./gamma)
privateRbUlse = log2(1 + privateRbUlsinr./gamma)
commonRbDlse = log2(1 + commonRbDlsinr./gamma)
commonRbUlse = log2(1 + commonRbUlsinr./gamma)

%% to get the MCS, you need to do a manual lookup into 3GPP R1-081483
%% starting from the spectral efficiency value.
%% See the Testing section in the LTE module documentation for more info
%% on how this is done. You might as well look into lte_amc.m

commonRbDlMsc = 4;
commonRbUlMsc = 2;

privateRbDlMsc = 28;
privateRbUlMsc = 2; %% new CQI generation approach is only for DL, so in UL CQI is the same

printf("AddTestCase (new LteCqiGenerationTestCase (\"UsePdcchForCqiGeneration\", \"true\", %d, %d , ));\n", commonRbDlMsc, commonRbUlMsc)
printf("AddTestCase (new LteCqiGenerationTestCase (\"UsePdcchForCqiGeneration\", \"false\", %d, %d , ));\n", privateRbDlMsc, commonRbUlMsc)
