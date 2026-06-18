
clear all;
close all;

snr_db = (-5:30)';
snr = (10.^(snr_db./10));


%% this model is from:
%% G. Piro, N. Baldo. M. Miozzo, "An LTE module for the ns-3 network simulator",
%% WNS3 2011 (in conjunction with SimuTOOLS 2011)
%% which cites this one:
%% "A Proportional-Fair Power Allocation Scheme for Fair and Efficient Multiuser OFDM Systems"


ber = 0.00005;
gamma = -log (5*ber)./1.5
spectral_efficiency_piro2011 = log2(1 + snr./gamma);



% %% this eventually would be an alternative model from:
% %% Preben Mogensen et al., "LTE Capacity compared to the Shannon Bound"
% %% IEEE VTC Spring 2007
%
% snr_eff = 1.25;
% bw_eff_times_eta = 0.75;
% spectral_efficiency_mogensen2007= bw_eff_times_eta .* log2(1 + snr./snr_eff);
%
% plot (snr_db, spectral_efficiency_piro2011, ";piro 2011;",
%       snr_db, spectral_efficiency_mogensen2007, ";morgensen2007;");


[snr_db spectral_efficiency_piro2011]

%% 
%% now that we got the spectral efficiency for each value of SNR in dB
%% you should do the following:
%% we look up (manually) into the XLS sheet annexed to 3GPP R1-081483 "Conveying MCS
%% and TB size via PDCCH". Look at the tab "MCS Table", quantize the
%% spectral efficiency based on the CQI (rounding to the lowest value), and get the corresponding MCS
%% scheme (i.e., the MCS index that appears on the same line looking at
%% the MCS table on the right). Note that the quantization of the CQI is
%% coarser than the spectral efficiency reported in the CQI table.
%% Finally, note that there are some discrepancies between the MCS index
%% in R1-081483 and that indicated by the standard: TS 36.213 Table
%% 7.1.7.1-1 says that the MCS index goes from 0 to 31, and 0 appears to
%% be a valid MCS scheme (TB size is not 0) but in R1-081483 the first useful MCS index is 1.
%% Hence to get the value as intended by the standard we need to
%% subtract 1 from the index reported in R1-081483.

%% the resulting values after the manual lookup are reported here:

%% SNR (dB)     sp. eff  MCS index

%%  -5.00000    0.08024  -1
%%  -4.00000    0.10030  -1
%%  -3.00000    0.12518  -1
%%  -2.00000    0.15589  0
%%  -1.00000    0.19365  0
%%   0.00000    0.23983  2
%%   1.00000    0.29593  2
%%   2.00000    0.36360  2
%%   3.00000    0.44451  4
%%   4.00000    0.54031  4
%%   5.00000    0.65251  6
%%   6.00000    0.78240  6
%%   7.00000    0.93086  8
%%   8.00000    1.09835  8
%%   9.00000    1.28485  10
%%  10.00000    1.48981  12
%%  11.00000    1.71229  12
%%  12.00000    1.95096  14
%%  13.00000    2.20429  14
%%  14.00000    2.47062  16
%%  15.00000    2.74826  18
%%  16.00000    3.03560  18
%%  17.00000    3.33115  20
%%  18.00000    3.63355  20
%%  19.00000    3.94163  22
%%  20.00000    4.25439  22
%%  21.00000    4.57095  24
%%  22.00000    4.89060  24
%%  23.00000    5.21276  26
%%  24.00000    5.53693  26
%%  25.00000    5.86271  28
%%  26.00000    6.18980  28
%%  27.00000    6.51792  28
%%  28.00000    6.84687  28
%%  29.00000    7.17649  28
%%  30.00000    7.50663  28



