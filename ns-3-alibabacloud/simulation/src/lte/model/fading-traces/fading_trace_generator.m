clear all;
close all;

c = 3e8; 
% Excess taps delay (according to 3GPP TS 36.104 Annex B.2)
delays_pedestrianEPA = [0 30e-9 70e-9 90e-9 120e-9 190e-9 410e-9];
delays_vehicularEVA = [0 30e-9 150e-9 310e-9 370e-9 710e-9 1090e-9 1730e-9 2510e-9];
delays_urbanETU = [0 50e-9 120e-9 200e-9 230e-9 500e-9 1600e-9 2300e-9 5000e-9];
Td = 100/c;

% Realtive power of taps (according to 3GPP TS 36.104 Annex B.2)
power_pedestrianEPA = [0.0 -1.0 -2.0 -3.0 -8.0 -17.2 -20.8];
power_vehicularEVA = [0.0 -1.5 -1.4 -3.6 -0.6 -9.1 -7.0 -12.0 -16.9];
power_urbanETU = [-1.0 -1.0 -1.0 0.0 0.0 0.0 -3.0 -5.0 -7.0];


% DL EARFCN=500
% fc = 2160e6; %

% UL  EARFCN=18100
fc = 1930e6;

lambda = c/fc;
v_km_h = 3.0;  % speed of mobile node
v_m_s = v_km_h / 3.6;
fd = v_m_s / lambda; % doppler shift




% when working with an FFT, the normalized frequency w is
% w = 2 * pi * (f/fs) * t
% hence the max normalized frequency w=2*pi corresponds to f = fs,
% hence fs is also the max frequecy of our PowerSpectralDensity
fs = 20e6; 

% sampling period must be determined corresponding to the sampling
% frequency, because of the properties of an FFT
% in other words, if ts = 1/fs, then the samples of the FFT will be
% spaced by fs (which is what we want)
ts = 1/fs; % sampling period (i.e., 1 subframe duration)


% create the channel object
c = comm.RayleighChannel('SampleRate', 1/ts, 'MaximumDopplerShift', fd, 'PathDelays', delays_pedestrianEPA, 'AveragePathGains', power_pedestrianEPA);
%c = comm.RayleighChannel('SampleRate', 1/ts, 'MaximumDopplerShift', fd, 'PathDelays', delays_vehicularEVA, 'AveragePathGains', power_vehicularEVA);
%c = comm.RayleighChannel('SampleRate', 1/ts, 'MaximumDopplerShift', fd, 'PathDelays', delays_urbanETU, 'AveragePathGains', power_urbanETU);

TTI = 0.001;

% number of samples of one channel realization
numSamples = TTI / ts;

% total trace duration in s
traceDuration = 10.0;

numRBs = 100;

sig = zeros(numSamples, 1); % Signal
sig(1) = 1; % dirac impulse

[psdsig,F] = pwelch(sig,[],[],numRBs,fs,'twosided');     

for ii=1:round((traceDuration/TTI))
        
    % y is the frequency response of the channel
    y = c(sig);   
    
%     [Pxx,F] = PWELCH(X,WINDOW,NOVERLAP,NFFT,Fs) returns a PSD computed as
%     a function of physical frequency (Hz).  Fs is the sampling frequency
%     specified in Hz.  If Fs is empty, it defaults to 1 Hz.
%  
%     F is the vector of frequencies at which the PSD is estimated and has
%     units of Hz.  For real signals, F spans the interval [0,Fs/2] when NFFT
%     is even and [0,Fs/2) when NFFT is odd.  For complex signals, F always
%     spans the interval [0,Fs).


    [psdy,F] = pwelch(y,[],[],numRBs,fs);      
    
    %% the gain in frequency is obtained by dividing the psd of the received signal
    %% by the psd of the original signal. Note that the psd of the original
    %% signal is constant in frequency (since the transform of a delta is a
    %% constant)
    ppssdd(:,ii) = psdy ./ psdsig;  
    

    % this to plot
    %   figure;
    %    PWELCH(y,[],[],numRBs,fs);

    % alternative plot
    %figure;
    %plot(F, 10.*log10(ppssdd(:,ii)));
end

surf(10.*log10(ppssdd));
title ('Fading Pedestrian EPA 3kmph');
%title ('Fading Vehicular EVA 60kmph');
%title('Fading Urban ETU 0 kmph');
%title('Fading Urban ETU 3 kmph');
%title('Fading Urban ETU 30 kmph');
tag = 'EPA_3kmph';
%tag = 'EVA_60kmph';
%tag = 'ETU_0kmph';
%tag = 'ETU_3kmph';
%tag = 'ETU_30kmph';
ylabel ('RB index');
xlabel ('time [ms]');
zlabel ('Amplitude [dB]');

%FILE GENERATION
TAG = upper(tag);
len = round((traceDuration/TTI));

file = fopen(strcat('fading_trace_',tag,'.fad'),'at');   

        
for j=1:numRBs
    for s=1:len
        fprintf(file, '%g ',10.*log10(ppssdd(j, s)));

    end
    fprintf(file, '\n');
end

fclose(file);

