clear all;
close all;

%% LTE from theory to practice
%% Table 22.7 Reference sensitivity.

f = 2160e6;  % carrier freq Hz, EARFCN = 500 (downlink)
nrbs = 25; % tx bandwidth configuration in number of RBs
bw = nrbs * 180000; % bandwidth in Hz, note that this is smaller than
		    % the nominal Channel Bandwidth, see TS 36.101 fig 5.6-1
kT = -174; % noise PSD in dBm / Hz
n = kT + 10*log10(bw);  % noise power dBm
p = 20;  % tx power dBm
txPsd = p - 10*log10(bw); % power / bandwidth in linear units
nf = 9; % receiver noise figure in dB

pW = 10.^((p-30)./10); % tx power in W
nW = 10.^((n+nf-30)./10); % noise power in W

%%d = logspace (0,5,100);
x = linspace (0,4000,100);
g1 = gain_freespace(abs (x-1000),f); % propagation gain from
				% eNB 1
g2 = gain_freespace(abs (x-2000),f); % propagation gain from
				% eNB 2

sinr1 = 10.*log10 (pW .* g1 ./ (nW + pW .* g2)); %dB
sinr2 = 10.*log10 (pW .* g2 ./ (nW + pW .* g1)); %dB

# semilogy (x, g1, ";eNB 1;", x, g2, ";eNB 2;");
# xlabel ("X-coordinate of UE");
# ylabel ("gain (linear)");
# title ("gain X2 handover measures");

hox200mps=200*8.41193; # speed * handover time
hox150mps=150*10.7719; # speed * handover time

figure;
plot (x, sinr1, ";eNB 1;",
      x, sinr2, ";eNB 2;",
      [hox150mps hox150mps], [min(sinr1), max(sinr1)], ";HO point at 150 m/s;",
      [hox200mps hox200mps], [min(sinr1), max(sinr1)], ";HO point at 200 m/s;");
xlabel ("X-coordinate of UE");
ylabel ("SINR (dB)");
title ("LTE link budget X2 handover measures");
myaxis = axis;
myaxis(3:4) = [-20 20];
axis (myaxis);

grid on;
print ("lte_link_budget_x2_handover_measures.png", "-dpng");
      


