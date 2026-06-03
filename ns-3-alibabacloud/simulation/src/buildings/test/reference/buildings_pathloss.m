clear all;
close all;

%% BuildingsPropagationLossModel pathloss test reference script


%% ITU1411, COST231, OH, etc. scripts are in the propagation module
addpath ("../../../propagation/test/reference/");


%f = 2114e6;  % carrier freq Hz, EARFCN = 500 (downlink)
%f = 1900e6;  % carrier freq Hz, EARFCN = 500 (downlink)
f = 869e6;
%f = 2620e6;
d = 2; %km
hb = 30;
hm = 1;
hr = 20;
fmhz = f/1e6;

% Test #1
g = loss_OH_large_cities_urban (d, hb, hm, fmhz);
disp ("Test #1: the value of OH for large cities is:"), disp (g)

g = loss_OH_small_cities_urban (d, hb, hm, fmhz);
disp ("Test #1: the value of OH for small cities is:"), disp (g)

g = loss_OH_suburban (d, hb, hm, fmhz);
disp ("Test #1: the value of OH in suburban is:"), disp (g)

g = loss_OH_openareas (d, hb, hm, fmhz);
disp ("Test #1: the value of OH in openareas is:"), disp (g)

% Test #2 and #7
fmhz = 2114;
g = loss_COST231_large_cities_urban (d, hb, hm, fmhz);
disp ("Test #2: the value of COST231 for large cities is:"), disp (g)

g = loss_COST231_small_cities_urban (d, hb, hm, fmhz);
disp ("Test #2: the value of COST231 for small cities is:"), disp (g)

% Test #3
g = loss_Kun_2_6GHz (d*1000);
disp ("Test #3: the value of Kun 2.6 GHz is:"), disp (g)

% Test #4
d = 1041.2; %m
f = 2114e6;
g = loss_ITU1411_LOS (d, hb, hm, f);
disp ("Test #4: the value of ITU1411 in LOS is:"), disp (g)

% Test #5
d = 900;
l = 80;
b = 50;
st_w = 20;
phi = 45;
big = 1; % metropolitan centre
g = loss_ITU1411_NLOS_over_rooftop (d, hb, hm, hr, f, l, b, st_w, phi, big);
disp ("Test #5: the value of ITU1411 in NLOS over the roof-top is:"), disp (g)

% Test #6
n_floors = 2;
built_t = 2;
d = 31.3209;
g = loss_ITU1238 (d, fmhz, n_floors, built_t);
disp ("Test #6: the value of ITU1238 is:"), disp (g)

% Test #9
d = 100.404;
f = 2114e6;
hb = 10;
g = loss_ITU1411_LOS (d, hb, hm, hr, f);
disp ("Test #9: the value of ITU1411 in LOS is:"), disp (g)



% Test #10
d = 500;
l = 80;
b = 50;
st_w = 20;
phi = 45;
big = 1; % metropolitan centre
hb = 10;
g = loss_ITU1411_NLOS_over_rooftop (d, hb, hm, hr, f, l, b, st_w, phi, big);
disp ("Test #10: the value of ITU1411 in NLOS over the roof-top is:"), disp (g)