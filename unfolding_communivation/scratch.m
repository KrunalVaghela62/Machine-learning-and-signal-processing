%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN SCRIPT : Compare signal lengths for two sampling approaches
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% Parameters
E  = 1;
W  = 10;          % Bandlimit parameter
OF = 7;           % Oversampling factor
S  = [-1 1];      % Not used but kept for compatibility

%% ================= APPROACH 1 =================
% Td = Ts/1000 and simple downsample

[x_orig_cont1, t_d1, CoS1, NoS1] = generate_analog_transmit(E, W, OF, S);
[x_orig1, t_s1] = generate_sampled_signal(x_orig_cont1, t_d1, W, OF);

len_cont1 = length(x_orig_cont1);
len_samp1 = length(x_orig1);

%% ================= APPROACH 2 =================
% Fix Td = 10e-7 and use step-based sampling

% --- Manually modify Td ---
NoS = 4;
BW  = 2*pi*W;

Wnyq = 2*BW;
Tnyq = 2*pi/Wnyq;
Ws   = OF*Wnyq;
Ts   = 2*pi/Ws;

Td = 4/1200000;

t2 = (-2:Td:2).';
f2 = zeros(length(t2),1);

CoS = [1,2,1,-1];

for m = 1:NoS
    y_m = CoS(m) * sinc( 2*W * (t2 - ((2*m-(NoS))*Tnyq)));
    f2 = f2 + y_m;
end

x_orig_cont2 = f2;
t_d2 = t2;

% ===== sampling using commented code logic =====
dt = t_d2(2) - t_d2(1);
step = round(Ts/dt);

idx = 1:step:length(t_d2);

x_orig2 = x_orig_cont2(idx);
t_s2    = t_d2(idx);

len_cont2 = length(x_orig_cont2);
len_samp2 = length(x_orig2);

%% ================= RESULTS =================

fprintf('----------- APPROACH 1 -----------\n');
fprintf('Continuous signal length = %d\n', len_cont1);
fprintf('Sampled signal length    = %d\n\n', len_samp1);

fprintf('----------- APPROACH 2 -----------\n');
fprintf('Continuous signal length = %d\n', len_cont2);
fprintf('Sampled signal length    = %d\n\n', len_samp2);

%% Optional visualization
figure
subplot(2,1,1)
plot(t_s1,x_orig1,'o-')
title('Sampled Signal (Approach 1)')
grid on

subplot(2,1,2)
plot(t_s2,x_orig2,'o-')
title('Sampled Signal (Approach 2)')
grid on