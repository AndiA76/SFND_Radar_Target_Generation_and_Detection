%% Simulation of FMCW Radar Target Generation and Detection

% Closs all figures
close all
% Clear workspace
clear all
% Clear command window
clc


%% FMCW (Frequency Modulated Continuous Wave) Radar Specifications 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency of operation = 77 GHz
% Max Range = 200 m
% Range Resolution = 1 m
% Max Velocity = 70 m/s (= 252 km/h)
% Velocity Resolution = 3 m/s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Operating carrier frequency of FMCW Radar 
f_c = 77e9;  % [Hz]

% Speed of light c ~ 3e8 m/s
c = physconst('LightSpeed');  % [m/s]

% Maximum target detection range r_t_max = 0 ... 200 m
r_t_max = 200.0;  % [m]

% Desired range resolution r_t_res = 1 m
r_t_res = 1.0;  % [m]

% Assumed maximum target velocity of cars v_t_max = 70 m/s = 252 km/h
v_t_max = 70.0;  % [m/s]

% Desired velocity resolution v_t_res = 3 m/s
v_t_res = 3.0;  % [m/s]

% Print the specifications for the waveform design of the FMCW Radar
fprintf('\nSpecifications for the waveform design of the FMCW Radar:\n');
fprintf('-------------------------------------------------------------\n');
fprintf('Operating carrier frequency      f_c = %4.1e Hz\n', f_c);
fprintf('Speed of light                     c = %6.4e m/s\n', c);
fprintf('Maximum detection range      r_t_max = %4.1f m\n', r_t_max);
fprintf('Desired range resolution     r_t_res = %3.1f m\n', r_t_res);
fprintf('Max. target velocity         v_t_max = %6.4f m/s\n', v_t_max);
fprintf('Desired velocity resolution  v_t_res = %3.1f m/s\n', v_t_res);
fprintf('-------------------------------------------------------------\n');
fprintf('\n');


%% User Defined Range and Velocity of target
% Define the target's initial position and velocity. Note: Target velocity
% remains contant in this simulation.

% Choose between MANUAL and RANDOM generation of an initial target state
MANUAL = true;

% Initial target range from the Radar sensor r_t = 0 ... r_t_max
if MANUAL
    % Fixed user-specified target distance
    r_t_start = 110.0;  % [m]
else
    % Get a random target range from a uniform distribution between 
    % r_t = 0 ... r_t_max
    r_t_start = rand * r_t_max;  % [m]
end

% Target velocity v_t = -v_t_max ... +v_t_max (assumed to be constant here)
if MANUAL
    % Fixed user-specified target velocity
    v_t = -20.0;  % [m/s]
else
    % Get a random target velocity from a uniform distribution between 
    % v_t = -v_t_max ... +v_t_max
    v_t = (2 * rand - 1) * v_t_max;  % [m/s]
end

% Print initial state of the target object w. r. t. Radar sensor
fprintf('\nInitial state of our target object w. r. t. Radar sensor:\n');
fprintf('-------------------------------------------------------------\n');
fprintf('Initial target range       r_t_start = %6.4e m\n', r_t_start);
fprintf('Constant target velocity         v_t = %6.4e m/s\n', v_t);
fprintf('-------------------------------------------------------------\n');
fprintf('\n');


%% FMCW Waveform Generation
% Design the FMCW waveform by giving the specs of each of its parameters.
% Calculate the Bandwidth (B_sweep), Chirp Time (T_sweep) and Slope (slope)
% of the FMCW chirp (linear frequency sweep) using the requirements above.

% Wavelength of Radar carrier signal
lambda_c = c / f_c;  % [m]

% Maximum signal round trip time (time delay)
t_d_max = 2 * r_t_max / c;  % [s]

% Define sweep time based on the maximum signal round trip time. Usually,
% the sweep time of the linear FMCW chrip signal should be approx. 5...6 
% times the maximum signal round trip time.
T_sweep = 5.5 * 2 * r_t_max / c;  % [s]

% Derive bandwidth B_sweep of the FMCW chrip from the required range 
% resolution r_t_res = c / (2 * B_sweep).
B_sweep = c / (2 * r_t_res);  % [Hz]

% Maximum beat frequency f_beat_max is the sweep frequency reached at 
% maximum signal trip time t_d_max when varying the chrip frequency 
% from 0 to B_sweep.
% f_beat = B_sweep * t_d_max / T_sweep;
% fprintf('f_beat = %4.3e Hz\n', B_sweep * t_d_max / T_sweep);

% Maximum detection range
% r_t_max = c * f_beat * T_sweep / (2 * B_sweep);
% fprintf('r_t_max = %5.1f m\n', c * f_beat * T_sweep / (2 * B_sweep));

% Derive maximum beat frequency from maximum detection range using the 
% similarity equation f_beat_max / ((2 * r_t_max) / c) = B_sweep / T_sweep.
f_beat_max = 2 * B_sweep * r_t_max / (c * T_sweep);   % [Hz]

% Calculate the slope of the FMCW chirp (linear frequency sweep).
sweepSlope = B_sweep / T_sweep;  % [Hz/s]

% Set number of Doppler cells or number of chirps in one sequence of the
% simulation according to the desired vecolity resolution. The whole range
% from -v_t_max to +v_t_max is covered by the number Doppler cells.
% Remark: Its ideal to have power of 2^ values to allow an application of
% the more efficient Fast Fourier Transform instead of Discrete Fourier 
% Transform.
Nd = 2^nextpow2(2 * v_t_max / v_t_res);

% Approximate maximum Doppler shift of the Radar signal frequency in [Hz]
f_Doppler_max = 2 * v_t_max / lambda_c;

% Get the maximum frequency to be resolved in order to drive the required
% sampling frequency. We also need to take the Nyquist criterion into 
% account as a minimum requirement.
% Remark: We ignore resolving the carrier frequency f_c here what will lead
% to aliasing effects in the simulation! We will take this effect as noise.
% f_max = max([(f_beat_max + f_Doppler_max), B_sweep]);  % [Hz]
f_max = f_beat_max + f_Doppler_max;  % [Hz]

% Derive the number of samples on each chirp sequence or the number of
% range cells as power of 2^ to allow an application of the more efficient 
% Fast Fourier Transform instead of Discrete Fourier Transform.
% Remark: The number of samples must be set more than twice as high as the
% maximum frequency to be resolved to satisfy the Nyquist criterion! 
Nr = 2^(nextpow2(T_sweep * f_max) + 1);

% Get actual sampling time and sampling frequency
Ts = T_sweep / Nr;  % [s]
fs = Nr / T_sweep;  % [Hz]

% Print the sampling parameters
fprintf('\nSampling parameters:\n');
fprintf('-------------------------------------------------------------\n');
fprintf('Number of samples per sweep       Nr = %d\n', Nr);
fprintf('Number of sweep sequences         Nd = %d\n', Nd);
fprintf('Sampling time                     Ts = %4.1e s\n', Ts);
fprintf('Sampling frequency                fs = %4.1e Hz\n', fs);
fprintf('-------------------------------------------------------------\n');
fprintf('\n');

% Print the FMCW Radar wave form design for the chirp signal
fprintf('\nWaveform design of the FMCW Radar chirp signal:\n');
fprintf('-------------------------------------------------------------\n');
fprintf('Carrier frequency                f_c = %6.4e Hz\n', f_c);
fprintf('Carrier signal wavelength   lambda_c = %6.4e m\n', lambda_c);
fprintf('Maximum detection range      r_t_max = %4.1f m\n', r_t_max);
fprintf('Detection range resolution   r_t_res = %3.1f m\n', ...
    c / (2 * B_sweep));
fprintf('Max. signal round trip time  t_d_max = %6.4e s\n', t_d_max);
fprintf('Chirp sweep time             T_sweep = %6.4e s\n', T_sweep);
fprintf('Maximum chirp bandwidth      B_sweep = %6.4e Hz\n', B_sweep);
fprintf('Max beat frequency        f_beat_max = %6.4e Hz\n', f_beat_max);
fprintf('Slope of linear chirp     sweepSlope = %6.4e Hz/s\n', sweepSlope);
fprintf('Max. target velocity         v_t_max = %6.4f m/s\n', v_t_max);
fprintf('Target velocity resolution   v_t_res = %3.1f m/s\n', ...
    2 * v_t_max / Nd);
fprintf('Max. Doppler shift     f_Doppler_max = %6.4e Hz\n',f_Doppler_max);
fprintf('\n');
fprintf('Explanation\n');
fprintf('f_Doppler > 0: Target receding from Radar sensor\n');
fprintf('f_Doppler = 0: Const. distance between target & Radar sensor\n');
fprintf('f_Doppler < 0; Target approaching Radar sensor\n');
fprintf('-------------------------------------------------------------\n');
fprintf('\n');


%% Signal generation and Moving Target simulation
% Running the radar scenario over the time.

% Timestamps for running the displacement scenario for every sample on each
% periodic chirp signal sequence
t = linspace(0, Nd * T_sweep, Nr * Nd);  % [s]

% Instead of looping over all time samples in all Nd chirp sequences we
% simulate the target motion as well as the transmitted and the received 
% FMCW Radar signals using Matlab's more time-efficient matrix operations.

% Update the range of the target for each time stamp assuming a constant 
% target velocity.
r_t = r_t_start + v_t * t;  % [m]

% Get final target range
r_t_final = r_t(length(r_t));  % [m]

% Calculate signal round trip time or time delay between transceived and 
% received signal for each target range position
t_d = 2 * r_t / c;  % [s]

% For each time sample we need to calculate both the transmitted Radar 
% signal Tx and the received Radar signal Rx assuming a linear sine sweep.
Tx = cos(2 * pi * (f_c .* t + sweepSlope * t.^2 / 2));
Rx = cos(2 * pi * (f_c .* (t - t_d) + sweepSlope * (t - t_d).^2 / 2));
    
% Generate the Beat signal by mixing Transmit and Receive signal using
% elementwise matrix multiplication of Transmit and Receive signal
Mix = Tx .* Rx;

% Note: As we do not resolve the carrier frequency, or ignoring the Nyquist
% citerion for the carrier frequency, respectively, we get strong aliasing
% effects, which we take for noise in this simplified simulation!

% Plot the frequency sweep (chirp) sequences of the mixed or beat signal
figure('Name', 'Beat or Mixed Signal in Time Domain')
plot(t, Mix, 'Color', 'b', 'LineStyle', '-', 'LineWidth', 1)
grid on
xlabel('Time [s]'), ylabel('Signal amplitude')
title('Beat or Mixed Signal in Time Domain')
legend('Mixed = Tx .* Rx', 'Location', 'NorthWest')
set(gcf, 'Color', 'w', 'Position', [676, 549, 720, 480])


%% RANGE MEASUREMENT

% Actual beat frequency for the initial target range
f_beat_start = 2 * B_sweep * r_t_start / (c * T_sweep);   % [Hz]

% Actual beat frequency for the final target range
f_beat_final = 2 * B_sweep * r_t_final / (c * T_sweep);  % [Hz]

% Print actual beat frequency for the initial and the final target state
fprintf('\nBeat frequency for the initial and final target range:\n');
fprintf('-------------------------------------------------------------\n');
fprintf('Initial target range       r_t_start = %6.4e m\n', r_t_start);
fprintf('Final target range         r_t_final = %6.4e m\n', r_t_final);
fprintf('\n');
fprintf('Initial beat frequency  f_beat_start = %6.4e Hz\n', f_beat_start);
fprintf('Final beat frequency    f_beat_final = %6.4e Hz\n', f_beat_final);
fprintf('-------------------------------------------------------------\n');
fprintf('\n');

% Plot the power spectrum of the beat signal (= mixed transmitted and 
% received Radar chirp signal) as heat map versus frequency and chirp time
% We will have a constant horizontal line at the beat frequency that
% corresponds to the actual target position which hardly changes over time
% considering the very short time period displayed.
figure('Name', 'Beat Signal Spectrogram')
pspectrum(Mix, fs, 'spectrogram', ...
    'TimeResolution', T_sweep, 'OverlapPercent', 99, 'Leakage', 0.85);
set(gcf, 'Color', 'w', 'Position', [676, 549, 720, 480])

% Reshape the vector into Nr*Nd array. Nr and Nd here would also define the
% size of Range and Doppler FFT respectively.
X = reshape(Mix, [Nr, Nd]);

% Run the FFT on the beat signal along the range bins dimension (Nr), or
% along the columns, respectively, take the absolute value of the fft 
% output and normalize over the number of data points.
X_fft = abs(fft(X, Nr)) / Nr;

% The FFT provides an axis-symmetric spectrum from -fs/2 ... +fs/2 (where
% fs is the sampling frequency) with a symmetry axis going through f = 0.
% We only use the upper half from f = 0 ... +fs/2 and throw away the rest.
X_fft = X_fft(1 : Nr / 2 + 1, :);
%fprintf('Length of the Fourier Spectrum = %d\n', length(X_fft(:, 1)));

% Define the frequency vector from f = 0 ... +fs/2 along the chirp or range
% axis, respectivly
f_fft = linspace(0, fs / 2, Nr / 2 + 1);

% Calculate signal power of each frequency component in [dB]
Px_fft = pow2db(X_fft .* conj(X_fft) / Nr);

% Calculate the range vector
range_fft = c * f_fft * T_sweep / (2 * B_sweep);

% Find the estimated target position (assumption: only one target peak 
% exists in this simulation).
r_t_est = range_fft(find(X_fft == max(X_fft), 1, 'first'));

% Print true and estimated target object distance w. r. t. Radar sensor
fprintf('\nEstimated target range using FFT of the first chirp signal:\n');
fprintf('-------------------------------------------------------------\n');
fprintf('Initial target range       r_t_start = %6.4e m\n', r_t_start);
fprintf('Estimated target range       r_t_est = %6.4e m\n', r_t_est);
fprintf('-------------------------------------------------------------\n');
fprintf('\n');

% Plotting the range FFT output
figure('Name','Range from FFT of first beat signal chirp')
subplot(2,1,1)
plot(f_fft, X_fft(:, 1))
grid on
xlabel('Frequency in [Hz]'), ylabel('Signal magnitude')
title('Fourier Transform of first beat signal chirp')
subplot(2,1,2)
plot(range_fft, X_fft(:, 1))
grid on
% axis([0 200 0 1])
set(gca, 'XLim', [0, 200])
xlabel('Range in [m]'), ylabel('Signal magnitude')
set(gcf, 'Color', 'w', 'Position', [676, 549, 720, 480])

% Plot the signal power spectrum
figure('Name', 'Range from signal power spectrum of 1st beat signal chirp')
subplot(2,1,1)
plot(f_fft, Px_fft(:, 1))
grid on
xlabel('Frequency in [Hz]'), ylabel('Signal power [dB]')
title('Power Spectrum of first beat signal chirp')
subplot(2,1,2)
plot(range_fft, Px_fft(:, 1))
grid on
set(gca, 'XLim', [0, 200])
xlabel('Range in [m]'), ylabel('Signal power [dB]')
set(gcf, 'Color', 'w', 'Position', [676, 549, 720, 480])


%% RANGE DOPPLER RESPONSE
% Range Doppler Map Generation using 2D FFT applied on the mixed or beat
% signal, which is split into single chirp sequences that are arranged into
% a 2D array. The first axis corresponds to the length of the chirp signal
% sequence (range axis) and the second axis corrensponds to the number of
% chirp sequences send out and received one after another (Doppler axis).
% The output of the 2D FFT is an image that has reponse in the range and
% doppler FFT bins. It is necessary to convert the range and Doppler axis
% from frequency bin sizes to range and velocity values using the maximum
% possible values given by the FMCW waveform.

% Approximate Doppler frequency shift for the target velocity
f_Doppler = 2 * v_t / lambda_c;

% Print actual Doppler shift for the target velocity
fprintf('\nActual Doppler shift for the target velocity:\n');
fprintf('-------------------------------------------------------------\n');
fprintf('Constant target velocity         v_t = %6.4e m/s\n', v_t);
fprintf('\n');
fprintf('Actual Doppler shift       f_Doppler = %6.4e Hz\n', f_Doppler);
fprintf('\n');
fprintf('Explanation\n');
fprintf('f_Doppler > 0: Target receding from Radar sensor\n');
fprintf('f_Doppler = 0: Const. distance between target & Radar sensor\n');
fprintf('f_Doppler < 0; Target approaching Radar sensor\n');
fprintf('-------------------------------------------------------------\n');
fprintf('\n');

% Reshape the mix signal vector into an Nr*Nd array. Nr and Nd here would 
% also define the size of Range and Doppler FFT respectively.
Mix = reshape(Mix, [Nr, Nd]);

% 2D FFT using the FFT size for both dimensions.
sig_fft2 = fft2(Mix, Nr, Nd);

% Shift zero Doppler frequency component to the center of the spectrum
sig_fft2 = fftshift(sig_fft2, 2);  % shift along Doppler axis only

% Taking just one side of signal from Range dimension.
sig_fft2 = sig_fft2(1 : Nr / 2 + 1, 1 : Nd);
RDM = abs(sig_fft2);
RDM = 10 * log10(RDM);

% Define the frequency vector from f = 0 ... +fs/2 along the chirp or range
% axis, respectivly
f_fft = linspace(0, fs / 2, Nr / 2 + 1);

% Define the Dopper frequency shift vector along the Doppler axis starting
% from min(f_Doppler) ... max(f_Doppler)
f_Doppler_fft = linspace(-1 / T_sweep / 2, 1 / T_sweep / 2, Nd);

% Use the surf function to plot the output of 2DFFT and to show axis in 
% both dimensions where range and Doppler axis are converted to represent
% estimated target distance and velocity, respectively
doppler_axis = f_Doppler_fft / 2 * lambda_c;
range_axis = c * f_fft * T_sweep / (2 * B_sweep);

% Plot the Range Doppler Response Map
figure('Name', 'Range Doppler Map')
surf(doppler_axis, range_axis,RDM, 'LineStyle', ':', 'LineWidth', 0.5);
xlabel('Doppler or velocity axis [m/s]')
ylabel('Range axis [m]')
zlabel('Signal level [dB]')
title('Range Doppler Map (2D FFT on mixed beat signal sequences)')
set(gcf, 'Color', 'w', 'Position', [676, 549, 720, 480])


%% 2D CFAR implementation

% Implement 2D CFAR by sliding a window of training and guard cells around
% a cell under test (CUT) on the Range Doppler Map RDM[x,y] obtained from
% the 2D FFT.

% Select the number of Training Cells in both the dimensions.
Tr = 16;  % range dimension
Td = 8;  % Doppler dimension

% Select the number of Guard Cells in both dimensions around the Cell under 
% Test (CUT) for accurate estimation
Gr = 8;  % range dimension
Gd = 4;  % Doppler dimension

% Offset the threshold by SNR value in dB
SNR_offset_dB = 8;

% Create a vector to store noise_level for each iteration on training cells
noise_level = zeros(1,1);

% Design a loop such that slides the cell under test (CUT) across the Range
% Doppler Map by giving margins at the edges for Training and Guard Cells.
% For every iteration sum the signal level within all the training cells.
% To sum up the signal values convert the value from logarithmic to linear
% using db2pow function. Average the summed values over all the training
% cells used. After averaging convert it back to logarithimic using pow2db.
% Further add the offset to it to determine the threshold. Next, compare 
% the signal level in CUT with this threshold. If the CUT signal level >
% threshold assign the CUT a value of 1 for a detection, else set it to 0
% for no detection.

% The process above will generate a thresholded block, which is smaller 
% than the Range Doppler Map as the CUT cannot be located at the edges of
% matrix. Hence, few cells will not be thresholded. To keep the map size 
% same set those values to 0.
cfar2D = zeros(size(RDM));

% Width of the 2D CFAR sliding window in Range and Doppler dimension
wr = 2 * (Gr + Tr) + 1;
wd = 2 * (Gd + Td) + 1;

% 2D array to hold the threshold values
threshold_cfar = zeros(Nr / 2 + 1 - 2 * (Tr + Gr), Nd - 2 * (Td + Gd));

% 2D array to hold the final signal after thresholding
sig_cfar2D = zeros(Nr / 2 + 1 - 2 * (Tr + Gr), Nd - 2 * (Td + Gd));

% Generate 2D mesh grid the cfar threshold and filtered signal
[X_cfar,Y_cfar] = meshgrid((Td + Gd) : 1 : (Nd - (Td + Gd) - 1), ...
    (Tr + Gr) : 1 : (Nr / 2 + 1 - (Tr + Gr) - 1));

% Slide window across the rows of the 2D FFT RDM array where (i, j) 
% is the lower left starting point of the 2D sliding window
for i = 1 : (Nr/2+1 - wr + 1)
    
    % Slide window across the columns of the 2D FFT RDM array where (i, j) 
    % is the lower lewrft starting point of the 2D sliding window.
    for j = 1 : (Nd - wd + 1)
        
        % Determine the noise threshold by measuring the noise level in the
        % training cells (before/after as well as below/above the cell 
        % under test (CUT)) of the sliding window within the 2D FFT
        % converted from logarithmic to linear signal power.
        noise_level = ...
            sum(sum(db2pow(RDM(i : i + wr - 1, j : j + wd - 1)))) - ...
            sum(sum(db2pow(RDM(i + Tr : i + Tr + 2 * Gr + 1, ...
            j + Td : j + Td + 2 * Gd + 1))));
        
        % Number of training cells
        NT = wr * wd - (2 * Gr + 1) * (2 * Gd + 1);
        
        % To determine the noise threshold take the average of summed noise
        % over all training cells, convert it back to logarithmic signal
        % values and add the logarithmic SNR offset.
        threshold = pow2db(noise_level / NT) + SNR_offset_dB;
        threshold_cfar(i, j) = threshold;
          
        % Now pick the cell under test (CUT) right in the center of the 2D
        % sliding window which is Tr + Gr cells above and Td + Gd to the 
        % right of the lower left corner of the 2D sliding window and 
        % measure the signal level within the CUT.
        signal = RDM(Tr + Gr + i, Td + Gd + j);
        
        % Filter the signal above threshold: If the signal level at the 
        % cell under test (CUT) falls below the threshold assign it a 0 
        % value.
        if (signal < threshold)
            sig_cfar2D(i, j) = 0;
            cfar2D(Tr + Gr + i, Td + Gd + j) = 0;
        else
            sig_cfar2D(i, j) = signal;
            cfar2D(Tr + Gr + i, Td + Gd + j) = 1;
        end        
        
    end
    
end

% Display the CFAR2D output using the surf() function like we did for Range
% Doppler Response output.
figure('Name', '2D CFAR on Range Doppler Map')
surf(doppler_axis, range_axis, cfar2D, 'LineStyle', ':', 'LineWidth', 0.5)
colorbar
xlabel('Doppler or velocity axis [m/s]')
ylabel('Range axis [m]')
zlabel('Signal level [dB]')
title('2D CFAR Output on Range Doppler Map')
set(gcf, 'Color', 'w', 'Position', [676, 549, 720, 480])

% Find the estimated target range and velocity in 2D CFAR map (assumption:
% only one target peak exists in this simulation).
[rows, cols, vals] = find(cfar2D == 1);
r_t_est = range_axis(round((min(rows) + max(rows)) / 2));
v_t_est = doppler_axis(round((min(cols) + max(cols)) / 2));

% Print true and estimated target range and velocity
fprintf('\nEstimated target range and velocity from 2D CFAR:\n');
fprintf('-------------------------------------------------------------\n');
fprintf('True average target range   r_t_mean = %6.4e m\n', ...
    (r_t_start + r_t_final) / 2);
fprintf('Estimated target range       r_t_est = %6.4e m\n', r_t_est);
fprintf('\n');
fprintf('True target velocity             v_t = %6.4e m/s\n', v_t);
fprintf('Estimated target velocity    v_t_est = %6.4e m/s\n', v_t_est);
fprintf('-------------------------------------------------------------\n');
fprintf('\n');


 