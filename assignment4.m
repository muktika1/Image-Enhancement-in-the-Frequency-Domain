% Name : Muktika Manohar
% Contact Email : mm87150n@pace.edu
% Assignment 4

% Part 1 : Problem 1

% Part A

% Load image
img = imread('/Users/muku/Downloads/Sample.jpg');

% Perform FFT
F = fftshift(fft2(double(img)));
[M, N] = size(F);

[X, Y] = meshgrid(-N/2:N/2-1, -M/2:M/2-1);
D = sqrt(X.^2 + Y.^2); 

std = 40;
H_gaussian = exp(-(D.^2) / (2 * std^2));
F_gaussian = F .* H_gaussian;
img_gaussian = abs(ifft2(ifftshift(F_gaussian)));

% Results
figure;
subplot(1, 3, 1), imshow(img, []), title('Original Image');
subplot(1, 3, 2), imshow(log(1 + abs(H_gaussian)), []), title('Gaussian Low-pass Filter');
subplot(1, 3, 3), imshow(img_gaussian, []), title('Filtered Image');
pause;

% Part B
D0 = 80;
n = 2;
H_butterworth = 1 ./ (1 + (D0 ./ D).^(2 * n));
F_butterworth = F .* H_butterworth;
img_butterworth = abs(ifft2(ifftshift(F_butterworth)));

% Results
figure(2);
subplot(1, 3, 1), imshow(img, []), title('Original Image');
subplot(1, 3, 2), imshow(log(1 + abs(H_butterworth)), []), title('Butterworth High-pass Filter');
subplot(1, 3, 3), imshow(img_butterworth, []), title('Filtered Image');
pause;

% Part C
disp('Effects of Gaussian Low-pass Filter:');
disp('- Smoothes the image and reduces noise.');
disp('- High-frequency details are softened.');
disp('- Creates a more uniform appearance.');
disp('- Suitable for noise reduction and when fine details are not a priority.');
disp(' ');

disp('Effects of Butterworth High-pass Filter:');
disp('- Enhances fine details and edges.');
disp('- Low-frequency components are reduced.');
disp('- The image appears sharper with accentuated features.');
disp('- Ideal for edge detection, sharpening, and emphasizing textures.');
pause;

% Part D
close all;
clear ;

% Problem 2: Exercises on Frequency Domain Operations 
% Part A

close all; clear; clc;
% Load Images
sampleImg = imread('/Users/muku/Downloads/Sample.jpg');
newCapitolImg = imread('/Users/muku/Downloads/NewCapitol.jpg');

sampleImg = im2gray(sampleImg);
newCapitolImg = im2gray(newCapitolImg);

% Compute Fourier Magnitude and Phase for Sample Image
sampleMagnitude = abs(fftshift(fft2(double(sampleImg))));
samplePhase = angle(fftshift(fft2(double(sampleImg))));

% Compute Fourier Magnitude and Phase for NewCapitol Image
newCapitolMagnitude = abs(fftshift(fft2(double(newCapitolImg))));
newCapitolPhase = angle(fftshift(fft2(double(newCapitolImg))));

% Display Fourier Magnitude and Phase
figure('Name', 'Fourier Transform Analysis', 'NumberTitle', 'off');

subplot(2, 2, 1), imshow(log(1 + sampleMagnitude), []), title('Sample Fourier Magnitude');
subplot(2, 2, 2), imshow(samplePhase, []), title('Sample Fourier Phase');
subplot(2, 2, 3), imshow(log(1 + newCapitolMagnitude), []), title('NewCapitol Fourier Magnitude');
subplot(2, 2, 4), imshow(newCapitolPhase, []), title('NewCapitol Fourier Phase');

pause;

% Part B

magnitude_only_sample = abs(fftshift(fft2(double(sampleImg))));
phase_only_sample = fftshift(fft2(double(sampleImg))) ./ (magnitude_only_sample + (magnitude_only_sample == 0));

magnitude_only_newcapitol = abs(fftshift(fft2(double(newCapitolImg))));
phase_only_newcapitol = fftshift(fft2(double(newCapitolImg))) ./ (magnitude_only_newcapitol + (magnitude_only_newcapitol == 0));

% Reconstruct Images from Modified Fourier Attributes

reconstructed_magnitude_sample = real(ifft2(ifftshift(magnitude_only_sample)));
reconstructed_phase_sample = real(ifft2(ifftshift(phase_only_sample)));

reconstructed_magnitude_newcapitol = real(ifft2(ifftshift(magnitude_only_newcapitol)));
reconstructed_phase_newcapitol = real(ifft2(ifftshift(phase_only_newcapitol)));

% Display the reconstructed images

figure('Name', 'Reconstructed Images from Modified Attributes', 'NumberTitle', 'off');

% Display Magnitude-only and Phase-only reconstructed images for Sample
subplot(2, 2, 1), imshow(reconstructed_magnitude_sample, []), title('Sample (Magnitude Only)');
subplot(2, 2, 2), imshow(reconstructed_phase_sample, []), title('Sample (Phase Only)');

% Display Magnitude-only and Phase-only reconstructed images for NewCapitol
subplot(2, 2, 3), imshow(reconstructed_magnitude_newcapitol, []), title('NewCapitol (Magnitude Only)');
subplot(2, 2, 4), imshow(reconstructed_phase_newcapitol, []), title('NewCapitol (Phase Only)');

pause;

% Display the reconstructed images using logarithmic scale

figure('Name', 'Reconstructed Images from Modified Fourier Attributes With Logarithmic Scale', 'NumberTitle', 'off');

% Display Magnitude-only and Phase-only reconstructed images for Sample with a logarithmic scale
subplot(2, 2, 1), imshow(log(1 + reconstructed_magnitude_sample), []), title('Sample (Magnitude Only)');
subplot(2, 2, 2), imshow(reconstructed_phase_sample, []), title('Sample (Phase Only)');

% Display Magnitude-only and Phase-only reconstructed images for NewCapitol with a logarithmic scale
subplot(2, 2, 3), imshow(log(1 + reconstructed_magnitude_newcapitol), []), title('NewCapitol (Magnitude Only)');
subplot(2, 2, 4), imshow(reconstructed_phase_newcapitol, []), title('NewCapitol (Phase Only)');

pause;

% Part D - Explanation
disp('Insight into Magnitude-only Reconstructed Images:');
disp('In the realm of Fourier Transforms, images speak two dialects: magnitude and phase.');
disp("Magnitude represents the volume of each frequency's voice, showing how loud or quiet they are.");
disp('However, when we rebuild an image using only this magnitude, we silence the spatial details encoded in phase.');
disp("The result is a representation of the image's intensity without the fine-grained storylines.");
disp("In this form, the image emerges softer, devoid of the original's intricacies.");

pause;

% Problem 3
% Load and Display Noisy Image
noisy_image = imread('/Users/muku/Downloads/boy_noisy.gif');

% a - Fourier Transform of Noisy Image
noisy_image_fft = fftshift(fft2(double(noisy_image)));

% b

% Compute the magnitude of the FFT and mask the central value
magnitude = abs(noisy_image_fft);
magnitude(size(noisy_image, 1) / 2, size(noisy_image, 2) / 2) = 0;
% Find the threshold based on the top magnitudes
top_magnitudes = maxk(magnitude(:), 10);
threshold = min(top_magnitudes);

% c
mask = magnitude > threshold;
average_value = mean(noisy_image_fft(~mask));
noisy_image_fft(mask) = average_value;

% d -  Inverse Fourier Transform
enhanced_image = real(ifft2(ifftshift(noisy_image_fft)));

% Linearly adjust the image to match the original brightness
enhanced_image = enhanced_image * (mean2(noisy_image) / mean2(enhanced_image));

% Display

figure;
subplot(1, 2, 1);
imshow(noisy_image, []);
title('Original Image');

subplot(1, 2, 2);
imshow(enhanced_image, []);
title('Enhanced Image');

%e
disp('Explanation for Choosing Four Largest Magnitude Peaks:');
disp('The four largest distinct magnitude values correspond to the dominant frequency components');
disp('in the Fourier spectrum of the noisy image. These frequencies are likely to represent the');
disp('cosine interference added to the image. By modifying these magnitudes and their symmetrical');
disp('counterparts, we effectively suppress the dominant noise components. This approach is effective');
disp('since it targets the most significant noise contributors while preserving other image details.');

% Close all figures and all variables in the workspace
pause;
close all;
clear; 










