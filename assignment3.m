% Assignment 3 - Image Enhancement in the Spatial Domain
% Name: Muktika Manohar
% Email: mm87150n@pace.edu

% Problem 1:  Exercises on Low-pass and High-pass Filters in the Spatial Domain
%Part a
function main()
function filteredImage = MeanFilter(inputImage, mask)
    [rows, cols] = size(inputImage);
    [maskRows, maskCols] = size(mask);
    
    % Initialize the filtered image
    filteredImage = zeros(rows, cols);
    
    % Calculate the border offset for the mask
    borderOffset = floor((maskRows - 1) / 2);
    
    % Iterate through the image pixels
    for i = 1:rows
        for j = 1:cols
            % Initialize the sum for the mean calculation
            pixelSum = 0;
            
            % Iterate through the mask
            for m = -borderOffset:borderOffset
                for n = -borderOffset:borderOffset
                    % Check for boundary conditions
                    if (i + m >= 1 && i + m <= rows && j + n >= 1 && j + n <= cols)
                        pixelSum = pixelSum + inputImage(i + m, j + n) * mask(m + borderOffset + 1, n + borderOffset + 1);
                    end
                end
            end
            
            % Calculate the mean and assign it to the filtered image
            filteredImage(i, j) = pixelSum;
        end
    end
    
    % Normalize the filtered image
    filteredImage = uint8(filteredImage);
end
% Load the noisy image 'Circuit.jpg'
inputImage = imread('/Users/muku/Documents/Circuit.jpg');

% Define the 3x3 and 5x5 averaging filters
mask3x3 = ones(3) / 9;
mask5x5 = ones(5) / 25;

% Apply the mean filter with a 3x3 mask
filteredImage3x3 = MeanFilter(inputImage, mask3x3);

% Apply the mean filter with a 5x5 mask
filteredImage5x5 = MeanFilter(inputImage, mask5x5);

% Display the original image and filtered images
figure(1);
subplot(1, 3, 1);
imshow(inputImage);
title('Original Image');

subplot(1, 3, 2);
imshow(filteredImage3x3, []);
title('3x3 Mean Filtered Image');

subplot(1, 3, 3);
imshow(filteredImage5x5, []);
title('5x5 Mean Filtered Image');
disp('-----Finish Solving Part a-----');
pause;

%Part b
% Create a 3-by-3 averaging filter using fspecial
mask3x3_fs = fspecial('average', [3, 3]);

% Create a 5-by-5 averaging filter using fspecial
mask5x5_fs = fspecial('average', [5, 5]);

% Apply the 3-by-3 and 5-by-5 filters using filter2
filteredImage3x3_fs = filter2(mask3x3_fs, inputImage);
filteredImage5x5_fs = filter2(mask5x5_fs, inputImage);


% Compare the images using if/else
if isequal(filteredImage3x3_fs, filteredImage3x3)
    disp('3x3 Filtered Images are the same.');
else
    disp('3x3 Filtered Images are different.');
end

if isequal(filteredImage5x5_fs, filteredImage5x5)
    disp('5x5 Filtered Images are the same.');
else
    disp('5x5 Filtered Images are different.');
end

% Display the original image and the two sets of processed images
figure(1);
subplot(2, 3, 1);
imshow(inputImage);
title('Original Image');

subplot(2, 3, 2);
imshow(filteredImage3x3_fs, []);
title('3x3 Filter fspecial');

subplot(2, 3, 3);
imshow(filteredImage3x3, []);
title('My Mean 3x3 Filter');

subplot(2, 3, 5);
imshow(filteredImage5x5_fs, []);
title('5x5 Filter fspecial');

subplot(2, 3, 6);
imshow(filteredImage5x5, []);
title('My Mean 5x5 Filter');
disp('-----Finish Solving Part b-----');
pause;

% Part c
% Weighted 3-by-3 Median Filter
function filteredImage = Weighted3x3MedianFilter(inputImage)
    [m, n] = size(inputImage);
    
    % Initialize the output image
    filteredImage = uint8(zeros(m, n));
    
    % Pad the input image to handle border pixels
    paddedImage = padarray(inputImage, [1, 1], 'replicate');
    
    for i = 2:m+1
        for j = 2:n+1
            % Extract the 3x3 neighborhood of the current pixel
            neighborhood = paddedImage(i-1:i+1, j-1:j+1);
            
            % Flatten the neighborhood
            neighborhood = double(neighborhood(:));
            
            % Apply the weighted median filter
            weights = [1 2 2; 1 1 1; 2 1 1];
            weighted_neighborhood = sort(neighborhood .* weights(:));
            filteredImage(i-1, j-1) = uint8(weighted_neighborhood(5));
        end
    end
end
% Standard 5-by-5 Median Filter
function filteredImage = Standard5x5MedianFilter(inputImage)
    [m, n] = size(inputImage);
    
    % Initialize the output image
    filteredImage = uint8(zeros(m, n));
    
    % Pad the input image to handle border pixels
    paddedImage = padarray(inputImage, [2, 2], 'replicate');
    
    for i = 3:m+2
        for j = 3:n+2
            % Extract the 5x5 neighborhood of the current pixel
            neighborhood = paddedImage(i-2:i+2, j-2:j+2);
            
            % Flatten the neighborhood
            neighborhood = double(neighborhood(:));
            
            % Apply the standard 5x5 median filter
            filteredImage(i-2, j-2) = uint8(median(neighborhood));
        end
    end
end

% Load the noisy image
inputImage = imread('/Users/muku/Documents/Circuit.jpg');

% Apply the Weighted 3x3 Median Filter
filtered_image_weighted_3x3 = Weighted3x3MedianFilter(inputImage);

% Apply the Standard 5x5 Median Filter
filtered_image_standard_5x5 = Standard5x5MedianFilter(inputImage);

% Display the original and filtered images
figure(2);
subplot(1, 3, 1);
imshow(inputImage);
title('Original Image');

subplot(1, 3, 2);
imshow(filtered_image_weighted_3x3);
title('Weighted 3x3 Median Filter');

subplot(1, 3, 3);
imshow(filtered_image_standard_5x5);
title('Standard 5x5 Median Filter');
disp('-----Finish Solving Part c-----');
pause;

% Part d
% Define the 3x3 and 5x5 median filter sizes
filterSize_3x3 = 3;
filterSize_5x5 = 5;

% Apply a 3x3 median filter using medfilt2
filteredImage_medfilt2_3x3 = medfilt2(inputImage, [filterSize_3x3, filterSize_3x3]);

% Apply a 5x5 median filter using medfilt2
filteredImage_medfilt2_5x5 = medfilt2(inputImage, [filterSize_5x5, filterSize_5x5]);

% Compare the results
areEqual_3x3 = isequal(filteredImage_medfilt2_3x3, filtered_image_weighted_3x3);
areEqual_5x5 = isequal(filteredImage_medfilt2_5x5, filtered_image_standard_5x5);


figure(2);
subplot(2, 2, 1);
imshow(filteredImage_medfilt2_3x3);
title('medfilt2 3x3');

subplot(2, 2, 2);
imshow(filtered_image_weighted_3x3);
title('My median filter 3x3');

subplot(2, 2, 3);
imshow(filteredImage_medfilt2_5x5);
title('medfilt2 5x5');

subplot(2, 2, 4);
imshow(filtered_image_standard_5x5);
title('My median filter 5x5');

% Display comparison results
fprintf('Are the results of the 3x3 filters the same? %d\n', areEqual_3x3);
fprintf('Are the results of the 5x5 filters the same? %d\n', areEqual_5x5);
disp('-----Finish Solving Part d-----');
pause;

% Part e
% Load the Moon image
moonImage = imread('/Users/muku/Downloads/moon.jpg');

% Define the Laplacian filter kernel
laplacianFilter = [0 1 0; 1 -4 1; 0 1 0];

% Apply the Laplacian filter using imfilter
filteredImage = imfilter(moonImage, laplacianFilter);

% Scale the filtered image for better visualization
scaledFilteredImage = filteredImage / max(abs(filteredImage(:)));

% Calculate the enhanced image
enhancedImage = moonImage - filteredImage;

% Display the images
figure(3);

% Original image
subplot(2, 2, 1);
imshow(moonImage);
title('Original Image');

% Filtered image
subplot(2, 2, 2);
imshow(filteredImage);
title('Filtered Image');

% Scaled filtered image (for better visualization)
subplot(2, 2, 3);
imshow(scaledFilteredImage);
title('Scaled Filtered Image');

% Enhanced image
subplot(2, 2, 4);
imshow(enhancedImage);
title('Enhanced Image');

% Adjust the figure layout
set(gcf, 'Position', [100, 100, 800, 600]);
disp('-----Finish Solving Part e-----');
pause;
end



