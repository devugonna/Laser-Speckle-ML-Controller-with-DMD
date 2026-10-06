% SPECKLE_CONTRAST  Stage 01 - local speckle contrast (LSCI) analysis
%   Needs hardware : NO (reads a captured frame from the dataset)
%   Input          : data/dataset/input_01501.png  (index 1501 = all-off frame)
%                    Run stages 02-03 first, or download the dataset (data/README.md)
%% Speckle Contrast Analysis using LSCI method
% Load image and compute local speckle contrast K = sigma/mu
% Uses 7x7 sliding window for intensity statistics

%% PATHS (relative to the repo root, so it runs from any working directory)
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end   % fallback if run via "Run Section"
repoRoot  = fileparts(scriptDir);
IMG_PATH = fullfile(repoRoot, 'data', 'dataset', 'input_01501.png');
if ~exist(IMG_PATH, 'file')
    error('Image not found: %s\nRun stages 02-03 or download the dataset (see data/README.md).', IMG_PATH);
end

%% Load and preprocess
img = imread(IMG_PATH);

if size(img, 3) == 3
    img = rgb2gray(img);
end

I = double(img);
windowSize = 7;

%% Compute speckle contrast K
h = fspecial('average', windowSize);

mu = imfilter(I, h, 'replicate');          % local mean
mu2 = imfilter(I.^2, h, 'replicate');     % local mean of squared intensity
sigma = sqrt(mu2 - mu.^2);                 % local standard deviation
K = sigma ./ (mu + eps);                  % speckle contrast

%% Display results
figure;
subplot(1,2,1);
imshow(I, []);
title('Input Image');
colorbar;

subplot(1,2,2);
imshow(K, []);
colorbar;
title(sprintf('Speckle Contrast K (window=%dx%d)', windowSize, windowSize));

%% Overlay visualization
Knorm = mat2gray(K);
overlay = ind2rgb(gray2ind(Knorm, 256), jet(256));

figure;
imshow(img, []);
hold on;
h_overlay = imshow(overlay);
set(h_overlay, 'AlphaData', 0.5);
title('Speckle Contrast Overlay (jet colormap, 50% transparent)');

%% Summary statistics
fprintf('=== Speckle Contrast Analysis ===\n');
fprintf('Image: %s\n', IMG_PATH);
fprintf('Window size: %d x %d\n', windowSize, windowSize);
fprintf('Mean K (overall): %.4f\n', mean(K(:)));
fprintf('Std K: %.4f\n', std(K(:)));
fprintf('Min K: %.4f\n', min(K(:)));
fprintf('Max K: %.4f\n', max(K(:)));