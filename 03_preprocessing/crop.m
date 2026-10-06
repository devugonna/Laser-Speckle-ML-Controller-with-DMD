% CROP  Stage 03a - batch centre-crop of raw camera frames
%   Needs hardware : NO
%   Input          : data/raw/finalinp/*.png            (from rpatt.m)
%   Output         : data/dataset/input_#####.png       (640x640)
%                    data/dataset/output32_#####.png    (ground truth, copied across
%                                                        so inputs and targets sit together)
%   Next: run flip.m, then shift1.m
%% Batch centre-crop: 800x640 -> 640x640
% Removes 80px from left and right edges equally

%% PARAMETERS
SRC_W = 800;   % original width
SRC_H = 640;   % original height
CROP  = 640;   % target square size

TRIM_X = (SRC_W - CROP) / 2;   % 80px from each side
TRIM_Y = (SRC_H - CROP) / 2;   % 0 (height already 640)

X1 = TRIM_X + 1;
X2 = TRIM_X + CROP;
Y1 = TRIM_Y + 1;
Y2 = TRIM_Y + CROP;

%% PATHS (relative to the repo root, so it runs from any working directory)
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end   % fallback if run via "Run Section"
repoRoot  = fileparts(scriptDir);
RAW_DIR    = fullfile(repoRoot, 'data', 'raw');
INPUT_DIR  = fullfile(RAW_DIR, 'finalinp');
OUTPUT_DIR = fullfile(repoRoot, 'data', 'dataset');

if ~exist(OUTPUT_DIR, 'dir'), mkdir(OUTPUT_DIR); end

%% FIND ALL PNGs
files = dir(fullfile(INPUT_DIR, '*.png'));

if isempty(files)
    error('No PNG files found in: %s', INPUT_DIR);
end

fprintf('=== Centre-crop %dx%d -> %dx%d ===\n', SRC_W, SRC_H, CROP, CROP);
fprintf('  Trimming %dpx from left and right\n', TRIM_X);
fprintf('  Source : %s\n', INPUT_DIR);
fprintf('  Output : %s\n', OUTPUT_DIR);
fprintf('  Files  : %d\n\n', numel(files));

%% CROP LOOP
tStart = tic;

for k = 1:numel(files)
    srcPath = fullfile(INPUT_DIR, files(k).name);
    dstPath = fullfile(OUTPUT_DIR, files(k).name);

    img = imread(srcPath);
    cropped = img(Y1:Y2, X1:X2, :);   % [H x W x C]
    imwrite(cropped, dstPath);

    if mod(k, 100) == 0 || k == numel(files)
        fprintf('  [%5d / %5d]  elapsed: %.1f s\n', k, numel(files), toc(tStart));
    end
end

fprintf('\nDone — %d images cropped in %.2f s\n', numel(files), toc(tStart));

%% COPY GROUND-TRUTH 32x32 PATTERNS NEXT TO THE CROPPED INPUTS
% (originally done by hand; later stages expect input_*.png and output32_*.png
%  in the same folder)
GT_DIR = fullfile(RAW_DIR, 'finaloutp', '32x32');
if exist(GT_DIR, 'dir')
    copyfile(fullfile(GT_DIR, 'output32_*.png'), OUTPUT_DIR);
    fprintf('Copied ground-truth patterns from %s\n', GT_DIR);
else
    warning('Ground-truth folder not found: %s (copy output32_*.png into %s manually)', GT_DIR, OUTPUT_DIR);
end
