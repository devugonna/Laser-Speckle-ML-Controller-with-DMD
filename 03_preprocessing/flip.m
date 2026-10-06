% FLIP  Stage 03b - vertical flip correction (in place)
%   Needs hardware : NO
%   Operates on    : data/dataset/input_*.png   (run ONCE only - flipping twice undoes it)
%   Next: run shift1.m
%% Batch vertical flip - dataset folder
% Flips each image top-to-bottom (flipud), overwrites in place

%% PATHS (relative to the repo root, so it runs from any working directory)
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end   % fallback if run via "Run Section"
repoRoot  = fileparts(scriptDir);
INPUT_DIR = fullfile(repoRoot, 'data', 'dataset');

files = dir(fullfile(INPUT_DIR, 'input_*.png'));   % inputs only

if isempty(files)
    error('No PNG files found in: %s', INPUT_DIR);
end

fprintf('=== Vertical flip — %d images ===\n', numel(files));
fprintf('  Folder: %s\n\n', INPUT_DIR);

tStart = tic;

for k = 1:numel(files)
    fpath = fullfile(INPUT_DIR, files(k).name);
    img = imread(fpath);
    imwrite(flipud(img), fpath);

    if mod(k, 100) == 0 || k == numel(files)
        fprintf('  [%5d / %5d]  elapsed: %.1f s\n', k, numel(files), toc(tStart));
    end
end

fprintf('\nDone — %d images flipped in %.2f s\n', numel(files), toc(tStart));