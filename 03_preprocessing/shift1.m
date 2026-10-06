% SHIFT1  Stage 03c - frame-latency alignment (I_aligned(i) <- I_raw(i+1))
% Shifts input_xxxxx filenames left by one and deletes pairs with index >= 10000
%   Needs hardware : NO
%   Operates on    : data/dataset/  (input_*.png and output32_*.png, IN PLACE - run ONCE only)
%   Next: run 04_models/slnn.m and ppo.m
%% PATHS (relative to the repo root, so it runs from any working directory)
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end   % fallback if run via "Run Section"
repoRoot  = fileparts(scriptDir);
folder = fullfile(repoRoot, 'data', 'dataset');

%% STEP 1: Collect and sort all input files
inputFiles = dir(fullfile(folder, 'input_*.png'));
inputNames = sort({inputFiles.name});
fprintf('Found %d input files.\n', numel(inputNames));

if numel(inputNames) == 0
    error('No input_*.png files found in folder.');
end

%% STEP 2: Extract numeric indices from input filenames
inputIndices = zeros(1, numel(inputNames));
for i = 1:numel(inputNames)
    tok = regexp(inputNames{i}, 'input_(\d+)\.png', 'tokens');
    if isempty(tok)
        error('Unexpected filename format: %s', inputNames{i});
    end
    inputIndices(i) = str2double(tok{1}{1});
end

%% STEP 3: Delete the first input file (smallest index)
[~, sortOrder] = sort(inputIndices);
firstFile = fullfile(folder, inputNames{sortOrder(1)});
fprintf('Deleting first input file: %s\n', inputNames{sortOrder(1)});
delete(firstFile);

% Remove from lists
inputNames(sortOrder(1))   = [];
inputIndices(sortOrder(1)) = [];

% Re-sort after deletion
[inputIndices, sortOrder] = sort(inputIndices);
inputNames = inputNames(sortOrder);
fprintf('%d input files remain after deletion.\n', numel(inputNames));

%% STEP 4: Rename inputs with a temp prefix to avoid collisions
fprintf('Renaming input files to temporary names...\n');
for i = 1:numel(inputNames)
    oldName = fullfile(folder, inputNames{i});
    tmpName = fullfile(folder, sprintf('TEMP_input_%05d.png', i));
    movefile(oldName, tmpName);
end

% Rename temp files to final input_xxxxx names
fprintf('Renaming temporary files to final input names...\n');
for i = 1:numel(inputNames)
    tmpName  = fullfile(folder, sprintf('TEMP_input_%05d.png', i));
    newName  = fullfile(folder, sprintf('input_%05d.png', i));
    movefile(tmpName, newName);
end
fprintf('Input renaming complete.\n');

%% STEP 5: Collect output32 files
outputFiles = dir(fullfile(folder, 'output32_*.png'));
outputNames = sort({outputFiles.name});
fprintf('Found %d output32 files.\n', numel(outputNames));

%% STEP 6: Delete input and output32 pairs with new index >= 10000
DELETE_FROM = 10000;  % Delete indices 10000 and above

fprintf('Deleting input files with index >= %d...\n', DELETE_FROM);
inputFilesAll = dir(fullfile(folder, 'input_*.png'));
for i = 1:numel(inputFilesAll)
    tok = regexp(inputFilesAll(i).name, 'input_(\d+)\.png', 'tokens');
    if ~isempty(tok)
        idx = str2double(tok{1}{1});
        if idx >= DELETE_FROM
            fp = fullfile(folder, inputFilesAll(i).name);
            fprintf('  Deleting: %s\n', inputFilesAll(i).name);
            delete(fp);
        end
    end
end

fprintf('Deleting output32 files with index >= %d...\n', DELETE_FROM);
for i = 1:numel(outputNames)
    tok = regexp(outputNames{i}, 'output32_(\d+)\.png', 'tokens');
    if ~isempty(tok)
        idx = str2double(tok{1}{1});
        if idx >= DELETE_FROM
            fp = fullfile(folder, outputNames{i});
            fprintf('  Deleting: %s\n', outputNames{i});
            delete(fp);
        end
    end
end

%% DONE
fprintf('\nAll done!\n');
fprintf('  - Deleted original input_00001 (first file)\n');
fprintf('  - Shifted remaining inputs left by one\n');
fprintf('  - Deleted all input_ and output32_ pairs with new index >= %d\n', DELETE_FROM);