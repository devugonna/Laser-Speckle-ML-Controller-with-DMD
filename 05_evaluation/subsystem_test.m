%% MLP SUBSYSTEM TEST -- matches slnn.m pipeline
%   Needs hardware : NO
%   Input          : data/subsystem_test/{train,val,test}/  (small paired PNG subset:
%                    input_*.png + output32_*.png; see data/README.md)
%   Output         : results/subsystem_test/mlp_subsystem_test_results.mat
% Task: Reconstruction -- speckle 320x320 --> DMD 32x32 [0,1]
% Mirrors FORWARD model in slnn.m exactly but smaller/faster

clear; clc;

%% PATHS (relative to the repo root, so it runs from any working directory)
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end   % fallback if run via "Run Section"
repoRoot  = fileparts(scriptDir);

%% 1. Paths
root = fullfile(repoRoot, 'data', 'subsystem_test');
train_folder = fullfile(root, 'train');
val_folder   = fullfile(root, 'val');
test_folder  = fullfile(root, 'test');

INPUT_SIZE  = [320, 320];
OUTPUT_SIZE = [32,  32];
input_dim   = prod(INPUT_SIZE);   % 102400
output_dim  = prod(OUTPUT_SIZE);  % 1024

%% 2. Load paired PNG datasets
fprintf('Loading datasets...\n');
[X_train, Y_train] = load_paired_dataset(train_folder, INPUT_SIZE, OUTPUT_SIZE);
[X_val,   Y_val]   = load_paired_dataset(val_folder,   INPUT_SIZE, OUTPUT_SIZE);
[X_test,  Y_test]  = load_paired_dataset(test_folder,  INPUT_SIZE, OUTPUT_SIZE);

fprintf('  Train: %d pairs\n', size(X_train,1));
fprintf('  Val:   %d pairs\n', size(X_val,1));
fprintf('  Test:  %d pairs\n', size(X_test,1));

if size(X_train,1) == 0
    error('No training pairs found. Check folder paths and file naming.');
end

%% 3. Network (scaled-down slnn.m forward model)
layers_forward = [
    featureInputLayer(input_dim, 'Normalization', 'none', 'Name', 'speckle_in')
    fullyConnectedLayer(256, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.3, 'Name', 'drop1')
    fullyConnectedLayer(128, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(output_dim, 'Name', 'fc_out')
    sigmoidLayer('Name', 'sigmoid_out')
    regressionLayer('Name', 'regression_out')
];

%% 4. Training options
val_data = [];
if size(X_val,1) > 0
    val_data = {X_val, Y_val};
end

opts = trainingOptions('adam', ...
    'InitialLearnRate',    1e-4, ...
    'MaxEpochs',           10, ...
    'MiniBatchSize',       max(1, min(4, size(X_train,1))), ...
    'Shuffle',             'every-epoch', ...
    'ValidationData',      val_data, ...
    'ValidationFrequency', 5, ...
    'Plots',               'training-progress', ...
    'Verbose',             true, ...
    'ExecutionEnvironment','cpu');

%% 5. Subsystem checks
fprintf('\n=== Running subsystem checks ===\n');
passed = 0; total = 0;
results = struct();

% CHECK 1: Architecture
total = total + 1;
try
    analyzeNetwork(layers_forward);
    results.architecture = 'PASS';
    passed = passed + 1;
    fprintf('[1] Architecture valid: PASS\n');
catch ME
    results.architecture = ['FAIL: ' ME.message];
    fprintf('[1] Architecture valid: FAIL -- %s\n', ME.message);
end

% CHECK 2: Training
total = total + 1;
try
    net = trainNetwork(X_train, Y_train, layers_forward, opts);
    results.training = 'PASS';
    passed = passed + 1;
    fprintf('[2] Training completed: PASS\n');
catch ME
    results.training = ['FAIL: ' ME.message];
    fprintf('[2] Training completed: FAIL -- %s\n', ME.message);
    fprintf('Cannot continue without a trained network.\n');
    fprintf('Passed %d / %d checks.\n', passed, total);
    return;
end

Y_pred = predict(net, X_test);          % [n x 1024], values in [0,1]
Y_pred_bin = double(Y_pred >= 0.5);     % thresholded binary DMD pattern
Y_true = double(Y_test);

% CHECK 3: Output shape
total = total + 1;
if isequal(size(Y_pred), [size(X_test,1), output_dim])
    results.output_shape = 'PASS';
    passed = passed + 1;
    fprintf('[3] Output shape [%d x %d]: PASS\n', size(Y_pred,1), output_dim);
else
    results.output_shape = 'FAIL';
    fprintf('[3] Output shape: FAIL\n');
end

% CHECK 4: Sigmoid output in [0,1]
total = total + 1;
if min(Y_pred(:)) >= -1e-6 && max(Y_pred(:)) <= 1+1e-6
    results.output_range = 'PASS';
    passed = passed + 1;
    fprintf('[4] Output range [0,1]: PASS (min=%.4f max=%.4f)\n', ...
        min(Y_pred(:)), max(Y_pred(:)));
else
    results.output_range = 'FAIL';
    fprintf('[4] Output range [0,1]: FAIL (min=%.4f max=%.4f)\n', ...
        min(Y_pred(:)), max(Y_pred(:)));
end

% CHECK 5: Reconstruction MSE beats all-zeros baseline
total = total + 1;
mse_model = mean((Y_true(:) - double(Y_pred(:))).^2);
mse_baseline = mean(Y_true(:).^2);
results.mse_model = mse_model;
results.mse_baseline = mse_baseline;
if mse_model < mse_baseline
    passed = passed + 1;
    results.mse_check = 'PASS';
    fprintf('[5] MSE reconstruction < baseline: PASS (MSE=%.5f baseline=%.5f)\n', ...
        mse_model, mse_baseline);
else
    results.mse_check = 'FAIL';
    fprintf('[5] MSE reconstruction < baseline: FAIL (MSE=%.5f baseline=%.5f)\n', ...
        mse_model, mse_baseline);
end

% CHECK 6: Binary pixel accuracy
total = total + 1;
bin_acc = mean(Y_pred_bin(:) == Y_true(:));
results.binary_accuracy = bin_acc;
if bin_acc > 0.5
    passed = passed + 1;
    results.bin_acc_check = 'PASS';
    fprintf('[6] Binary pixel accuracy > 50%%: PASS (acc=%.2f%%)\n', bin_acc*100);
else
    results.bin_acc_check = 'FAIL';
    fprintf('[6] Binary pixel accuracy > 50%%: FAIL (acc=%.2f%%)\n', bin_acc*100);
end

% CHECK 7: Pixel-wise correlation
total = total + 1;
r_vals = zeros(size(Y_test,1), 1);
for s = 1:size(Y_test,1)
    c = corrcoef(Y_pred(s,:)', Y_true(s,:)');
    r_vals(s) = c(1,2);
end
mean_r = mean(r_vals, 'omitnan');
results.mean_correlation = mean_r;
if mean_r > 0
    passed = passed + 1;
    results.corr_check = 'PASS';
    fprintf('[7] Pixel correlation with true DMD: PASS (mean r=%.4f)\n', mean_r);
else
    results.corr_check = 'FAIL';
    fprintf('[7] Pixel correlation with true DMD: FAIL (mean r=%.4f)\n', mean_r);
end

% CHECK 8: Visual diff
total = total + 1;
try
    n_vis = min(3, size(X_test,1));
    fig = figure('Name', 'DMD Reconstruction -- Subsystem Test', ...
                 'Position', [80 80 220*n_vis 600]);

    for s = 1:n_vis
        speckle_in   = reshape(X_test(s,:), INPUT_SIZE);
        dmd_true     = reshape(Y_true(s,:), OUTPUT_SIZE);
        dmd_pred_raw = reshape(double(Y_pred(s,:)), OUTPUT_SIZE);
        dmd_pred_bin = reshape(Y_pred_bin(s,:), OUTPUT_SIZE);
        diff_img     = abs(dmd_true - dmd_pred_bin);

        cols = 5;
        subplot(n_vis, cols, (s-1)*cols + 1);
        imshow(speckle_in, []); title(sprintf('Speckle #%d', s), 'FontSize', 8);

        subplot(n_vis, cols, (s-1)*cols + 2);
        imshow(dmd_true, []); title('True DMD', 'FontSize', 8);

        subplot(n_vis, cols, (s-1)*cols + 3);
        imshow(dmd_pred_raw, []); title('Pred (sigmoid)', 'FontSize', 8);

        subplot(n_vis, cols, (s-1)*cols + 4);
        imshow(dmd_pred_bin, []); title('Pred (binary)', 'FontSize', 8);

        subplot(n_vis, cols, (s-1)*cols + 5);
        imshow(diff_img, []);
        colormap(gca, 'hot');
        title(sprintf('Error\n(%.0f%% wrong)', mean(diff_img(:))*100), 'FontSize', 8);
    end

    sgtitle('MLP Subsystem Test: DMD Pattern Reconstruction');
    results.visualisation = 'PASS';
    passed = passed + 1;
    fprintf('[8] Visualisation rendered: PASS\n');
catch ME
    results.visualisation = ['FAIL: ' ME.message];
    fprintf('[8] Visualisation rendered: FAIL -- %s\n', ME.message);
end

%% 6. Summary
fprintf('\n=== Subsystem test complete: %d / %d checks passed ===\n', passed, total);
fprintf('    Reconstruction MSE:        %.6f\n', mse_model);
fprintf('    Binary pixel accuracy:     %.2f%%\n', bin_acc*100);
fprintf('    Mean pixel correlation r:  %.4f\n', mean_r);

results.passed = passed;
results.total  = total;
results.net    = net;
res_dir = fullfile(repoRoot, 'results', 'subsystem_test');
if ~exist(res_dir, 'dir'), mkdir(res_dir); end
save(fullfile(res_dir, 'mlp_subsystem_test_results.mat'), 'results');
fprintf('Results saved -> %s\n', fullfile(res_dir, 'mlp_subsystem_test_results.mat'));

%% HELPER FUNCTIONS

function [X, Y] = load_paired_dataset(folder, input_size, output_size)
    input_dim  = prod(input_size);
    output_dim = prod(output_size);

    files = dir(fullfile(folder, 'input_*.png'));
    n = numel(files);
    X = zeros(n, input_dim, 'single');
    Y = zeros(n, output_dim, 'single');
    kept = 0;

    for i = 1:n
        fname = files(i).name;
        tokens = regexp(fname, 'input_(\d+)\.png', 'tokens');
        if isempty(tokens), continue; end
        idx_str = tokens{1}{1};

        input_path = fullfile(folder, fname);
        output_path = fullfile(folder, sprintf('output32_%s.png', idx_str));

        if ~exist(output_path, 'file')
            warning('Missing output file for %s -- skipping.', fname);
            continue;
        end

        kept = kept + 1;
        X(kept, :) = load_and_normalize(input_path, input_size);
        Y(kept, :) = load_and_normalize(output_path, output_size);
    end

    X = X(1:kept, :);
    Y = Y(1:kept, :);
end

function vec = load_and_normalize(filepath, target_size)
    img = imread(filepath);
    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    img = imresize(img, target_size);
    vec = single(img(:))' / 255.0;
end