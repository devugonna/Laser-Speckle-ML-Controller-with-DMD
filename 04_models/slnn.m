%% SLNN.M - Forward and Inverse Controller MLP training with full testing suite
%   Needs hardware : NO (CPU only; Deep Learning Toolbox)
%   Input          : data/dataset/  (input_#####.png + output32_#####.png)
%   Output         : results/slnn/  (trained models .mat, split indices, figures, CSVs)
%   Next: run 05_evaluation/model_eval3.m
clear; clc; close all;
%% PATHS (relative to the repo root, so it runs from any working directory)
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end   % fallback if run via "Run Section"
repoRoot  = fileparts(scriptDir);


%% 1. CONFIGURATION
config = struct();
config.data_dir    = fullfile(repoRoot, 'data', 'dataset');
config.output_dir  = fullfile(repoRoot, 'results', 'slnn');
config.num_samples = 9999;

config.input_size  = [64, 64];
config.input_dim   = prod(config.input_size);    % 4096
config.target_size = [96, 96];
config.target_dim  = prod(config.target_size);   % 9216
config.output_size = [32, 32];
config.output_dim  = prod(config.output_size);   % 1024

config.train_ratio   = 0.8;
config.val_ratio     = 0.1;
config.test_ratio    = 0.1;
config.epochs        = 30;
config.learning_rate = 0.0001;
config.batch_size    = 128;
config.shuffle       = 'every-epoch';

LATENCY_REPS = 100;
THRESH_SWEEP = 0.1 : 0.1 : 0.9;

if ~exist(config.output_dir, 'dir'), mkdir(config.output_dir); end

fprintf('=== SLNN Forward / Inverse-Controller Training (Full Test Suite) ===\n');
metrics_log = {};

%% 2. DATA LOADING
fprintf('\n--- Loading data ---\n');
t_load = tic;

X_speckle = zeros(config.num_samples, config.input_dim,  'single');
X_target  = zeros(config.num_samples, config.target_dim, 'single');
Y_dmd     = zeros(config.num_samples, config.output_dim, 'single');

for i = 1:config.num_samples
    input_file  = fullfile(config.data_dir, sprintf('input_%05d.png',    i));
    output_file = fullfile(config.data_dir, sprintf('output32_%05d.png', i));

    tmp           = loadAndNormalize(input_file, config.input_size);
    X_speckle(i,:) = tmp(:)';

    tmp           = loadAndNormalize(input_file, config.target_size);
    X_target(i,:) = tmp(:)';

    tmp          = loadAndNormalize(output_file, config.output_size);
    Y_dmd(i,:)   = tmp(:)';

    if mod(i,500) == 0
        fprintf('  Loaded %d/%d (%.1f s)\n', i, config.num_samples, toc(t_load));
    end
end
load_time = toc(t_load);
fprintf('Data loaded: %.2f s\n', load_time);
metrics_log{end+1} = {'DataLoadTime_s', load_time};

% Dataset speckle contrast
speckle_flat   = double(X_speckle);
K_dataset_mean = mean(std(speckle_flat, 0, 2) ./ (mean(speckle_flat, 2) + 1e-8));
fprintf('Dataset-level mean speckle contrast K = %.4f\n', K_dataset_mean);
metrics_log{end+1} = {'Dataset_SpeckleK_mean', K_dataset_mean};

%% 3. TRAIN / VAL / TEST SPLIT
rng(42);
perm    = randperm(config.num_samples);
n_train = round(config.num_samples * config.train_ratio);
n_val   = round(config.num_samples * config.val_ratio);

idx_train = perm(1 : n_train);
idx_val   = perm(n_train+1 : n_train+n_val);
idx_test  = perm(n_train+n_val+1 : end);

X_train = X_speckle(idx_train,:);  Y_train = Y_dmd(idx_train,:);
X_val   = X_speckle(idx_val,  :);  Y_val   = Y_dmd(idx_val,  :);
X_test  = X_speckle(idx_test, :);  Y_test  = Y_dmd(idx_test, :);

T_train = X_target(idx_train,:);
T_val   = X_target(idx_val,  :);
T_test  = X_target(idx_test, :);

metrics_log{end+1} = {'N_train', n_train};
metrics_log{end+1} = {'N_val',   n_val};
metrics_log{end+1} = {'N_test',  numel(idx_test)};

save(fullfile(config.output_dir,'slnn_training_data.mat'), ...
     'idx_train','idx_val','idx_test','config','-v7.3');

%% 4. FORWARD MODEL (Speckle --> DMD)
fprintf('\n=== Training FORWARD model ===\n');

layers_forward = [
    featureInputLayer(config.input_dim,'Normalization','none','Name','speckle_in')
    fullyConnectedLayer(2048,'Name','fc1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    dropoutLayer(0.3,'Name','drop1')
    fullyConnectedLayer(1024,'Name','fc2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    dropoutLayer(0.2,'Name','drop2')
    fullyConnectedLayer(config.output_dim,'Name','fc_out')
    sigmoidLayer('Name','sigmoid_out')
    regressionLayer('Name','regression_out')
];

options_forward = trainingOptions('adam', ...
    'InitialLearnRate',     config.learning_rate, ...
    'MaxEpochs',            config.epochs, ...
    'MiniBatchSize',        config.batch_size, ...
    'Shuffle',              config.shuffle, ...
    'ValidationData',       {X_val, Y_val}, ...
    'ValidationFrequency',  30, ...
    'Plots',                'training-progress', ...
    'Verbose',              true, ...
    'OutputNetwork',        'best-validation-loss', ...
    'ExecutionEnvironment', 'cpu');

t_fwd = tic;
[net_forward, info_forward] = trainNetwork(X_train, Y_train, layers_forward, options_forward);
time_forward = toc(t_fwd);
metrics_log{end+1} = {'FWD_TrainingTime_s', time_forward};

export_training_curve(info_forward, config.output_dir, 'forward');
save(fullfile(config.output_dir,'slnn_forward_model.mat'), 'net_forward','config','-v7.3');

%% 5. INVERSE CONTROLLER MODEL (Target --> DMD)
fprintf('\n=== Training INVERSE CONTROLLER model ===\n');

layers_inverse = [
    featureInputLayer(config.target_dim,'Normalization','none','Name','target_in')
    fullyConnectedLayer(config.output_dim*2,'Name','fc1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    dropoutLayer(0.3,'Name','drop1')
    fullyConnectedLayer(config.output_dim,'Name','fc_out')
    sigmoidLayer('Name','sigmoid_out')
    regressionLayer('Name','regression_out')
];

options_inverse = trainingOptions('adam', ...
    'InitialLearnRate',     config.learning_rate, ...
    'MaxEpochs',            config.epochs, ...
    'MiniBatchSize',        config.batch_size, ...
    'Shuffle',              config.shuffle, ...
    'ValidationData',       {T_val, Y_val}, ...
    'ValidationFrequency',  30, ...
    'Plots',                'training-progress', ...
    'Verbose',              true, ...
    'OutputNetwork',        'best-validation-loss', ...
    'ExecutionEnvironment', 'cpu');

t_inv = tic;
[net_inverse, info_inverse] = trainNetwork(T_train, Y_train, layers_inverse, options_inverse);
time_inverse = toc(t_inv);
metrics_log{end+1} = {'INV_TrainingTime_s', time_inverse};

export_training_curve(info_inverse, config.output_dir, 'inverse');
save(fullfile(config.output_dir,'slnn_inverse_controller_model.mat'), 'net_inverse','config','-v7.3');

%% 6. FORWARD MODEL TEST EVALUATION
fprintf('\n=== Evaluating FORWARD model on test set ===\n');

Y_pred_fwd     = predict(net_forward, X_test);
Y_pred_fwd_bin = double(Y_pred_fwd >= 0.5);
Y_test_d       = double(Y_test);

mse_fwd  = mean((Y_test_d(:) - double(Y_pred_fwd(:))).^2);
rmse_fwd = sqrt(mse_fwd);
acc_fwd  = mean(Y_pred_fwd_bin(:) == Y_test_d(:));

[ssim_fwd, psnr_fwd, speckleK_fwd, jaccard_fwd] = ...
    per_sample_metrics(Y_test_d, Y_pred_fwd_bin, config.output_size);

[cm_fwd, prec_fwd, rec_fwd, f1_fwd] = pixel_confusion(Y_test_d, Y_pred_fwd_bin);
heatmap_fwd = pixel_mae_heatmap(Y_test_d, double(Y_pred_fwd), config.output_size);
thresh_acc_fwd = threshold_sweep(Y_test_d, double(Y_pred_fwd), THRESH_SWEEP);
[lat_mean_fwd, lat_std_fwd] = latency_benchmark(net_forward, X_test(1,:), LATENCY_REPS, 'forward');

fprintf('\n--- FORWARD MODEL RESULTS ---\n');
fprintf('  MSE: %.6f, RMSE: %.6f, Binary Acc: %.2f%%\n', mse_fwd, rmse_fwd, acc_fwd*100);
fprintf('  Precision: %.4f, Recall: %.4f, F1: %.4f\n', prec_fwd, rec_fwd, f1_fwd);
fprintf('  Jaccard: %.4f, SSIM: %.4f, PSNR: %.2f dB, SpeckleK: %.4f\n', ...
    mean(jaccard_fwd), mean(ssim_fwd), mean(psnr_fwd), mean(speckleK_fwd));
fprintf('  Latency: %.3f +/- %.3f ms\n', lat_mean_fwd, lat_std_fwd);

% Log forward metrics
metrics_log{end+1} = {'FWD_MSE', mse_fwd}; metrics_log{end+1} = {'FWD_RMSE', rmse_fwd};
metrics_log{end+1} = {'FWD_BinaryAcc', acc_fwd}; metrics_log{end+1} = {'FWD_F1', f1_fwd};
metrics_log{end+1} = {'FWD_Jaccard_mean', mean(jaccard_fwd)}; metrics_log{end+1} = {'FWD_SSIM_mean', mean(ssim_fwd)};
metrics_log{end+1} = {'FWD_PSNR_mean_dB', mean(psnr_fwd)}; metrics_log{end+1} = {'FWD_SpeckleK_mean', mean(speckleK_fwd)};
metrics_log{end+1} = {'FWD_Latency_mean_ms', lat_mean_fwd};

%% 7. INVERSE MODEL TEST EVALUATION
fprintf('\n=== Evaluating INVERSE CONTROLLER model on test set ===\n');

Y_pred_inv     = predict(net_inverse, T_test);
Y_pred_inv_bin = double(Y_pred_inv >= 0.5);

mse_inv  = mean((Y_test_d(:) - double(Y_pred_inv(:))).^2);
rmse_inv = sqrt(mse_inv);
acc_inv  = mean(Y_pred_inv_bin(:) == Y_test_d(:));

[ssim_inv, psnr_inv, speckleK_inv, jaccard_inv] = ...
    per_sample_metrics(Y_test_d, Y_pred_inv_bin, config.output_size);

[cm_inv, prec_inv, rec_inv, f1_inv] = pixel_confusion(Y_test_d, Y_pred_inv_bin);
heatmap_inv = pixel_mae_heatmap(Y_test_d, double(Y_pred_inv), config.output_size);
thresh_acc_inv = threshold_sweep(Y_test_d, double(Y_pred_inv), THRESH_SWEEP);
[lat_mean_inv, lat_std_inv] = latency_benchmark(net_inverse, T_test(1,:), LATENCY_REPS, 'inverse');

fprintf('\n--- INVERSE CONTROLLER RESULTS ---\n');
fprintf('  MSE: %.6f, RMSE: %.6f, Binary Acc: %.2f%%\n', mse_inv, rmse_inv, acc_inv*100);
fprintf('  Precision: %.4f, Recall: %.4f, F1: %.4f\n', prec_inv, rec_inv, f1_inv);
fprintf('  Jaccard: %.4f, SSIM: %.4f, PSNR: %.2f dB, SpeckleK: %.4f\n', ...
    mean(jaccard_inv), mean(ssim_inv), mean(psnr_inv), mean(speckleK_inv));
fprintf('  Latency: %.3f +/- %.3f ms\n', lat_mean_inv, lat_std_inv);

metrics_log{end+1} = {'INV_MSE', mse_inv}; metrics_log{end+1} = {'INV_F1', f1_inv};
metrics_log{end+1} = {'INV_Jaccard_mean', mean(jaccard_inv)}; metrics_log{end+1} = {'INV_SSIM_mean', mean(ssim_inv)};
metrics_log{end+1} = {'INV_PSNR_mean_dB', mean(psnr_inv)}; metrics_log{end+1} = {'INV_Latency_mean_ms', lat_mean_inv};

%% 8. FIGURES - METRIC DISTRIBUTIONS
fprintf('\n--- Generating figures ---\n');

% SSIM distribution
fig_ssim = figure('Position',[50 50 900 420],'Visible','off');
histogram(ssim_fwd, 30, 'FaceColor',[0.20 0.60 0.80],'EdgeColor','none','FaceAlpha',0.75); hold on;
histogram(ssim_inv, 30, 'FaceColor',[0.85 0.33 0.10],'EdgeColor','none','FaceAlpha',0.65);
xlabel('SSIM'); ylabel('Count'); legend('Forward','Inverse'); grid on;
saveas(fig_ssim, fullfile(config.output_dir,'slnn_ssim_distribution.png')); close(fig_ssim);

% PSNR distribution
fig_psnr = figure('Position',[50 50 900 420],'Visible','off');
histogram(psnr_fwd, 30, 'FaceColor',[0.20 0.60 0.80],'EdgeColor','none','FaceAlpha',0.75); hold on;
histogram(psnr_inv, 30, 'FaceColor',[0.85 0.33 0.10],'EdgeColor','none','FaceAlpha',0.65);
xlabel('PSNR (dB)'); ylabel('Count'); legend('Forward','Inverse'); grid on;
saveas(fig_psnr, fullfile(config.output_dir,'slnn_psnr_distribution.png')); close(fig_psnr);

% Speckle contrast distribution
fig_k = figure('Position',[50 50 900 420],'Visible','off');
histogram(speckleK_fwd, 30,'FaceColor',[0.20 0.60 0.80],'EdgeColor','none','FaceAlpha',0.75); hold on;
histogram(speckleK_inv, 30,'FaceColor',[0.85 0.33 0.10],'EdgeColor','none','FaceAlpha',0.65);
xlabel('Speckle Contrast K'); ylabel('Count'); legend('Forward','Inverse'); grid on;
saveas(fig_k, fullfile(config.output_dir,'slnn_speckle_K_distribution.png')); close(fig_k);

% Pixel MAE heatmaps
fig_heat = figure('Position',[50 50 900 380],'Visible','off');
subplot(1,2,1); imagesc(heatmap_fwd); colorbar; title('Forward: Mean |Error|'); axis square;
subplot(1,2,2); imagesc(heatmap_inv); colorbar; title('Inverse: Mean |Error|'); axis square;
sgtitle('Pixel-wise Mean Absolute Error (32x32 DMD)');
saveas(fig_heat, fullfile(config.output_dir,'slnn_pixel_mae_heatmap.png')); close(fig_heat);

% Threshold sweep
fig_thresh = figure('Position',[50 50 800 420],'Visible','off');
plot(THRESH_SWEEP, thresh_acc_fwd*100, 'b-o','LineWidth',1.8); hold on;
plot(THRESH_SWEEP, thresh_acc_inv*100, 'r-s','LineWidth',1.8);
xlabel('Binarisation Threshold'); ylabel('Binary Accuracy (%)'); legend('Forward','Inverse'); grid on;
saveas(fig_thresh, fullfile(config.output_dir,'slnn_threshold_sweep.png')); close(fig_thresh);

% Confusion matrices
plot_confusion(cm_fwd, config.output_dir, 'forward');
plot_confusion(cm_inv, config.output_dir, 'inverse');

% Latency comparison
fig_lat = figure('Position',[50 50 600 420],'Visible','off');
b = bar([lat_mean_fwd, lat_mean_inv], 0.5, 'FaceColor','flat');
b.CData = [0.20 0.60 0.80; 0.85 0.33 0.10]; hold on;
errorbar([1,2],[lat_mean_fwd,lat_mean_inv],[lat_std_fwd,lat_std_inv],'k.','LineWidth',1.5);
set(gca,'XTick',[1 2],'XTickLabel',{'Forward','Inverse'});
ylabel('Inference latency (ms)'); title('Single-sample inference latency'); grid on;
saveas(fig_lat, fullfile(config.output_dir,'slnn_latency_comparison.png')); close(fig_lat);

%% 9. GENERALISATION PROBE
fprintf('\n--- Generalisation probe (novel Gaussian targets) ---\n');

[gx, gy] = meshgrid(linspace(-1,1,config.target_size(2)), linspace(-1,1,config.target_size(1)));
probe_sigmas = [0.05, 0.08, 0.12, 0.20, 0.30];
n_probe = numel(probe_sigmas);
probe_ssim = zeros(n_probe,1);
probe_speckleK = zeros(n_probe,1);

fig_gen = figure('Position',[50 50 1400 480],'Visible','off');
for k = 1:n_probe
    sigma = probe_sigmas(k);
    tgt = exp(-(gx.^2 + gy.^2) ./ (2*sigma^2));
    tgt = tgt / max(tgt(:));
    tgt_v = single(tgt(:))';
    
    dmd_r = predict(net_inverse, tgt_v);
    dmd_bin = double(dmd_r >= 0.5);
    dmd_img = reshape(dmd_bin, config.output_size);
    
    probe_ssim(k) = ssim(dmd_img, zeros(config.output_size));
    if mean(dmd_img(:)) > 1e-6
        probe_speckleK(k) = std(dmd_img(:)) / mean(dmd_img(:));
    else
        probe_speckleK(k) = NaN;
    end
    
    subplot(2, n_probe, k);
    imshow(tgt,[]); title(sprintf('sig=%.2f', sigma));
    subplot(2, n_probe, n_probe+k);
    imshow(dmd_img,[]); title(sprintf('K=%.3f', probe_speckleK(k)));
end
sgtitle('Generalisation Probe: Gaussian Targets -> Inverse Controller DMD');
saveas(fig_gen, fullfile(config.output_dir,'slnn_generalisation_probe.png')); close(fig_gen);

%% 10. WRITE METRICS SUMMARY CSV
csv_path = fullfile(config.output_dir,'slnn_metrics_summary.csv');
fid = fopen(csv_path,'w');
fprintf(fid,'Metric,Value\n');
for r = 1:numel(metrics_log)
    entry = metrics_log{r};
    if isnumeric(entry{2})
        fprintf(fid,'%s,%.8f\n', entry{1}, entry{2});
    else
        fprintf(fid,'%s,%s\n', entry{1}, entry{2});
    end
end
fclose(fid);
fprintf('\nMetrics summary saved -> slnn_metrics_summary.csv\n');
fprintf('\n=== Training complete. All outputs in: %s ===\n', config.output_dir);

%% LOCAL FUNCTIONS

function [ssim_v, psnr_v, k_v, jac_v] = per_sample_metrics(Y_true, Y_bin, out_sz)
    n = size(Y_true, 1);
    ssim_v = zeros(n,1); psnr_v = zeros(n,1); k_v = zeros(n,1); jac_v = zeros(n,1);
    for i = 1:n
        gt = reshape(Y_true(i,:), out_sz);
        pred = reshape(Y_bin(i,:), out_sz);
        ssim_v(i) = ssim(pred, gt);
        psnr_v(i) = psnr(pred, gt, 1.0);
        if mean(pred(:)) > 1e-6, k_v(i) = std(pred(:)) / mean(pred(:)); else, k_v(i) = NaN; end
        inter = sum(pred(:) & gt(:));
        uni = sum(pred(:) | gt(:));
        if uni > 0, jac_v(i) = inter/uni; else, jac_v(i) = 1.0; end
    end
    k_v(isnan(k_v)) = mean(k_v(~isnan(k_v)));
end

function [cm, precision, recall, f1] = pixel_confusion(Y_true, Y_bin)
    gt_flat = logical(Y_true(:));
    pred_flat = logical(Y_bin(:));
    TP = sum(pred_flat & gt_flat);
    FP = sum(pred_flat & ~gt_flat);
    FN = sum(~pred_flat & gt_flat);
    TN = sum(~pred_flat & ~gt_flat);
    cm = [TP, FP; FN, TN];
    precision = TP / max(TP + FP, 1);
    recall = TP / max(TP + FN, 1);
    f1 = 2*precision*recall / max(precision + recall, 1e-8);
end

function hmap = pixel_mae_heatmap(Y_true, Y_pred_soft, out_sz)
    n = size(Y_true,1);
    hmap = zeros(out_sz);
    for i = 1:n
        gt = reshape(Y_true(i,:), out_sz);
        pred = reshape(Y_pred_soft(i,:), out_sz);
        hmap = hmap + abs(gt - pred);
    end
    hmap = hmap / n;
end

function acc_vec = threshold_sweep(Y_true, Y_pred_soft, thresholds)
    acc_vec = zeros(size(thresholds));
    for t = 1:numel(thresholds)
        bin = double(Y_pred_soft >= thresholds(t));
        acc_vec(t) = mean(bin(:) == Y_true(:));
    end
end

function [mu_ms, sd_ms] = latency_benchmark(net, sample, reps, ~)
    times = zeros(reps,1);
    for r = 1:reps
        t0 = tic;
        predict(net, sample);
        times(r) = toc(t0) * 1000;
    end
    times = times(6:end);  % Warm-up discard
    mu_ms = mean(times);
    sd_ms = std(times);
end

function export_training_curve(info, out_dir, tag)
    tl = info.TrainingLoss;
    vl = info.ValidationLoss;
    vl = vl(~isnan(vl));
    
    fid = fopen(fullfile(out_dir, sprintf('slnn_%s_training_loss.csv', tag)), 'w');
    fprintf(fid, 'Iteration,TrainingLoss\n');
    for k = 1:numel(tl), fprintf(fid,'%d,%.8f\n', k, tl(k)); end
    fclose(fid);
    
    fig = figure('Position',[50 50 800 400],'Visible','off');
    plot(tl, 'Color',[0.20 0.60 0.80],'LineWidth',1.3); hold on;
    if ~isempty(vl)
        v_iters = round(linspace(1, numel(tl), numel(vl)));
        plot(v_iters, vl, 'r--o','LineWidth',1.3);
        legend('Train loss','Val loss');
    end
    xlabel('Iteration'); ylabel('MSE Loss'); title(sprintf('SLNN %s: Training curve', tag)); grid on;
    saveas(fig, fullfile(out_dir, sprintf('slnn_%s_training_curve.png', tag))); close(fig);
end

function plot_confusion(cm, out_dir, tag)
    fig = figure('Position',[50 50 500 420],'Visible','off');
    imagesc(cm); colormap(gca,'parula'); colorbar;
    set(gca,'XTick',[1 2],'XTickLabel',{'On (1)','Off (0)'}, ...
            'YTick',[1 2],'YTickLabel',{'On (1)','Off (0)'});
    xlabel('Predicted'); ylabel('True'); title(sprintf('Confusion Matrix: %s', tag));
    for r = 1:2, for c = 1:2, text(c, r, num2str(cm(r,c)), 'HorizontalAlignment','center', 'FontSize',14,'Color','w'); end, end
    saveas(fig, fullfile(out_dir, sprintf('slnn_%s_confusion_matrix.png', tag))); close(fig);
end

function img_norm = loadAndNormalize(filepath, target_size)
    if exist(filepath,'file')
        img = imread(filepath);
        if size(img,3) > 1, img = rgb2gray(img); end
        img = imresize(img, target_size);
        img_norm = single(img) / 255.0;
    else
        warning('File not found: %s', filepath);
        img_norm = zeros(target_size,'single');
    end
end