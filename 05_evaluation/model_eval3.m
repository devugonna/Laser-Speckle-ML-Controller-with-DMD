%% model_eval3.m -- Dissertation Section 3 Model Evaluation
% Evaluates trained SLNN (MLP) and/or PPO models on saved test set
% Produces all quantitative data, figures, and CSV exports for Section 3

%   Needs hardware : NO
%   Input          : results/slnn/ and results/ppo/ (trained models; run slnn.m / ppo.m
%                    first, or download the pretrained models - see data/README.md)
%                    data/dataset/ (test samples)
%   Output         : results/evaluation/  (figures + CSV tables)
%   NOTE: the test-split indices are read from results/slnn/slnn_training_data.mat,
%         so slnn.m must have been run (or that file downloaded) even for MODEL_TYPE='ppo'.

clear; clc; close all;
%% PATHS (relative to the repo root, so it runs from any working directory)
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end   % fallback if run via "Run Section"
repoRoot  = fileparts(scriptDir);

%% 1. CONFIGURATION
MODEL_TYPE = 'inverse';  % 'forward' | 'inverse' | 'ppo' | 'compare'

cfg = struct();
cfg.slnn_model_dir = fullfile(repoRoot, 'results', 'slnn');
cfg.ppo_model_dir  = fullfile(repoRoot, 'results', 'ppo');
cfg.data_dir       = fullfile(repoRoot, 'data', 'dataset');
cfg.output_dir     = fullfile(repoRoot, 'results', 'evaluation');

% Specification thresholds (D5 merit figures)
SPEC.min_binary_accuracy = 0.70;
SPEC.min_ssim            = 0.60;
SPEC.max_mse             = 0.08;
SPEC.min_psnr_db         = 10.0;
SPEC.max_speckle_K       = 0.50;

if ~exist(cfg.output_dir, 'dir'), mkdir(cfg.output_dir); end

fprintf('=== Dissertation Section 3 -- Model Evaluation ===\n');
fprintf('Model type  : %s\n', upper(MODEL_TYPE));
fprintf('Output dir  : %s\n\n', cfg.output_dir);

%% 2. LOAD MODEL(S)
fprintf('--- Loading model(s) ---\n');

switch MODEL_TYPE
    case 'forward'
        [net_A, config_A] = load_slnn(cfg.slnn_model_dir, 'forward');
        label_A = 'MLP-Forward';
        run_compare = false;
    case 'inverse'
        [net_A, config_A] = load_slnn(cfg.slnn_model_dir, 'inverse');
        label_A = 'MLP-Inverse';
        run_compare = false;
    case 'ppo'
        [net_A, config_A] = load_ppo(cfg.ppo_model_dir);
        label_A = 'PPO';
        run_compare = false;
    case 'compare'
        [net_A, config_A] = load_slnn(cfg.slnn_model_dir, 'forward');
        label_A = 'MLP';
        [net_B, config_B] = load_ppo(cfg.ppo_model_dir);
        label_B = 'PPO';
        run_compare = true;
    otherwise
        error('Unknown MODEL_TYPE ''%s''.', MODEL_TYPE);
end

%% 3. LOAD TEST DATA
fprintf('\n--- Loading test indices ---\n');

data_file = fullfile(cfg.slnn_model_dir, 'slnn_training_data.mat');
if ~exist(data_file, 'file')
    data_file = fullfile(cfg.ppo_model_dir, 'ppo_training_data.mat');
end
if ~exist(data_file, 'file')
    error('No training-data index file found. Run slnn.m or ppo.m first (see 04_models/).');
end

ld = load(data_file, 'idx_test');
idx_test = ld.idx_test;
n_test = numel(idx_test);
fprintf('Test samples: %d\n', n_test);

config_use = config_A;
[X_test, Y_test] = load_test_data(idx_test, cfg.data_dir, config_use, MODEL_TYPE);

if run_compare
    [X_test_B, ~] = load_test_data_ppo(idx_test, cfg.data_dir, config_B, net_B);
end

%% 4. INFERENCE
fprintf('\n--- Running inference ---\n');
t_inf_A = tic;
[Y_pred_A, Y_bin_A] = run_inference(net_A, X_test, MODEL_TYPE);
time_A = toc(t_inf_A);
fprintf('Model A (%s): inference in %.2f s (%.1f ms/sample)\n', ...
    label_A, time_A, time_A/n_test*1000);

if run_compare
    t_inf_B = tic;
    [Y_pred_B, Y_bin_B] = run_inference(net_B, X_test_B, 'ppo');
    time_B = toc(t_inf_B);
    fprintf('Model B (%s): inference in %.2f s (%.1f ms/sample)\n', ...
        label_B, time_B, time_B/n_test*1000);
end

%% 5. PER-SAMPLE METRIC COMPUTATION
fprintf('\n--- Computing per-sample metrics ---\n');

metrics_A = compute_metrics(Y_test, Y_bin_A, Y_pred_A, config_use);

if run_compare
    metrics_B = compute_metrics(Y_test, Y_bin_B, Y_pred_B, config_use);
end

print_summary(label_A, metrics_A, time_A, n_test);
if run_compare, print_summary(label_B, metrics_B, time_B, n_test); end

%% 6. STATISTICAL SIGNIFICANCE (compare mode only)
if run_compare
    fprintf('\n--- Statistical significance tests (MLP vs PPO) ---\n');

    [h_t, p_t, ~, stats_t] = ttest2(metrics_A.ssim, metrics_B.ssim);
    [h_ks, p_ks] = kstest2(metrics_A.ssim, metrics_B.ssim);

    fprintf('  Two-sample t-test (SSIM): t=%.4f, p=%.4f --> %s\n', ...
        stats_t.tstat, p_t, iif(h_t, 'SIGNIFICANT', 'not significant'));
    fprintf('  KS test (SSIM): p=%.4f --> %s\n', p_ks, iif(h_ks, 'SIGNIFICANT', 'not significant'));

    pooled_std = sqrt((std(metrics_A.ssim)^2 + std(metrics_B.ssim)^2) / 2);
    cohens_d = (mean(metrics_A.ssim) - mean(metrics_B.ssim)) / pooled_std;
    fprintf('  Cohen''s d (SSIM gap): %.4f ', cohens_d);
    if abs(cohens_d) < 0.2, fprintf('(negligible effect)\n');
    elseif abs(cohens_d) < 0.5, fprintf('(small effect)\n');
    elseif abs(cohens_d) < 0.8, fprintf('(medium effect)\n');
    else, fprintf('(large effect)\n');
    end
end

%% 7. SPECIFICATION VALIDATION TABLE
fprintf('\n--- Specification validation ---\n');

pass_fail = @(cond) iif(cond, 'PASS', 'FAIL');

spec_rows = {
    'D3/D4', 'Binary pixel accuracy >= 70%', ...
        sprintf('%.2f%%', metrics_A.acc*100), pass_fail(metrics_A.acc >= SPEC.min_binary_accuracy);
    'D3/D4', 'Mean SSIM >= 0.60', ...
        sprintf('%.4f', mean(metrics_A.ssim)), pass_fail(mean(metrics_A.ssim) >= SPEC.min_ssim);
    'D3/D4', 'Mean MSE <= 0.08', ...
        sprintf('%.6f', metrics_A.mse), pass_fail(metrics_A.mse <= SPEC.max_mse);
    'D5', 'Mean PSNR >= 10 dB', ...
        sprintf('%.2f dB', mean(metrics_A.psnr)), pass_fail(mean(metrics_A.psnr) >= SPEC.min_psnr_db);
    'D5', 'Speckle contrast K <= 0.50', ...
        sprintf('%.4f', mean(metrics_A.speckle_K)), pass_fail(mean(metrics_A.speckle_K) <= SPEC.max_speckle_K);
};

fprintf('\n  %-6s  %-35s  %-14s  %s\n', 'Del.', 'Criterion', 'Measured', 'Result');
fprintf('  %s\n', repmat('-', 1, 70));
for r = 1:size(spec_rows, 1)
    fprintf('  %-6s  %-35s  %-14s  %s\n', spec_rows{r, :});
end

spec_file = fullfile(cfg.output_dir, 'specification_validation_table.csv');
fid = fopen(spec_file, 'w');
fprintf(fid, 'Deliverable,Criterion,Measured Value,Result\n');
for r = 1:size(spec_rows, 1)
    fprintf(fid, '%s,%s,%s,%s\n', spec_rows{r, :});
end
fclose(fid);

%% 8. EXPORT CSVs
export_per_sample_csv(cfg.output_dir, label_A, idx_test, metrics_A);
if run_compare
    export_per_sample_csv(cfg.output_dir, label_B, idx_test, metrics_B);
end

save_headline_csv(cfg.output_dir, label_A, metrics_A, time_A, n_test);
if run_compare
    save_headline_csv(cfg.output_dir, label_B, metrics_B, time_B, n_test);
    save_compare_csv(cfg.output_dir, label_A, metrics_A, time_A, ...
                     label_B, metrics_B, time_B, n_test);
end

%% 9. FIGURES
fprintf('\n--- Generating figures ---\n');

% SSIM distribution
fig_ssim = figure('Position', [50 50 800 400], 'Visible', 'off');
histogram(metrics_A.ssim, 30, 'FaceColor', [0.20 0.60 0.80], 'EdgeColor', 'none');
if run_compare
    hold on;
    histogram(metrics_B.ssim, 30, 'FaceColor', [0.85 0.33 0.10], ...
              'EdgeColor', 'none', 'FaceAlpha', 0.6);
    legend(label_A, label_B);
end
xlabel('SSIM'); ylabel('Count'); grid on;
saveas(fig_ssim, fullfile(cfg.output_dir, [lower(label_A) '_ssim_distribution.png']));
close(fig_ssim);

% PSNR distribution
fig_psnr = figure('Position', [50 50 800 400], 'Visible', 'off');
histogram(metrics_A.psnr, 30, 'FaceColor', [0.47 0.67 0.19], 'EdgeColor', 'none');
if run_compare
    hold on;
    histogram(metrics_B.psnr, 30, 'FaceColor', [0.85 0.33 0.10], ...
              'EdgeColor', 'none', 'FaceAlpha', 0.6);
    legend(label_A, label_B);
end
xlabel('PSNR (dB)'); ylabel('Count'); grid on;
saveas(fig_psnr, fullfile(cfg.output_dir, [lower(label_A) '_psnr_distribution.png']));
close(fig_psnr);

% Speckle contrast distribution
fig_K = figure('Position', [50 50 800 400], 'Visible', 'off');
histogram(metrics_A.speckle_K, 30, 'FaceColor', [0.63 0.08 0.18], 'EdgeColor', 'none');
hold on;
plot([SPEC.max_speckle_K SPEC.max_speckle_K], ylim, 'r--', 'LineWidth', 1.5);
text(SPEC.max_speckle_K + 0.01, 0, 'Spec limit', 'Color', 'r', 'FontSize', 9, ...
    'VerticalAlignment', 'bottom');
xlabel('Speckle Contrast K (std/mean)'); ylabel('Count'); grid on;
saveas(fig_K, fullfile(cfg.output_dir, [lower(label_A) '_speckle_contrast_distribution.png']));
close(fig_K);

% Best/Worst by SSIM
plot_best_worst(cfg.output_dir, label_A, X_test, Y_test, Y_bin_A, ...
                metrics_A.ssim, config_use, MODEL_TYPE);

% Error heatmap
plot_error_heatmap(cfg.output_dir, label_A, Y_test, Y_pred_A, config_use);

% Compare side-by-side
if run_compare
    plot_compare(cfg.output_dir, label_A, label_B, ...
                 X_test, Y_test, Y_bin_A, Y_bin_B, ...
                 metrics_A, metrics_B, config_use);
end

% SSIM vs PSNR scatter
fig_corr = figure('Position', [50 50 700 500], 'Visible', 'off');
scatter(metrics_A.ssim, metrics_A.psnr, 20, [0.20 0.60 0.80], 'filled', ...
        'MarkerFaceAlpha', 0.5);
xlabel('SSIM'); ylabel('PSNR (dB)');
[r_val, p_val] = corr(metrics_A.ssim', metrics_A.psnr');
text(0.05, 0.92, sprintf('r = %.3f, p = %.3e', r_val, p_val), ...
     'Units', 'normalized', 'FontSize', 10);
grid on;
saveas(fig_corr, fullfile(cfg.output_dir, [lower(label_A) '_ssim_vs_psnr.png']));
close(fig_corr);

% Inference time bar (compare mode only)
if run_compare
    fig_time = figure('Position', [50 50 600 400], 'Visible', 'off');
    bar_data = [time_A/n_test, time_B/n_test] * 1000;
    bar_h = bar(bar_data, 0.5, 'FaceColor', 'flat');
    bar_h.CData = [0.20 0.60 0.80; 0.85 0.33 0.10];
    set(gca, 'XTick', [1 2], 'XTickLabel', {label_A, label_B}, 'FontSize', 12);
    ylabel('Inference time per sample (ms)'); grid on;
    saveas(fig_time, fullfile(cfg.output_dir, 'compare_inference_time.png'));
    close(fig_time);
end

%% 10. SAVE PREDICTIONS
fprintf('\nSaving prediction struct... ');
preds.Y_pred = Y_pred_A; preds.Y_binary = Y_bin_A;
preds.Y_uint8 = uint8(Y_bin_A * 255); preds.Y_true = Y_test;
preds.ssim = metrics_A.ssim; preds.psnr = metrics_A.psnr;
preds.speckle_K = metrics_A.speckle_K; preds.mse = metrics_A.mse;
preds.binary_acc = metrics_A.acc; preds.model_type = MODEL_TYPE;
preds.inference_s = time_A;
save(fullfile(cfg.output_dir, [lower(label_A) '_predictions.mat']), 'preds', '-v7.3');
fprintf('Done.\n');

fprintf('\n=== Evaluation complete. Outputs -> %s ===\n', cfg.output_dir);

%% LOCAL FUNCTIONS

function [net, config] = load_slnn(model_dir, type)
    if strcmp(type, 'forward')
        f = fullfile(model_dir, 'slnn_forward_model.mat');
        d = load(f, 'net_forward', 'config');
        net = d.net_forward;
    else
        f = fullfile(model_dir, 'slnn_inverse_controller_model.mat');
        d = load(f, 'net_inverse', 'config');
        net = d.net_inverse;
    end
    config = d.config;
end

function [net, config] = load_ppo(model_dir)
    f = fullfile(model_dir, 'ppo_forward_model.mat');
    if ~exist(f, 'file')
        f = fullfile(model_dir, 'ppo_agent.mat');
    end
    d = load(f, 'net_forward', 'config');
    net = d.net_forward;
    config = d.config;

    % Load PCA params for PPO
    pca_file = fullfile(model_dir, 'pca_params.mat');
    if exist(pca_file, 'file')
        p = load(pca_file, 'pca_params');
        config.pca_coeff = p.pca_params.coeff;
        config.pca_mean = p.pca_params.mean;
    else
        warning('pca_params.mat not found in %s', model_dir);
    end
end

function [X, Y] = load_test_data(idx_test, data_dir, config, model_type)
    n = numel(idx_test);
    if strcmp(model_type, 'inverse')
        in_size = config.target_size;
        in_dim = config.target_dim;
    else
        in_size = config.input_size;
        in_dim = config.input_dim;
    end
    X_raw = zeros(n, in_dim, 'single');
    Y = zeros(n, prod(config.output_size), 'single');
    for i = 1:n
        fi = fullfile(data_dir, sprintf('input_%05d.png', idx_test(i)));
        fo = fullfile(data_dir, sprintf('output32_%05d.png', idx_test(i)));
        tmp = loadNorm(fi, in_size);
        X_raw(i, :) = tmp(:)';
        tmp = loadNorm(fo, config.output_size);
        Y(i, :) = tmp(:)';
    end

    if strcmp(model_type, 'ppo') && isfield(config, 'pca_coeff') && isfield(config, 'pca_mean')
        X = single((double(X_raw) - config.pca_mean) * config.pca_coeff);
    else
        X = X_raw;
    end
end

function [X, Y] = load_test_data_ppo(idx_test, data_dir, config, ~)
    n = numel(idx_test);
    X_raw = zeros(n, config.input_dim, 'single');
    Y = zeros(n, prod(config.output_size), 'single');
    for i = 1:n
        fi = fullfile(data_dir, sprintf('input_%05d.png', idx_test(i)));
        fo = fullfile(data_dir, sprintf('output32_%05d.png', idx_test(i)));
        tmp = loadNorm(fi, config.input_size);
        X_raw(i, :) = tmp(:)';
        tmp = loadNorm(fo, config.output_size);
        Y(i, :) = tmp(:)';
    end
    if isfield(config, 'pca_coeff') && isfield(config, 'pca_mean')
        X = single((double(X_raw) - config.pca_mean) * config.pca_coeff);
    else
        X = X_raw;
    end
end

function [Y_pred, Y_bin] = run_inference(net, X, ~)
    if isa(net, 'dlnetwork')
        Y_pred = double(minibatchpredict(net, X, ExecutionEnvironment='cpu'));
        if max(abs(Y_pred(:))) > 1.5
            Y_pred = 1 ./ (1 + exp(-Y_pred));
        end
    else
        Y_pred = double(predict(net, X));
    end
    Y_bin = double(Y_pred >= 0.5);
end

function m = compute_metrics(Y_true, Y_bin, Y_pred, config)
    n = size(Y_true, 1);
    out_size = config.output_size;
    m.ssim = zeros(n, 1);
    m.psnr = zeros(n, 1);
    m.speckle_K = zeros(n, 1);
    for i = 1:n
        gt = double(reshape(Y_true(i, :), out_size));
        pred = double(reshape(Y_bin(i, :), out_size));
        m.ssim(i) = ssim(pred, gt);
        m.psnr(i) = psnr(pred, gt, 1.0);
        if mean(pred(:)) > 1e-6
            m.speckle_K(i) = std(pred(:)) / mean(pred(:));
        else
            m.speckle_K(i) = NaN;
        end
    end
    m.speckle_K(isnan(m.speckle_K)) = mean(m.speckle_K(~isnan(m.speckle_K)));
    m.mse = mean((double(Y_true(:)) - double(Y_pred(:))).^2);
    m.acc = mean(Y_bin(:) == double(Y_true(:)));
    m.ssim_p5 = prctile(m.ssim, 5);
    m.ssim_p50 = prctile(m.ssim, 50);
    m.ssim_p95 = prctile(m.ssim, 95);
end

function print_summary(label, m, time_s, n)
    fprintf('\n--- %s RESULTS ---\n', upper(label));
    fprintf('  MSE: %.6f\n', m.mse);
    fprintf('  Binary Accuracy: %.2f%%\n', m.acc * 100);
    fprintf('  SSIM (mean): %.4f +/- %.4f\n', mean(m.ssim), std(m.ssim));
    fprintf('  SSIM (5/50/95): %.4f / %.4f / %.4f\n', m.ssim_p5, m.ssim_p50, m.ssim_p95);
    fprintf('  PSNR (mean): %.2f dB +/- %.2f\n', mean(m.psnr), std(m.psnr));
    fprintf('  Speckle K (mean): %.4f +/- %.4f\n', mean(m.speckle_K), std(m.speckle_K));
end

function export_per_sample_csv(out_dir, label, idx_test, m)
    fname = fullfile(out_dir, [lower(label) '_per_sample_metrics.csv']);
    fid = fopen(fname, 'w');
    fprintf(fid, 'SampleIndex,SSIM,PSNR_dB,SpeckleK\n');
    for i = 1:numel(idx_test)
        fprintf(fid, '%d,%.6f,%.4f,%.6f\n', idx_test(i), m.ssim(i), m.psnr(i), m.speckle_K(i));
    end
    fclose(fid);
end

function save_headline_csv(out_dir, label, m, time_s, n)
    fname = fullfile(out_dir, [lower(label) '_metrics_summary.csv']);
    fid = fopen(fname, 'w');
    fprintf(fid, 'Metric,Value\n');
    fprintf(fid, 'MSE,%.6f\n', m.mse);
    fprintf(fid, 'BinaryAccuracy,%.4f\n', m.acc);
    fprintf(fid, 'SSIM_mean,%.4f\n', mean(m.ssim));
    fprintf(fid, 'SSIM_std,%.4f\n', std(m.ssim));
    fprintf(fid, 'PSNR_mean_dB,%.4f\n', mean(m.psnr));
    fprintf(fid, 'SpeckleK_mean,%.4f\n', mean(m.speckle_K));
    fprintf(fid, 'InferencePerSample_ms,%.4f\n', time_s / n * 1000);
    fclose(fid);
end

function save_compare_csv(out_dir, lA, mA, tA, lB, mB, tB, n)
    fname = fullfile(out_dir, 'compare_metrics_table.csv');
    fid = fopen(fname, 'w');
    fprintf(fid, 'Metric,%s,%s,Winner\n', lA, lB);
    rows = {
        'MSE', mA.mse, mB.mse, 'lower';
        'Binary Accuracy', mA.acc, mB.acc, 'higher';
        'SSIM mean', mean(mA.ssim), mean(mB.ssim), 'higher';
        'PSNR mean (dB)', mean(mA.psnr), mean(mB.psnr), 'higher';
        'Speckle K mean', mean(mA.speckle_K), mean(mB.speckle_K), 'lower';
        'Inference/sample (ms)', tA/n*1000, tB/n*1000, 'lower';
    };
    for r = 1:size(rows, 1)
        vA = rows{r, 2}; vB = rows{r, 3}; dir = rows{r, 4};
        if strcmp(dir, 'lower')
            winner = iif(vA <= vB, lA, lB);
        else
            winner = iif(vA >= vB, lA, lB);
        end
        fprintf(fid, '%s,%.4f,%.4f,%s\n', rows{r, 1}, vA, vB, winner);
    end
    fclose(fid);
end

function plot_best_worst(out_dir, label, X, Y_true, Y_bin, ssim_v, config, mtype)
    [~, sorted] = sort(ssim_v, 'descend');
    n_show = min(3, numel(sorted));
    best_idx = sorted(1:n_show);
    worst_idx = sorted(end-n_show+1:end);

    if strcmp(mtype, 'inverse')
        in_size = config.target_size;
    else
        in_size = config.input_size;
    end
    out_size = config.output_size;

    fig = figure('Position', [50 50 1400 600], 'Visible', 'off');
    for i = 1:n_show
        s = best_idx(i);
        subplot(4, n_show, i);
        imshow(reshape(X(s, :), in_size), []);
        subplot(4, n_show, n_show + i);
        imshow(reshape(Y_true(s, :), out_size), []);
        title(sprintf('True SSIM=%.3f', ssim_v(s)), 'FontSize', 7);

        s = worst_idx(i);
        subplot(4, n_show, 2*n_show + i);
        imshow(reshape(X(s, :), in_size), []);
        subplot(4, n_show, 3*n_show + i);
        imshow(reshape(Y_true(s, :), out_size), []);
        title(sprintf('True SSIM=%.3f', ssim_v(s)), 'FontSize', 7);
    end
    saveas(fig, fullfile(out_dir, [lower(label) '_best_worst_predictions.png']));
    close(fig);
end

function plot_error_heatmap(out_dir, label, Y_true, Y_pred, config)
    out_size = config.output_size;
    n = size(Y_true, 1);
    err_accum = zeros(out_size);
    for i = 1:n
        gt = double(reshape(Y_true(i, :), out_size));
        pred = double(reshape(Y_pred(i, :), out_size));
        err_accum = err_accum + abs(gt - pred);
    end
    mean_err = err_accum / n;

    fig = figure('Position', [50 50 600 500], 'Visible', 'off');
    imagesc(mean_err); colorbar; colormap('hot');
    title(sprintf('%s: Mean Absolute Error per DMD Pixel', label));
    xlabel('DMD column'); ylabel('DMD row');
    saveas(fig, fullfile(out_dir, [lower(label) '_error_heatmap.png']));
    close(fig);
end

function plot_compare(out_dir, lA, lB, X, Y_true, binA, binB, mA, mB, config)
    [~, sA] = sort(mA.ssim, 'descend');
    picks = [sA(1), sA(end)];
    out_size = config.output_size;

    fig = figure('Position', [50 50 1400 700], 'Visible', 'off');
    for col = 1:2
        s = picks(col);
        subplot(3, 4, (col-1)*2 + 1);
        imshow(reshape(X(s, :), config.input_size), []);
        subplot(3, 4, (col-1)*2 + 2);
        imshow(reshape(Y_true(s, :), out_size), []);

        subplot(3, 4, 4 + (col-1)*2 + 1);
        imshow(reshape(binA(s, :), out_size), []);
        subplot(3, 4, 4 + (col-1)*2 + 2);
        imshow(reshape(binB(s, :), out_size), []);
    end
    sgtitle(sprintf('Comparative Analysis: %s vs %s', lA, lB));
    saveas(fig, fullfile(out_dir, 'compare_side_by_side.png'));
    close(fig);
end

function img_norm = loadNorm(filepath, target_size)
    if exist(filepath, 'file')
        img = imread(filepath);
        if size(img, 3) > 1, img = rgb2gray(img); end
        img = imresize(img, target_size);
        img_norm = single(img) / 255.0;
    else
        img_norm = zeros(target_size, 'single');
    end
end

function v = iif(cond, a, b)
    if cond, v = a; else, v = b; end
end