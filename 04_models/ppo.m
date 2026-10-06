%% PPO Forward Model for Wavefront Shaping
% Speckle (64x64) -> DMD (32x32 binary) with PCA compression, BC warm-start, PPO fine-tuning

%   Needs hardware : NO (CPU only; Deep Learning + Reinforcement Learning + Statistics Toolboxes)
%   Input          : data/dataset/  (input_#####.png + output32_#####.png)
%   Output         : results/ppo/   (PCA params, agent, models, figures, CSVs)
%   Next: run 05_evaluation/model_eval3.m

clear; clc; close all;
%% PATHS (relative to the repo root, so it runs from any working directory)
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end   % fallback if run via "Run Section"
repoRoot  = fileparts(scriptDir);

%% 1. CONFIGURATION
cfg = struct();
cfg.data_dir    = fullfile(repoRoot, 'data', 'dataset');
cfg.output_dir  = fullfile(repoRoot, 'results', 'ppo');
cfg.num_samples = 9999;

cfg.input_size  = [64, 64];
cfg.input_dim   = prod(cfg.input_size);    % 4096
cfg.output_size = [32, 32];
cfg.output_dim  = prod(cfg.output_size);   % 1024
cfg.pca_dims    = 256;                     % PCA compression dimension

cfg.train_ratio = 0.8;
cfg.val_ratio   = 0.1;
cfg.test_ratio  = 0.1;

% Behaviour Cloning pre-training
cfg.bc_epochs   = 20;
cfg.bc_batch    = 128;
cfg.bc_lr       = 1e-4;

% PPO fine-tuning
cfg.ppo_episodes   = 30000;
cfg.update_freq    = 512;
cfg.mini_batch     = 128;
cfg.ppo_epochs     = 10;
cfg.clip           = 0.1;
cfg.lr_actor       = 1e-4;
cfg.lr_critic      = 3e-4;
cfg.entropy_weight = 0.02;
cfg.discount       = 1.0;
cfg.gae_lambda     = 1.0;

% Testing settings
LATENCY_REPS = 100;
THRESH_SWEEP = 0.1:0.1:0.9;

if ~exist(cfg.output_dir,'dir'), mkdir(cfg.output_dir); end

fprintf('=== PPO Forward Model Training ===\n');
metrics_log = {};

%% 2. DATA LOADING
fprintf('\n--- Loading data ---\n');
t_load = tic;

X_raw = zeros(cfg.num_samples, cfg.input_dim, 'single');
Y_dmd = zeros(cfg.num_samples, cfg.output_dim, 'single');

for i = 1:cfg.num_samples
    in_file  = fullfile(cfg.data_dir, sprintf('input_%05d.png', i));
    out_file = fullfile(cfg.data_dir, sprintf('output32_%05d.png', i));
    tmp      = loadAndNormalize(in_file, cfg.input_size);
    X_raw(i,:) = tmp(:)';
    tmp      = loadAndNormalize(out_file, cfg.output_size);
    Y_dmd(i,:) = tmp(:)';
    if mod(i,500)==0
        fprintf('  Loaded %d/%d (%.1fs)\n', i, cfg.num_samples, toc(t_load));
    end
end
load_time = toc(t_load);
fprintf('Data loaded: %.1fs\n', load_time);
metrics_log{end+1} = {'DataLoadTime_s', load_time};

% Dataset speckle contrast
X_d = double(X_raw);
K_dataset_mean = mean(std(X_d,0,2) ./ (mean(X_d,2) + 1e-8));
fprintf('Dataset speckle contrast K = %.4f\n', K_dataset_mean);
metrics_log{end+1} = {'Dataset_SpeckleK_mean', K_dataset_mean};

%% 3. TRAIN / VAL / TEST SPLIT
rng(42);
perm = randperm(cfg.num_samples);
n_train = round(cfg.num_samples * cfg.train_ratio);
n_val   = round(cfg.num_samples * cfg.val_ratio);

idx_train = perm(1:n_train);
idx_val   = perm(n_train+1:n_train+n_val);
idx_test  = perm(n_train+n_val+1:end);

X_train_raw = double(X_raw(idx_train,:));
X_val_raw   = double(X_raw(idx_val,:));
X_test_raw  = double(X_raw(idx_test,:));
Y_train     = double(Y_dmd(idx_train,:));
Y_val       = double(Y_dmd(idx_val,:));
Y_test      = double(Y_dmd(idx_test,:));

clear X_raw X_d;

metrics_log{end+1} = {'N_train', n_train};
metrics_log{end+1} = {'N_val', n_val};
metrics_log{end+1} = {'N_test', numel(idx_test)};

%% 4. PCA COMPRESSION
fprintf('\n--- Fitting PCA (%d components) ---\n', cfg.pca_dims);
t_pca = tic;

X_train_mean = mean(X_train_raw, 1);
X_train_c    = X_train_raw - X_train_mean;

[coeff, score_train, ~, ~, explained] = pca(X_train_c, ...
    'NumComponents', cfg.pca_dims, 'Economy', true);

pca_time = toc(t_pca);
var_explained = sum(explained(1:cfg.pca_dims));
fprintf('PCA fit: %.1fs  Variance explained: %.2f%%\n', pca_time, var_explained);
metrics_log{end+1} = {'PCA_VarianceExplained_pct', var_explained};
metrics_log{end+1} = {'PCA_FitTime_s', pca_time};

X_train_pca = score_train;
X_val_pca   = (X_val_raw - X_train_mean) * coeff;
X_test_pca  = (X_test_raw - X_train_mean) * coeff;

clear X_train_raw X_val_raw X_test_raw X_train_c;

% Save PCA params
pca_params.coeff     = coeff;
pca_params.mean      = X_train_mean;
pca_params.pca_dims  = cfg.pca_dims;
pca_params.explained = explained;
save(fullfile(cfg.output_dir, 'pca_params.mat'), 'pca_params', '-v7.3');

cfg.obs_dim = cfg.pca_dims;

%% 5. BEHAVIOUR CLONING WARM-START
fprintf('\n--- Behaviour cloning warm-start (%d epochs) ---\n', cfg.bc_epochs);

bc_layers = [
    featureInputLayer(cfg.obs_dim, Normalization="none", Name="bc_in")
    fullyConnectedLayer(512, Name="bc_fc1")
    layerNormalizationLayer(Name="bc_ln1")
    reluLayer(Name="bc_relu1")
    dropoutLayer(0.3, Name="bc_drop1")
    fullyConnectedLayer(256, Name="bc_fc2")
    layerNormalizationLayer(Name="bc_ln2")
    reluLayer(Name="bc_relu2")
    dropoutLayer(0.2, Name="bc_drop2")
    fullyConnectedLayer(cfg.output_dim, Name="bc_fc_out")
    sigmoidLayer(Name="bc_sigmoid")
];

net_bc = dlnetwork(bc_layers);
bc_options = trainingOptions("adam", ...
    InitialLearnRate=cfg.bc_lr, MaxEpochs=cfg.bc_epochs, ...
    MiniBatchSize=cfg.bc_batch, Shuffle="every-epoch", ...
    ValidationData={X_val_pca, Y_val}, ValidationFrequency=30, ...
    Plots="training-progress", Verbose=true, ExecutionEnvironment="cpu");

t_bc = tic;
[net_bc, info_bc] = trainnet(X_train_pca, Y_train, net_bc, "mse", bc_options);
bc_time = toc(t_bc);
fprintf('BC pre-training done: %.1fs\n', bc_time);
metrics_log{end+1} = {'BC_TrainingTime_s', bc_time};

% BC validation metrics
Y_bc_pred = double(minibatchpredict(net_bc, X_val_pca, ExecutionEnvironment="cpu"));
Y_bc_bin  = double(Y_bc_pred >= 0.5);
bc_mse    = mean((Y_val(:) - Y_bc_pred(:)).^2);
bc_acc    = mean(Y_bc_bin(:) == Y_val(:));
metrics_log{end+1} = {'BC_Val_MSE', bc_mse};
metrics_log{end+1} = {'BC_Val_BinaryAcc', bc_acc};

%% 6. OBSERVATION AND ACTION SPECS
obs_info = rlNumericSpec([cfg.obs_dim, 1], ...
    LowerLimit=-inf(cfg.obs_dim,1), UpperLimit=inf(cfg.obs_dim,1));
obs_info.Name = 'speckle_pca';

act_info = rlNumericSpec([cfg.output_dim, 1], ...
    LowerLimit=-ones(cfg.output_dim,1), UpperLimit=ones(cfg.output_dim,1));
act_info.Name = 'dmd_logits';

%% 7. ENVIRONMENT
ppoEnvInit(X_train_pca, Y_train);
env = rlFunctionEnv(obs_info, act_info, @ppoEnvStep, @ppoEnvReset);
fprintf('Environment created (%d dims).\n', cfg.obs_dim);

%% 8. ACTOR NETWORK (initialised from BC weights)
actor_net = dlnetwork();

trunk_layers = [
    featureInputLayer(cfg.obs_dim, Normalization="none", Name="obs_in")
    fullyConnectedLayer(512, Name="a_fc1")
    layerNormalizationLayer(Name="a_ln1")
    reluLayer(Name="a_relu1")
    dropoutLayer(0.3, Name="a_drop1")
    fullyConnectedLayer(256, Name="a_fc2")
    layerNormalizationLayer(Name="a_ln2")
    reluLayer(Name="a_relu2")
    dropoutLayer(0.2, Name="a_drop2")
];
mean_layers = [
    fullyConnectedLayer(cfg.output_dim, Name="a_mean_fc")
    tanhLayer(Name="a_mean_out")
];
std_layers = [
    fullyConnectedLayer(cfg.output_dim, Name="a_std_fc")
    softplusLayer(Name="a_std_out")
];

actor_net = addLayers(actor_net, trunk_layers);
actor_net = addLayers(actor_net, mean_layers);
actor_net = addLayers(actor_net, std_layers);
actor_net = connectLayers(actor_net, "a_drop2", "a_mean_fc");
actor_net = connectLayers(actor_net, "a_drop2", "a_std_fc");
actor_net = initialize(actor_net);

% Transfer BC weights
bc_fc1_idx = findLayerIndex(net_bc, "bc_fc1");
bc_fc2_idx = findLayerIndex(net_bc, "bc_fc2");
bc_out_idx = findLayerIndex(net_bc, "bc_fc_out");

actor_net = replaceLayer(actor_net, "a_fc1", ...
    fullyConnectedLayer(512, Name="a_fc1", ...
        Weights=net_bc.Layers(bc_fc1_idx).Weights, ...
        Bias=net_bc.Layers(bc_fc1_idx).Bias));
actor_net = replaceLayer(actor_net, "a_fc2", ...
    fullyConnectedLayer(256, Name="a_fc2", ...
        Weights=net_bc.Layers(bc_fc2_idx).Weights, ...
        Bias=net_bc.Layers(bc_fc2_idx).Bias));
actor_net = replaceLayer(actor_net, "a_mean_fc", ...
    fullyConnectedLayer(cfg.output_dim, Name="a_mean_fc", ...
        Weights=net_bc.Layers(bc_out_idx).Weights, ...
        Bias=net_bc.Layers(bc_out_idx).Bias));
actor_net = initialize(actor_net);

actor_opt = rlOptimizerOptions(LearnRate=cfg.lr_actor, GradientThreshold=1.0);
actor = rlContinuousGaussianActor(actor_net, obs_info, act_info, ...
    ActionMeanOutputNames="a_mean_out", ...
    ActionStandardDeviationOutputNames="a_std_out", ...
    ObservationInputNames="obs_in");

%% 9. CRITIC NETWORK
critic_layers = [
    featureInputLayer(cfg.obs_dim, Normalization="none", Name="c_obs_in")
    fullyConnectedLayer(512, Name="c_fc1")
    reluLayer(Name="c_relu1")
    fullyConnectedLayer(128, Name="c_fc2")
    reluLayer(Name="c_relu2")
    fullyConnectedLayer(1, Name="c_out")
];
critic_net = dlnetwork(critic_layers);
critic_opt = rlOptimizerOptions(LearnRate=cfg.lr_critic, GradientThreshold=1.0);
critic = rlValueFunction(critic_net, obs_info, ObservationInputNames="c_obs_in");

%% 10. PPO AGENT
ppo_opts = rlPPOAgentOptions( ...
    ExperienceHorizon=cfg.update_freq, ClipFactor=cfg.clip, ...
    EntropyLossWeight=cfg.entropy_weight, MiniBatchSize=cfg.mini_batch, ...
    NumEpoch=cfg.ppo_epochs, AdvantageEstimateMethod="gae", ...
    GAEFactor=cfg.gae_lambda, DiscountFactor=cfg.discount, ...
    SampleTime=1, ActorOptimizerOptions=actor_opt, CriticOptimizerOptions=critic_opt);

agent = rlPPOAgent(actor, critic, ppo_opts);

train_opts = rlTrainingOptions( ...
    MaxEpisodes=cfg.ppo_episodes, MaxStepsPerEpisode=1, ...
    ScoreAveragingWindowLength=500, ...
    StopTrainingCriteria="EpisodeCount", StopTrainingValue=cfg.ppo_episodes, ...
    Verbose=true, Plots="training-progress");

%% 11. PPO FINE-TUNING
fprintf('\n=== PPO fine-tuning (%d episodes) ===\n', cfg.ppo_episodes);

t_ppo = tic;
train_result = train(agent, env, train_opts);
time_ppo = toc(t_ppo);
fprintf('PPO fine-tuning complete: %.1fs\n', time_ppo);
metrics_log{end+1} = {'PPO_TrainingTime_s', time_ppo};

save(fullfile(cfg.output_dir, 'ppo_agent.mat'), 'agent', 'cfg', '-v7.3');

%% 12. TEST EVALUATION (deterministic mean head)
fprintf('\n--- Test evaluation (deterministic mean) ---\n');

actor_trained = getActor(agent);
actor_model   = getModel(actor_trained);

Y_pred_mean = double(minibatchpredict(actor_model, X_test_pca, ...
    Outputs="a_mean_out", ExecutionEnvironment="cpu"));

Y_pred_prob = 1 ./ (1 + exp(-Y_pred_mean));   % sigmoid to [0,1]
Y_pred_bin  = double(Y_pred_prob >= 0.5);

% Core metrics
mse_val  = mean((Y_test(:) - Y_pred_prob(:)).^2);
rmse_val = sqrt(mse_val);
acc_val  = mean(Y_pred_bin(:) == Y_test(:));

% Per-sample metrics
[ssim_vals, psnr_vals, speckleK_vals, jaccard_vals] = ...
    per_sample_metrics(Y_test, Y_pred_bin, cfg.output_size);

% Confusion matrix
[cm_ppo, prec_ppo, rec_ppo, f1_ppo] = pixel_confusion(Y_test, Y_pred_bin);

% Pixel MAE heatmap
hmap_ppo = pixel_mae_heatmap(Y_test, Y_pred_prob, cfg.output_size);

% Threshold sweep
thresh_acc = threshold_sweep(Y_test, Y_pred_prob, THRESH_SWEEP);

% Latency benchmark
[lat_mean_ms, lat_std_ms] = latency_benchmark_ppo(actor_model, X_test_pca(1,:), LATENCY_REPS);

fprintf('\n--- PPO FORWARD TEST RESULTS ---\n');
fprintf('  MSE: %.6f, RMSE: %.6f, Binary Acc: %.2f%%\n', mse_val, rmse_val, acc_val*100);
fprintf('  F1: %.4f, Jaccard: %.4f, SSIM: %.4f, PSNR: %.2f dB\n', ...
    f1_ppo, mean(jaccard_vals), mean(ssim_vals), mean(psnr_vals));
fprintf('  Latency: %.3f +/- %.3f ms\n', lat_mean_ms, lat_std_ms);

% Log metrics
metrics_log{end+1} = {'PPO_MSE', mse_val};
metrics_log{end+1} = {'PPO_RMSE', rmse_val};
metrics_log{end+1} = {'PPO_BinaryAcc', acc_val};
metrics_log{end+1} = {'PPO_F1', f1_ppo};
metrics_log{end+1} = {'PPO_Jaccard_mean', mean(jaccard_vals)};
metrics_log{end+1} = {'PPO_SSIM_mean', mean(ssim_vals)};
metrics_log{end+1} = {'PPO_PSNR_mean_dB', mean(psnr_vals)};
metrics_log{end+1} = {'PPO_Latency_mean_ms', lat_mean_ms};

%% 13. SAVE MODEL (model_eval2 compatible)
fprintf('\n--- Saving model ---\n');

net_forward = build_inference_net(cfg, actor_model);

config = struct();
config.input_size  = cfg.input_size;
config.input_dim   = cfg.input_dim;
config.output_size = cfg.output_size;
config.output_dim  = cfg.output_dim;
config.pca_dims    = cfg.pca_dims;
config.pca_params  = pca_params;

save(fullfile(cfg.output_dir, 'ppo_forward_model.mat'), 'net_forward', 'config', '-v7.3');

%% 14. WRITE METRICS SUMMARY CSV
csv_path = fullfile(cfg.output_dir, 'ppo_metrics_summary.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, 'Metric,Value\n');
for r = 1:numel(metrics_log)
    e = metrics_log{r};
    if isnumeric(e{2})
        fprintf(fid, '%s,%.8f\n', e{1}, e{2});
    else
        fprintf(fid, '%s,%s\n', e{1}, e{2});
    end
end
fclose(fid);

%% 15. EXPORT CURVES
export_reward_curve(train_result.EpisodeReward, cfg.output_dir);
export_training_curve(info_bc, cfg.output_dir, 'bc');

%% 16. FIGURES
fprintf('--- Generating figures ---\n');

% PCA variance curve
fig_pca = figure('Position',[50 50 800 380],'Visible','off');
cum_exp = cumsum(explained(1:min(cfg.pca_dims,numel(explained))));
plot(cum_exp, 'b-', 'LineWidth',1.5); grid on;
xlabel('Components'); ylabel('Cumulative variance (%)');
title('PCA Variance Explained');
saveas(fig_pca, fullfile(cfg.output_dir, 'ppo_pca_variance.png')); close(fig_pca);

% SSIM distribution
fig_ssim = figure('Position',[50 50 800 380],'Visible','off');
histogram(ssim_vals, 30, 'FaceColor',[0.47 0.67 0.19]);
xlabel('SSIM'); ylabel('Count'); title('PPO SSIM Distribution'); grid on;
saveas(fig_ssim, fullfile(cfg.output_dir, 'ppo_ssim_dist.png')); close(fig_ssim);

% Pixel MAE heatmap
fig_heat = figure('Position',[50 50 500 430],'Visible','off');
imagesc(hmap_ppo); colorbar; colormap('hot');
title('PPO Mean Absolute Error per DMD pixel'); axis square;
saveas(fig_heat, fullfile(cfg.output_dir, 'ppo_pixel_mae.png')); close(fig_heat);

% Confusion matrix
plot_confusion(cm_ppo, cfg.output_dir, 'ppo');

fprintf('\n=== Done. Outputs in: %s ===\n', cfg.output_dir);

%% LOCAL FUNCTIONS

function [ssim_v, psnr_v, k_v, jac_v] = per_sample_metrics(Y_true, Y_bin, out_sz)
    n = size(Y_true,1);
    ssim_v = zeros(n,1); psnr_v = zeros(n,1);
    k_v = zeros(n,1); jac_v = zeros(n,1);
    for i = 1:n
        gt = reshape(Y_true(i,:), out_sz);
        pred = reshape(Y_bin(i,:), out_sz);
        ssim_v(i) = ssim(pred, gt);
        psnr_v(i) = psnr(pred, gt, 1.0);
        if mean(pred(:)) > 1e-6
            k_v(i) = std(pred(:)) / mean(pred(:));
        else
            k_v(i) = NaN;
        end
        inter = sum(pred(:) & gt(:));
        uni = sum(pred(:) | gt(:));
        jac_v(i) = inter/uni;
    end
    k_v(isnan(k_v)) = mean(k_v(~isnan(k_v)));
end

function [cm, precision, recall, f1] = pixel_confusion(Y_true, Y_bin)
    gt = logical(Y_true(:));
    pred = logical(Y_bin(:));
    TP = sum(pred & gt); FP = sum(pred & ~gt);
    FN = sum(~pred & gt); TN = sum(~pred & ~gt);
    cm = [TP, FP; FN, TN];
    precision = TP / max(TP+FP, 1);
    recall = TP / max(TP+FN, 1);
    f1 = 2*precision*recall / max(precision+recall, 1e-8);
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

function [mu_ms, sd_ms] = latency_benchmark_ppo(actor_model, sample_pca, reps)
    times = zeros(reps,1);
    for r = 1:reps
        t0 = tic;
        minibatchpredict(actor_model, sample_pca, Outputs="a_mean_out", ExecutionEnvironment="cpu");
        times(r) = toc(t0)*1000;
    end
    times = times(6:end);
    mu_ms = mean(times);
    sd_ms = std(times);
end

function export_training_curve(info, out_dir, tag)
    if ~isfield(info,'TrainingLoss'), return; end
    tl = info.TrainingLoss;
    fid = fopen(fullfile(out_dir, sprintf('ppo_%s_loss.csv',tag)),'w');
    fprintf(fid,'Iteration,Loss\n');
    for k = 1:numel(tl)
        fprintf(fid,'%d,%.8f\n', k, tl(k));
    end
    fclose(fid);
end

function export_reward_curve(rewards, out_dir)
    fid = fopen(fullfile(out_dir,'ppo_reward.csv'),'w');
    fprintf(fid,'Episode,Reward\n');
    for k = 1:numel(rewards)
        fprintf(fid,'%d,%.8f\n', k, rewards(k));
    end
    fclose(fid);
end

function plot_confusion(cm, out_dir, tag)
    fig = figure('Position',[50 50 500 420],'Visible','off');
    imagesc(cm); colormap('parula'); colorbar;
    set(gca,'XTick',[1 2],'XTickLabel',{'On','Off'},'YTick',[1 2],'YTickLabel',{'On','Off'});
    xlabel('Predicted'); ylabel('True');
    title(sprintf('Confusion: %s', upper(tag)));
    for r = 1:2, for c = 1:2, text(c, r, num2str(cm(r,c)), 'HorizontalAlignment','center', 'FontSize',14,'Color','w'); end, end
    saveas(fig, fullfile(out_dir, sprintf('ppo_%s_confusion.png', tag))); close(fig);
end

function net_f = build_inference_net(cfg, actor_model)
    infer_layers = [
        featureInputLayer(cfg.obs_dim, Normalization="none", Name="in")
        fullyConnectedLayer(512, Name="fc1")
        layerNormalizationLayer(Name="ln1")
        reluLayer(Name="relu1")
        dropoutLayer(0.3, Name="drop1")
        fullyConnectedLayer(256, Name="fc2")
        layerNormalizationLayer(Name="ln2")
        reluLayer(Name="relu2")
        dropoutLayer(0.2, Name="drop2")
        fullyConnectedLayer(cfg.output_dim, Name="fc_out")
        sigmoidLayer(Name="sigmoid")
    ];
    net_f = dlnetwork(infer_layers);
    net_f = initialize(net_f);
    a1 = findLayerIndex(actor_model, "a_fc1");
    a2 = findLayerIndex(actor_model, "a_fc2");
    am = findLayerIndex(actor_model, "a_mean_fc");
    net_f = replaceLayer(net_f, "fc1", fullyConnectedLayer(512, Name="fc1", ...
        Weights=actor_model.Layers(a1).Weights, Bias=actor_model.Layers(a1).Bias));
    net_f = replaceLayer(net_f, "fc2", fullyConnectedLayer(256, Name="fc2", ...
        Weights=actor_model.Layers(a2).Weights, Bias=actor_model.Layers(a2).Bias));
    net_f = replaceLayer(net_f, "fc_out", fullyConnectedLayer(cfg.output_dim, Name="fc_out", ...
        Weights=actor_model.Layers(am).Weights, Bias=actor_model.Layers(am).Bias));
    net_f = initialize(net_f);
end

function idx = findLayerIndex(net, layerName)
    names = {net.Layers.Name};
    idx = find(strcmp(names, layerName), 1);
    if isempty(idx), error('Layer "%s" not found.', layerName); end
end

function img_norm = loadAndNormalize(filepath, target_size)
    if exist(filepath,'file')
        img = imread(filepath);
        if size(img,3) > 1, img = rgb2gray(img); end
        img = imresize(img, target_size);
        img_norm = single(img) / 255.0;
    else
        img_norm = zeros(target_size,'single');
    end
end

%% Environment functions
function ppoEnvInit(X_pca, Y)
    ppoEnvStore('set', X_pca, Y);
end

function varargout = ppoEnvStore(mode, X, Y)
    persistent pX pY pN;
    if strcmp(mode,'set')
        pX = X; pY = Y; pN = size(X,1); return;
    end
    varargout{1} = pX; varargout{2} = pY; varargout{3} = pN;
end

function [obs, loggedSignals] = ppoEnvReset()
    [X, Y, n] = ppoEnvStore('get');
    idx = randi(n);
    obs = X(idx,:)';
    loggedSignals.gt_dmd = Y(idx,:)';
end

function [obs, reward, isDone, loggedSignals] = ppoEnvStep(action, loggedSignals)
    prob = 1./(1+exp(-double(action)));
    bin_mask = double(prob >= 0.5);
    gt = loggedSignals.gt_dmd;
    reward = -20 * mean((bin_mask - gt).^2);
    isDone = true;
    [X, Y, n] = ppoEnvStore('get');
    idx = randi(n);
    obs = X(idx,:)';
    loggedSignals.gt_dmd = Y(idx,:)';
end