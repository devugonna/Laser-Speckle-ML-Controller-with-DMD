% RPATT  Stage 02 - pattern generation + synchronised DMD/camera capture
%   Needs hardware : YES (DMD on a second display + ZWO camera)
%   Output (all under data/raw/):
%       finalinp/input_#####.png             raw camera frames (800x640 speckle)
%       finaloutp/output_#####.png           full DMD patterns
%       finaloutp/32x32/output32_#####.png   32x32 ground-truth patterns
%       final_test_log_10k.mat               capture log
%   Next: run 03_preprocessing/crop.m
%% RVITM Focus Timing Test — 10,000 patterns with camera acquisition
%% Block layout (indices):
%    1    – 1499 : 4x4 random binary checkerboard (pre)
%    1500        : all-on (solid white)
%    1501        : all-off (solid black)
%    1502 – 3500 : 4x4 random binary checkerboard (post)
%    3501 – 4524 : 32x32 Hadamard patterns (1024)
%    4525 – 5524 : 32x32 sparse RBP (1000)
%    5525 – 6524 : 32x32 random circular focal spots (1000)
%    6525 – 7524 : 32x32 stripe/grating patterns (1000)
%    7525 – 8524 : 32x32 quadrant/block patterns (1000)
%    8525 – 9524 : 32x32 ring/annular patterns (1000)
%    9525 – 10000: 4x4 validation RBP (476)

%% 0. USER PARAMETERS
CAMERA_INDEX   = 2;
CAMERA_FORMAT  = 'RGB8_800x640';
CAMERA_FPS     = 30;
SRC_EXPOSURE   = -6.8;
SRC_GAIN       = 10.3;

DISPLAY_GRID   = 4;      % 4x4 for coarse binary / validation
DISPLAY_GRID32 = 32;     % 32x32 for Hadamard / sparse / others

PATTERN_PX     = 528;    % full-resolution pattern side length
MACRO_PIXEL    = PATTERN_PX / DISPLAY_GRID;    % for 4x4 (132 px)
SCALE_32       = 8;      % 4x4 → 32x32 thumbnails

SETTLE_S       = 0.1;    % Settling time after pattern display

% Spot parameters
SPOT_RADIUS_MIN = 70;
SPOT_RADIUS_MAX = 90;
SPOT_MARGIN     = 90;

% Stripe parameters
STRIPE_PERIOD_MIN = 20;
STRIPE_PERIOD_MAX = 80;
STRIPE_ANGLES     = [0 45 90 135];

% Ring parameters
RING_R_MIN_IN   = 40;
RING_R_MAX_IN   = 120;
RING_THICK_MIN  = 10;
RING_THICK_MAX  = 40;

% Pattern counts and indices
NUM_PATTERNS   = 10000;

ALLON_IDX      = 1500;
ALLOFF_IDX     = 1501;
HAD_START      = 3501;
HAD_END        = 4524;   % 1024 Hadamard rows
SPARSE_START   = 4525;
SPARSE_END     = 5524;   % 1000 sparse RBP
SPOT_START     = 5525;
SPOT_END       = 6524;   % 1000 spot
STRIPE_START   = 6525;
STRIPE_END     = 7524;   % 1000 stripe
QUAD_START     = 7525;
QUAD_END       = 8524;   % 1000 quadrant
RING_START     = 8525;
RING_END       = 9524;   % 1000 ring
VAL_START      = 9525;
VAL_END        = 10000;  % 476 validation RBP

% Derived counts
NUM_BINARY_PRE  = ALLON_IDX - 1;                 % 1499
NUM_BINARY_POST = 3500 - ALLOFF_IDX;             % 1999
NUM_HAD         = HAD_END   - HAD_START   + 1;   % 1024
NUM_SPARSE      = SPARSE_END - SPARSE_START + 1; % 1000
NUM_SPOT        = SPOT_END   - SPOT_START   + 1; % 1000
NUM_STRIPE      = STRIPE_END - STRIPE_START + 1; % 1000
NUM_QUAD        = QUAD_END   - QUAD_START   + 1; % 1000
NUM_RING        = RING_END   - RING_START   + 1; % 1000
NUM_VAL         = VAL_END    - VAL_START    + 1; % 476

% Directories
%% PATHS (relative to the repo root, so it runs from any working directory)
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir), scriptDir = pwd; end   % fallback if run via "Run Section"
repoRoot  = fileparts(scriptDir);
OUTPUT_DIR  = fullfile(repoRoot, 'data', 'raw');
if ~exist(OUTPUT_DIR, 'dir'), mkdir(OUTPUT_DIR); end
INPUT_DIR   = fullfile(OUTPUT_DIR, 'finalinp');
OUT_DIR     = fullfile(OUTPUT_DIR, 'finaloutp');
OUT_DIR_32  = fullfile(OUT_DIR,   '32x32');
LOG_FILE    = fullfile(OUTPUT_DIR, 'final_test_log_10k.mat');

if ~exist(INPUT_DIR,  'dir'),  mkdir(INPUT_DIR);   end
if ~exist(OUT_DIR,    'dir'),  mkdir(OUT_DIR);     end
if ~exist(OUT_DIR_32, 'dir'),  mkdir(OUT_DIR_32);  end

fprintf('=== RVITM Focus Test — %d unique patterns ===\n', NUM_PATTERNS);

%% 1. BUILD ALL PATTERNS UP FRONT (10k total)
numCells4  = DISPLAY_GRID^2;      % 16
numCells32 = DISPLAY_GRID32^2;    % 1024

rng(42);   % Fixed seed for reproducibility

% 4x4 Binary RBP (pre + post)
binaryCores_pre  = cell(NUM_BINARY_PRE, 1);
for k = 1:NUM_BINARY_PRE
    binaryCores_pre{k} = uint8(randi([0 1], DISPLAY_GRID, DISPLAY_GRID));
end

binaryCores_post = cell(NUM_BINARY_POST, 1);
for k = 1:NUM_BINARY_POST
    binaryCores_post{k} = uint8(randi([0 1], DISPLAY_GRID, DISPLAY_GRID));
end

% 32x32 Hadamard basis (1024 rows)
H = hadamard(NUM_HAD);      % 1024x1024 ±1
hadCores = cell(NUM_HAD, 1);
for k = 1:NUM_HAD
    row = reshape(H(k,:), DISPLAY_GRID32, DISPLAY_GRID32);
    hadCores{k} = uint8((row + 1) / 2);   % {-1,+1} → {0,1}
end

% 32x32 Sparse RBP (max 1/3 fill)
sparseCores = cell(NUM_SPARSE, 1);
MAX_FILL_FRAC = 1/3;
MAX_ON_CELLS  = floor(numCells32 * MAX_FILL_FRAC);   % <= 341
for k = 1:NUM_SPARSE
    invert = randi([0 1]);
    n_on   = randi([1 MAX_ON_CELLS]);
    flat   = zeros(1, numCells32, 'uint8');
    idx    = randperm(numCells32, n_on);
    flat(idx) = 1;
    if invert, flat = 1 - flat; end
    sparseCores{k} = reshape(flat, DISPLAY_GRID32, DISPLAY_GRID32);
end

% Spot configs (1000)
[XX, YY] = meshgrid(1:PATTERN_PX, 1:PATTERN_PX);
numSpots_arr = randi([2 3], NUM_SPOT, 1);
spot_configs = cell(NUM_SPOT, 1);
for g = 1:NUM_SPOT
    ns     = numSpots_arr(g);
    cfg.n  = ns;
    cfg.cx = SPOT_MARGIN + rand(ns,1) * (PATTERN_PX - 2*SPOT_MARGIN);
    cfg.cy = SPOT_MARGIN + rand(ns,1) * (PATTERN_PX - 2*SPOT_MARGIN);
    cfg.r  = SPOT_RADIUS_MIN + rand(ns,1)*(SPOT_RADIUS_MAX - SPOT_RADIUS_MIN);
    spot_configs{g} = cfg;
end

% Stripe/grating configs (1000)
stripe_configs = cell(NUM_STRIPE, 1);
num_stripe_angles = length(STRIPE_ANGLES);
for g = 1:NUM_STRIPE
    sc.period = STRIPE_PERIOD_MIN + rand()*(STRIPE_PERIOD_MAX - STRIPE_PERIOD_MIN);
    sc.angle  = STRIPE_ANGLES(randi(num_stripe_angles));
    sc.phase  = rand() * sc.period;
    stripe_configs{g} = sc;
end

% Quadrant/block configs (1000)
quad_configs = cell(NUM_QUAD, 1);
quad_divisions = [2, 4, 8];
for g = 1:NUM_QUAD
    qc.divs = quad_divisions(randi(3));
    qc.pattern = uint8(randi([0 1], qc.divs, qc.divs));
    quad_configs{g} = qc;
end

% Ring/annular configs (1000)
ring_configs = cell(NUM_RING, 1);
for g = 1:NUM_RING
    rc.cx = SPOT_MARGIN + rand()*(PATTERN_PX - 2*SPOT_MARGIN);
    rc.cy = SPOT_MARGIN + rand()*(PATTERN_PX - 2*SPOT_MARGIN);
    rc.r_in  = RING_R_MIN_IN + rand()*(RING_R_MAX_IN - RING_R_MIN_IN);
    rc.thick = RING_THICK_MIN + rand()*(RING_THICK_MAX - RING_THICK_MIN);
    rc.r_out = rc.r_in + rc.thick;
    ring_configs{g} = rc;
end

% Validation 4x4 RBP (476 patterns, separate seed)
rng(137);
valCores = cell(NUM_VAL, 1);
for k = 1:NUM_VAL
    valCores{k} = uint8(randi([0 1], DISPLAY_GRID, DISPLAY_GRID));
end

fprintf('All pattern configs pre-built.\n');

%% 2. HELPER FUNCTIONS
function canvas = buildCanvas(patImg, padTop, padLeft, canvasH, canvasW, PATTERN_PX)
    canvas = zeros(canvasH, canvasW);
    canvas(padTop+1:padTop+PATTERN_PX, padLeft+1:padLeft+PATTERN_PX) = patImg;
end

function img = renderCircles(cfg, XX, YY, PATTERN_PX)
    mask = false(PATTERN_PX, PATTERN_PX);
    for s = 1:cfg.n
        mask = mask | ((XX-cfg.cx(s)).^2 + (YY-cfg.cy(s)).^2 <= cfg.r(s)^2);
    end
    img = uint8(mask) * 255;
end

function img = renderStripe(sc, PATTERN_PX)
    [Xs, Ys] = meshgrid(1:PATTERN_PX, 1:PATTERN_PX);
    theta = deg2rad(sc.angle);
    coord =  Xs*cos(theta) + Ys*sin(theta);
    val   = sin(2*pi*(coord + sc.phase)/sc.period);
    mask  = val > 0;
    img   = uint8(mask) * 255;
end

function img = renderQuad(qc, DISPLAY_GRID32, PATTERN_PX)
    grid32 = zeros(DISPLAY_GRID32, DISPLAY_GRID32, 'uint8');
    blockSize = DISPLAY_GRID32 / qc.divs;
    for by = 1:qc.divs
        for bx = 1:qc.divs
            v = qc.pattern(by, bx);
            yIdx = ((by-1)*blockSize + 1) : (by*blockSize);
            xIdx = ((bx-1)*blockSize + 1) : (bx*blockSize);
            grid32(yIdx, xIdx) = v;
        end
    end
    img = imresize(grid32*255, [PATTERN_PX PATTERN_PX], 'nearest');
end

function img = renderRing(rc, XX, YY)
    r2 = (XX-rc.cx).^2 + (YY-rc.cy).^2;
    mask = (r2 >= rc.r_in^2) & (r2 <= rc.r_out^2);
    img = uint8(mask) * 255;
end

%% 3. DISPLAY WINDOW
monitors  = get(0, 'MonitorPositions');
targetIdx = min(size(monitors, 1), 2);
targetPos = monitors(targetIdx, :);

screenW = targetPos(3);
screenH = targetPos(4);

padLeft = floor((screenW - PATTERN_PX) / 2);
padTop  = floor((screenH - PATTERN_PX) / 2);

fig = figure('Color', 'k', 'MenuBar', 'none', 'ToolBar', 'none', ...
             'NumberTitle', 'off', 'Name', 'RVITM Focus Test — 10000 patterns');
set(fig, 'Position', targetPos);
fig.WindowState = 'fullscreen';

ax = axes('Parent', fig, 'Position', [0, 0, 1, 1]);
axis(ax, 'off');

canvas = zeros(screenH, screenW);
hImg   = image(ax, repmat(canvas, 1, 1, 3));
axis(ax, 'off'); axis(ax, 'image');

fprintf('Display window ready.\n');

%% 4. CAMERA INIT
fprintf('Initialising camera...\n');
imaqreset;
vid = videoinput('winvideo', CAMERA_INDEX, CAMERA_FORMAT);
src = getselectedsource(vid);
src.Exposure         = SRC_EXPOSURE;
src.Gain             = SRC_GAIN;
vid.FramesPerTrigger = 1;
vid.TriggerRepeat    = NUM_PATTERNS - 1;
triggerconfig(vid, 'manual');

start(vid);
fprintf('Camera armed for %d manual triggers.\n', NUM_PATTERNS);

%% 5. ACQUISITION LOOP
patternTimestamps = zeros(NUM_PATTERNS, 1);
camFrames         = cell(NUM_PATTERNS, 1);
tStart            = tic;

for i = 1:NUM_PATTERNS

    if ~ishandle(fig)
        fprintf('Figure closed — aborting at pattern %d.\n', i);
        break;
    end

    core   = [];
    label  = '';

    % Build patImg based on index i
    if i < ALLON_IDX
        % 4x4 Binary block 1 (1–1499)
        core   = binaryCores_pre{i};
        patImg = kron(core, ones(MACRO_PIXEL, 'uint8')) * 255;
        label  = 'BIN-PRE';

    elseif i == ALLON_IDX
        % All-on: solid white
        patImg = ones(PATTERN_PX, PATTERN_PX, 'uint8') * 255;
        core   = ones(DISPLAY_GRID, DISPLAY_GRID, 'uint8');
        label  = 'ALL-ON';

    elseif i == ALLOFF_IDX
        % All-off: solid black
        patImg = zeros(PATTERN_PX, PATTERN_PX, 'uint8');
        core   = zeros(DISPLAY_GRID, DISPLAY_GRID, 'uint8');
        label  = 'ALL-OFF';

    elseif i <= 3500
        % 4x4 Binary block 2 (1502–3500)
        k      = i - ALLOFF_IDX;
        core   = binaryCores_post{k};
        patImg = kron(core, ones(MACRO_PIXEL, 'uint8')) * 255;
        label  = 'BIN-POST';

    elseif i >= HAD_START && i <= HAD_END
        % 32x32 Hadamard
        k      = i - HAD_START + 1;
        core32 = hadCores{k};
        patImg = imresize(core32*255, [PATTERN_PX PATTERN_PX], 'nearest');
        label  = 'HAD';

    elseif i >= SPARSE_START && i <= SPARSE_END
        % 32x32 sparse RBP
        k      = i - SPARSE_START + 1;
        core32 = sparseCores{k};
        patImg = imresize(core32*255, [PATTERN_PX PATTERN_PX], 'nearest');
        label  = 'SPARSE';

    elseif i >= SPOT_START && i <= SPOT_END
        % Spot block
        g      = i - SPOT_START + 1;
        patImg = renderCircles(spot_configs{g}, XX, YY, PATTERN_PX);
        label  = 'SPOT';

    elseif i >= STRIPE_START && i <= STRIPE_END
        % Stripe/grating block
        g      = i - STRIPE_START + 1;
        patImg = renderStripe(stripe_configs{g}, PATTERN_PX);
        label  = 'STRIPE';

    elseif i >= QUAD_START && i <= QUAD_END
        % Quadrant/block block
        g      = i - QUAD_START + 1;
        patImg = renderQuad(quad_configs{g}, DISPLAY_GRID32, PATTERN_PX);
        label  = 'QUAD';

    elseif i >= RING_START && i <= RING_END
        % Ring/annular block
        g      = i - RING_START + 1;
        patImg = renderRing(ring_configs{g}, XX, YY);
        label  = 'RING';

    else
        % Validation 4x4 RBP (VAL_START–VAL_END)
        k      = i - VAL_START + 1;
        core   = valCores{k};
        patImg = kron(core, ones(MACRO_PIXEL, 'uint8')) * 255;
        label  = 'VAL';
    end

    % Display pattern
    dispCanvas = buildCanvas(double(patImg)/255, padTop, padLeft, screenH, screenW, PATTERN_PX);
    set(hImg, 'CData', repmat(dispCanvas, 1, 1, 3));
    title(ax, sprintf('Pattern %d / %d  [%s]', i, NUM_PATTERNS, label), ...
          'Color', 'w', 'FontSize', 16);
    drawnow;
    pause(SETTLE_S);

    % Capture camera frame
    trigger(vid);
    camFrames{i}         = getdata(vid, 1);
    patternTimestamps(i) = toc(tStart);

    % Save full-resolution pattern
    imwrite(patImg, fullfile(OUT_DIR, sprintf('output_%05d.png', i)));

    % Save 32x32 thumbnail
    if ~isempty(core)
        % 4x4-based patterns -> upscale to 32x32
        core32 = kron(core, ones(SCALE_32, 'uint8')) * 255;
    else
        % 32x32-based patterns -> downscale to 32x32
        core32 = imresize(patImg, [32 32], 'nearest');
    end
    imwrite(core32, fullfile(OUT_DIR_32, sprintf('output32_%05d.png', i)));

    fprintf('  Pattern %5d / %5d  [%s]  elapsed: %.2f s\n', ...
            i, NUM_PATTERNS, label, toc(tStart));
end

stop(vid);
tCapture = toc(tStart);
fprintf('\nCapture done in %.2f s — writing camera frames...\n', tCapture);

%% 6. WRITE CAMERA FRAMES TO DISK
for i = 1:NUM_PATTERNS
    if ~isempty(camFrames{i})
        imwrite(camFrames{i}, fullfile(INPUT_DIR, sprintf('input_%05d.png', i)));
    end
end
tTotal = toc(tStart);

%% 7. SAVE LOG
log.numPatterns       = NUM_PATTERNS;
log.numBinaryPre      = NUM_BINARY_PRE;
log.numBinaryPost     = NUM_BINARY_POST;
log.numHad            = NUM_HAD;
log.numSparse         = NUM_SPARSE;
log.numSpots          = NUM_SPOT;
log.numStripes        = NUM_STRIPE;
log.numQuads          = NUM_QUAD;
log.numRings          = NUM_RING;
log.numVal            = NUM_VAL;

log.allOnIdx          = ALLON_IDX;
log.allOffIdx         = ALLOFF_IDX;
log.hadRange          = [HAD_START HAD_END];
log.sparseRange       = [SPARSE_START SPARSE_END];
log.spotRange         = [SPOT_START SPOT_END];
log.stripeRange       = [STRIPE_START STRIPE_END];
log.quadRange         = [QUAD_START QUAD_END];
log.ringRange         = [RING_START RING_END];
log.valRange          = [VAL_START VAL_END];

log.displayGrid       = DISPLAY_GRID;
log.displayGrid32     = DISPLAY_GRID32;
log.macroPixel        = MACRO_PIXEL;
log.patternPx         = PATTERN_PX;
log.scale32           = SCALE_32;
log.settleS           = SETTLE_S;
log.cameraFPS         = CAMERA_FPS;
log.totalSeconds      = tTotal;
log.captureSeconds    = tCapture;
log.patternTimestamps = patternTimestamps;
log.acquisitionDate   = datestr(now);
log.inputDir          = INPUT_DIR;
log.outputDir         = OUT_DIR;

log.spotRadiusRange   = [SPOT_RADIUS_MIN SPOT_RADIUS_MAX];
log.spotMargin        = SPOT_MARGIN;
log.spotConfigs       = spot_configs;
log.binaryCoresPre    = binaryCores_pre;
log.binaryCoresPost   = binaryCores_post;
log.hadCores          = hadCores;
log.sparseCores       = sparseCores;
log.stripeConfigs     = stripe_configs;
log.quadConfigs       = quad_configs;
log.ringConfigs       = ring_configs;
log.valCores          = valCores;

save(LOG_FILE, 'log', '-v7.3');
fprintf('Log saved: %s\n', LOG_FILE);

%% 8. CLEANUP
delete(vid);
close(fig);

fprintf('\n=== COMPLETE ===\n');
fprintf('  Total patterns : %d\n',       NUM_PATTERNS);
fprintf('  Capture time   : %.2f s\n',   tCapture);
fprintf('  Total time     : %.2f s\n',   tTotal);