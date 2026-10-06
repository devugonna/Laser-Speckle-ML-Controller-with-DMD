% EXPOSURE_CALIBRATION_TEST  Stage 01 - camera exposure calibration
% Captures one frame of whatever is displayed on the DMD at five exposure
% levels around BASE_EXPOSURE and saves them for visual comparison.
%   Needs hardware : YES (ZWO camera + DMD showing a test image)
%   Output         : results/calibration/exposure_images/exposure_<value>.png
function exposure_calibration_test()
    % Five-Level Exposure Calibration Capture
    % Captures the existing image on Screen 2 once per second
    % for 5 exposure levels: (E-1.0, E-0.5, E, E+0.5, E+1.0)

    %% USER SETTINGS
    CAMERA_INDEX  = 2;
    CAMERA_FORMAT = 'RGB8_800x640';
    BASE_EXPOSURE = -6.8;      % current exposure level
    GAIN          = 10.3;      % keep existing gain
    scriptDir     = fileparts(mfilename('fullpath'));
    if isempty(scriptDir), scriptDir = pwd; end
    repoRoot      = fileparts(scriptDir);
    SAVE_DIR      = fullfile(repoRoot, 'results', 'calibration', 'exposure_images');

    if ~exist(SAVE_DIR, 'dir')
        mkdir(SAVE_DIR);
    end

    % Build 5 exposure levels centered on BASE_EXPOSURE
    exposureLevels = BASE_EXPOSURE + [-1.0, -0.5, 0, +0.5, +1.0];

    %% CAMERA INIT
    imaqreset;
    vid = videoinput('winvideo', CAMERA_INDEX, CAMERA_FORMAT);
    src = getselectedsource(vid);

    src.Gain = GAIN;
    vid.FramesPerTrigger = 1;
    triggerconfig(vid, 'manual');
    start(vid);

    fprintf('\n=== Exposure Calibration Test ===\n');
    fprintf('Base exposure: %.2f\n', BASE_EXPOSURE);
    fprintf('Levels: %s\n\n', mat2str(exposureLevels));

    %% CAPTURE LOOP
    for k = 1:length(exposureLevels)
        expVal = exposureLevels(k);
        src.Exposure = expVal;

        fprintf('Setting exposure = %.2f ...\n', expVal);
        pause(1);   % allow camera to settle

        trigger(vid);
        frame = getdata(vid, 1);

        fname = fullfile(SAVE_DIR, sprintf('exposure_%+0.2f.png', expVal));
        imwrite(frame, fname);

        fprintf('  Saved: %s\n', fname);
    end

    %% CLEANUP
    stop(vid);
    delete(vid);

    fprintf('\n=== COMPLETE ===\n');
    fprintf('Images saved to: %s\n', SAVE_DIR);
end