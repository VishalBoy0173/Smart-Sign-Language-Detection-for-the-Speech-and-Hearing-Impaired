clc; clear; close all;

load('asl_cnn_model.mat'); % Make sure this loads aslNet

% If inputSize not stored, extract it:
if ~exist('inputSize', 'var')
    inputSize = aslNet.Layers(1).InputSize;
end

%run_live_webcam_detection(aslNet, inputSize);

% Create dark-themed main GUI
mainFig = uifigure('Name', 'Sign Language Detection System', ...
    'Position', [100 100 500 400], ...
    'Color', [0.1 0.1 0.1]);

uilabel(mainFig, ...
    'Text', 'Sign Language Detection System', ...
    'FontSize', 24, ...
    'FontWeight', 'bold', ...
    'FontColor', 'w', ...
    'HorizontalAlignment', 'center', ...
    'Position', [100 330 300 50], ...
    'BackgroundColor', [0.1 0.1 0.1]);

% Button: Live Webcam Detection
makeStyledButton(mainFig, [150 240 200 50], '📷 Live Detection', ...
    @(btn,event) run_live_webcam_detection_testingv2(aslNet, inputSize));

% Button: Image File Prediction
makeStyledButton(mainFig, [150 170 200 50], '🖼️ Upload Image File', ...
    @(btn,event) image_file_prediction(aslNet, inputSize));

% Button: Exit
makeStyledButton(mainFig, [150 100 200 50], '❌ Exit', ...
    @(btn,event) close(mainFig));

% Function definitions must be at the end of the file

% Button creation function
function makeStyledButton(fig, pos, label, callbackFcn)
    btn = uibutton(fig, ...
        'Text', label, ...
        'Position', pos, ...
        'ButtonPushedFcn', callbackFcn);
    
    try
        btn.FontSize = 16;
        btn.FontWeight = 'bold';
        btn.BackgroundColor = [0.2, 0.2, 0.2];
        btn.FontColor = 'w';
    catch
        % Ignore style issues on older MATLAB versions
    end
end
