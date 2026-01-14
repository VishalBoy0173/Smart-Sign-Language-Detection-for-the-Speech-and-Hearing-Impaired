% test_webcam.m - Standalone test script for webcam detection
clc; clear; close all;

% Load the trained model
disp('Loading ASL CNN model...');
load('asl_cnn_model.mat'); % This should load 'aslNet'

% Get input size from the network
if exist('aslNet', 'var')
    inputSize = aslNet.Layers(1).InputSize;
    disp(['Input size: ', num2str(inputSize(1)), 'x', num2str(inputSize(2))]);
    
    % Run the webcam detection
    run_live_webcam_detection_testingv2(aslNet, inputSize);
else
    error('Model file does not contain "aslNet". Check your .mat file.');

end
