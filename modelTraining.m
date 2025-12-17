clc; clear; close all;

fprintf('=== ASL Alphabet Recognition Model Training ===\n');

%% 1. Dataset Loading and Preparation
% [Elango] Loading dataset from the folder structured by ASL letter classes
dataFolder = fullfile('asl_alphabet_train'); % Adjust this path if needed
if ~exist(dataFolder, 'dir')
    error('Dataset folder "asl_alphabet_train" not found.');
end

fprintf('Loading dataset...\n');
imds = imageDatastore(dataFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Display label distribution
labelCount = countEachLabel(imds);
disp(labelCount);

fprintf('Total images: %d\n', numel(imds.Files));
fprintf('Total classes: %d\n', numel(unique(imds.Labels)));

%% 2. Data Preprocessing and Splitting
% [Rishika] Resizing all input images and splitting into training and validation sets
inputSize = [224 224 3];

rng(1); % For reproducibility
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

fprintf('Training images: %d\n', numel(imdsTrain.Files));
fprintf('Validation images: %d\n', numel(imdsVal.Files));

%% 3. Data Augmentation
% [Vishal] Applying data augmentation to reduce overfitting and improve generalization
fprintf('Applying data augmentation...\n');
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10,10], ...
    'RandXTranslation', [-5 5], ...
    'RandYTranslation', [-5 5], ...
    'RandXScale', [0.9 1.1], ...
    'RandYScale', [0.9 1.1]);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
augVal = augmentedImageDatastore(inputSize, imdsVal);

%% 4. Load Pretrained Network (ResNet-18)
% [Elango] Modifying ResNet-18 to classify 29 ASL letters (A-Z, space, nothing, del)
fprintf('Loading ResNet-18 and modifying for ASL classification...\n');
net = resnet18;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});

numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
    fullyConnectedLayer(256, 'Name', 'fc256', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    reluLayer('Name', 'relu_new')
    dropoutLayer(0.5, 'Name', 'dropout')
    fullyConnectedLayer(numClasses, 'Name', 'fc_final', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
];
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'fc256');

%% 5. Training Options
% [Rishika] Setting training parameters and enabling GPU acceleration if available
fprintf('Setting up training options...\n');
if canUseGPU()
    execEnv = 'gpu';
    disp('Using GPU for training.');
else
    execEnv = 'cpu';
    disp('Using CPU for training.');
end

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 64, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', execEnv);

%% 6. Train the Model
% [Vishal] Start training the CNN with augmented data and monitor training time
fprintf('Training the ASL model...\n');
tic;
aslNet = trainNetwork(augTrain, lgraph, options);
trainingTime = toc;
fprintf('Training completed in %.2f minutes.\n', trainingTime/60);

%% 7. Evaluate Accuracy
fprintf('Evaluating model performance...\n');
predictedLabels = classify(aslNet, augVal);
trueLabels = imdsVal.Labels;
accuracy = mean(predictedLabels == trueLabels);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

% Confusion matrix to visualize model performance
figure;
confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix - ASL Alphabet');

%% 8. (Save Model)
% [Elango] Saving the trained model and input size to be used during prediction
fprintf('Saving trained model...\n');
save('asl_cnn_model.mat', 'aslNet', 'inputSize');

fprintf('Model saved as "asl_cnn_model.mat"\n');
fprintf('=== Training Done ===\n');

