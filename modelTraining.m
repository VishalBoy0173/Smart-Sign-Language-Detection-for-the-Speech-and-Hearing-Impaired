%% ASL Alphabet Recognition Model Training
clc; clear; close all;

fprintf('=== ASL Alphabet Recognition Model Training ===\n');

%% 1. Dataset Loading and Preparation
dataFolder = fullfile('asl_alphabet_train'); % Make sure this folder exists

% Verify the dataset structure
if ~exist(dataFolder, 'dir')
    error('Dataset folder "asl_alphabet_train" not found. Please ensure it exists in the current directory.');
end

% List all subfolders (classes)
classFolders = dir(dataFolder);
classFolders = classFolders([classFolders.isdir]); % Get only directories
classFolders = classFolders(~ismember({classFolders.name}, {'.', '..'})); % Remove . and ..

fprintf('Found %d classes:\n', length(classFolders));
for i = 1:min(length(classFolders), 10) % Show first 10 classes
    fprintf('  - %s\n', classFolders(i).name);
end
if length(classFolders) > 10
    fprintf('  ... and %d more\n', length(classFolders)-10);
end

% Create image datastore
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
inputSize = [224 224 3];

% Split data: 80% training, 20% validation
rng(1); % For reproducibility
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

fprintf('Training images: %d\n', numel(imdsTrain.Files));
fprintf('Validation images: %d\n', numel(imdsVal.Files));

% Display sample images
fprintf('Displaying sample images...\n');
figure('Name', 'Sample Images from Dataset');
montage(imds.Files(1:30));
title('Sample ASL Alphabet Images');

%% 3. Data Augmentation
fprintf('Applying data augmentation...\n');
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-10, 10], ...
    'RandYTranslation', [-10, 10], ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandXScale', [0.9, 1.1], ...
    'RandYScale', [0.9, 1.1]);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
augVal = augmentedImageDatastore(inputSize, imdsVal);

%% 4. Load Pretrained Network (ResNet-18)
fprintf('Loading ResNet-18 and modifying for ASL classification...\n');

% Check if Deep Learning Toolbox is available
if ~license('test', 'Neural_Network_Toolbox')
    error('Deep Learning Toolbox is required. Please install it from MATLAB Add-Ons.');
end

% Load ResNet-18
try
    net = resnet18;
catch
    error('ResNet-18 not found. You may need to install the Deep Learning Toolbox Model for ResNet-18 Network from Add-Ons.');
end

% Create layer graph and modify for our task
lgraph = layerGraph(net);

% Remove the last 3 layers (classification layers)
lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});

% Number of classes (should be 29 for ASL Alphabet: A-Z, space, del, nothing)
numClasses = numel(categories(imdsTrain.Labels));
fprintf('Number of classes: %d\n', numClasses);

% Add new classification layers
newLayers = [
    fullyConnectedLayer(512, 'Name', 'fc512', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    reluLayer('Name', 'relu1')
    dropoutLayer(0.5, 'Name', 'dropout1')
    fullyConnectedLayer(256, 'Name', 'fc256', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    reluLayer('Name', 'relu2')
    dropoutLayer(0.3, 'Name', 'dropout2')
    fullyConnectedLayer(numClasses, 'Name', 'fc_final', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'fc512');

% Display the modified network architecture
analyzeNetwork(lgraph);

%% 5. Training Options
fprintf('Setting up training options...\n');

% Check for GPU availability
if canUseGPU()
    execEnv = 'gpu';
    fprintf('GPU detected. Using GPU for training.\n');
else
    execEnv = 'cpu';
    fprintf('No GPU detected. Using CPU for training (this will be slower).\n');
end

% Training options
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', execEnv, ...
    'CheckpointPath', 'checkpoints', ...
    'CheckpointFrequency', 2);

% Create checkpoint folder if it doesn't exist
if ~exist('checkpoints', 'dir')
    mkdir('checkpoints');
end

%% 6. Train the Model
fprintf('Training the ASL model...\n');
fprintf('This may take some time (30-60 minutes depending on your hardware).\n');
fprintf('Training progress will be displayed in a separate window.\n');

tic;
aslNet = trainNetwork(augTrain, lgraph, options);
trainingTime = toc;

fprintf('Training completed in %.2f minutes (%.2f hours).\n', trainingTime/60, trainingTime/3600);

%% 7. Evaluate Model Performance
fprintf('Evaluating model performance...\n');

% Predict on validation set
[predictedLabels, scores] = classify(aslNet, augVal);
trueLabels = imdsVal.Labels;

% Calculate accuracy
accuracy = mean(predictedLabels == trueLabels);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

% Display confusion matrix
figure('Name', 'Confusion Matrix', 'Position', [100, 100, 800, 600]);
confusionchart(trueLabels, predictedLabels);
title(sprintf('Confusion Matrix - ASL Alphabet (Accuracy: %.2f%%)', accuracy * 100));

% Display some sample predictions with images
figure('Name', 'Sample Predictions', 'Position', [100, 100, 1200, 400]);
numSamples = 12;
indices = randperm(numel(trueLabels), min(numSamples, numel(trueLabels)));

for i = 1:length(indices)
    idx = indices(i);
    img = readimage(imdsVal, idx);
    img = imresize(img, [224, 224]);
    
    subplot(3, 4, i);
    imshow(img);
    
    predLabel = char(predictedLabels(idx));
    trueLabel = char(trueLabels(idx));
    
    if strcmp(predLabel, trueLabel)
        titleColor = 'g';
    else
        titleColor = 'r';
    end
    
    title(sprintf('True: %s\nPred: %s', trueLabel, predLabel), ...
        'Color', titleColor, 'FontSize', 10);
end

%% 8. Save Model Properly
fprintf('Saving trained model...\n');

% Method 1: Save as MAT file with -v7.3 for large files
save('asl_cnn_model.mat', 'aslNet', 'inputSize', '-v7.3');

% Method 2: Also save with simpler name for easy loading
save('trained_asl_network.mat', 'aslNet', 'inputSize', '-v7.3');

% Method 3: Save network information for debugging
networkInfo = struct();
networkInfo.Layers = aslNet.Layers;
networkInfo.Connections = aslNet.Connections;
networkInfo.InputSize = inputSize;
networkInfo.ClassNames = categories(trueLabels);
networkInfo.TrainingTime = trainingTime;
networkInfo.Accuracy = accuracy;
save('asl_network_info.mat', 'networkInfo', '-v7.3');

fprintf('Model saved successfully!\n');
fprintf('  - asl_cnn_model.mat (main model file)\n');
fprintf('  - trained_asl_network.mat (backup)\n');
fprintf('  - asl_network_info.mat (debugging info)\n');

% Display model summary
fprintf('\n=== Model Summary ===\n');
fprintf('Network Architecture: Modified ResNet-18\n');
fprintf('Input Size: %d x %d x %d\n', inputSize(1), inputSize(2), inputSize(3));
fprintf('Number of Classes: %d\n', numClasses);
fprintf('Number of Layers: %d\n', numel(aslNet.Layers));
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Training Time: %.2f minutes\n', trainingTime/60);

%% 9. Test the Model with Sample Image
fprintf('\nTesting model with a sample image...\n');

% Try to classify one image from validation set
testImg = readimage(imdsVal, 1);
testImg = imresize(testImg, inputSize(1:2));

% Make sure image is RGB
if size(testImg, 3) == 1
    testImg = cat(3, testImg, testImg, testImg);
end

[testPred, testScores] = classify(aslNet, testImg);
testConfidence = max(testScores) * 100;

figure('Name', 'Model Test');
imshow(testImg);
title(sprintf('Test Prediction: %s (%.1f%% confidence)', ...
    char(testPred), testConfidence), 'FontSize', 14);

fprintf('Test prediction: %s (%.1f%% confidence)\n', char(testPred), testConfidence);

%% 10. Clean up checkpoint files (optional)
if exist('checkpoints', 'dir')
    fprintf('Cleaning up checkpoint files...\n');
    checkpointFiles = dir(fullfile('checkpoints', '*.mat'));
    for i = 1:length(checkpointFiles)
        delete(fullfile('checkpoints', checkpointFiles(i).name));
    end
    rmdir('checkpoints');
end

fprintf('\n=== Training Complete ===\n');
fprintf('You can now run your main GUI script.\n');
