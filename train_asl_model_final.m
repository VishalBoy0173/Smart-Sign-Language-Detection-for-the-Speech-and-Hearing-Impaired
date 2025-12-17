%% ASL Alphabet Recognition Model Training
clc; clear; close all;

fprintf('=== ASL Alphabet Recognition Model Training ===\n');

%% 1. Dataset Loading and Preparation
% Use your specific dataset path
dataFolder = 'G:\SEM 5\BAXI 3533 ARTIFICIAL INTELLIGENCE PROJECT MANAGEMENT\Project\Sign Language\Sign Language\asl_alphabet_train';

% Verify the dataset structure
fprintf('Checking dataset at: %s\n', dataFolder);

if ~exist(dataFolder, 'dir')
    % If not found, try to locate it automatically
    fprintf('Dataset not found at specified path.\n');
    fprintf('Trying to locate dataset...\n');
    
    % Try to find it in common locations
    possibleLocations = {
        pwd,  % Current directory
        'G:\SEM 5\BAXI 3533 ARTIFICIAL INTELLIGENCE PROJECT MANAGEMENT\Project',
        'G:\SEM 5\BAXI 3533 ARTIFICIAL INTELLIGENCE PROJECT MANAGEMENT',
        'G:\',
        fullfile(getenv('USERPROFILE'), 'Downloads'),
        fullfile(getenv('USERPROFILE'), 'Desktop')
    };
    
    found = false;
    for i = 1:length(possibleLocations)
        searchPath = possibleLocations{i};
        fprintf('Searching in: %s\n', searchPath);
        
        % List all folders
        allFolders = dir(searchPath);
        for j = 1:length(allFolders)
            if allFolders(j).isdir && contains(allFolders(j).name, 'asl', 'IgnoreCase', true)
                candidate = fullfile(searchPath, allFolders(j).name);
                % Check if it contains letter folders
                subfolders = dir(candidate);
                hasLetters = false;
                for k = 1:min(5, length(subfolders))  % Check first 5
                    if subfolders(k).isdir && length(subfolders(k).name) == 1
                        hasLetters = true;
                        break;
                    end
                end
                
                if hasLetters
                    dataFolder = candidate;
                    found = true;
                    fprintf('Found candidate dataset: %s\n', dataFolder);
                    break;
                end
            end
        end
        if found, break; end
    end
    
    if ~found
        % Ask user to select manually
        fprintf('Please select the dataset folder manually...\n');
        dataFolder = uigetdir(pwd, 'Select the ASL Alphabet Dataset Folder');
        if dataFolder == 0
            error('No folder selected. Training cancelled.');
        end
    end
end

fprintf('Using dataset from: %s\n', dataFolder);

% List all subfolders (classes) to verify
classFolders = dir(dataFolder);
classFolders = classFolders([classFolders.isdir]); % Get only directories
classFolders = classFolders(~ismember({classFolders.name}, {'.', '..'})); % Remove . and ..

fprintf('Found %d classes:\n', length(classFolders));
for i = 1:min(length(classFolders), 10) % Show first 10 classes
    subfolderPath = fullfile(dataFolder, classFolders(i).name);
    numImages = numel(dir(fullfile(subfolderPath, '*.jpg'))) + ...
                numel(dir(fullfile(subfolderPath, '*.png'))) + ...
                numel(dir(fullfile(subfolderPath, '*.jpeg')));
    fprintf('  - %s: %d images\n', classFolders(i).name, numImages);
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
fprintf('\nDataset Summary:\n');
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
figure('Name', 'Sample Images from Dataset', 'Position', [100, 100, 1200, 400]);
numSamples = min(12, numel(imds.Files));
sampleIndices = randperm(numel(imds.Files), numSamples);

for i = 1:numSamples
    img = readimage(imds, sampleIndices(i));
    subplot(3, 4, i);
    imshow(img);
    title(sprintf('%s', char(imds.Labels(sampleIndices(i)))), 'FontSize', 10);
end
sgtitle('Sample Images from ASL Alphabet Dataset');

%% 3. Data Augmentation
fprintf('Applying data augmentation...\n');
augmenter = imageDataAugmenter( ...
    'RandRotation', [-15, 15], ...
    'RandXTranslation', [-20, 20], ...
    'RandYTranslation', [-20, 20], ...
    'RandXReflection', true, ...
    'RandYReflection', false, ...
    'RandScale', [0.8, 1.2]);

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
    fprintf('ResNet-18 loaded successfully.\n');
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

% Display class names
classNames = categories(imdsTrain.Labels);
fprintf('Class names: ');
for i = 1:min(10, numClasses)
    fprintf('%s ', char(classNames(i)));
end
if numClasses > 10
    fprintf('... and %d more', numClasses-10);
end
fprintf('\n');

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
fprintf('Network architecture modified successfully.\n');
% analyzeNetwork(lgraph);  % Uncomment to view detailed network analysis

%% 5. Training Options
fprintf('Setting up training options...\n');

% Check for GPU availability
if canUseGPU()
    execEnv = 'gpu';
    gpuDevice = gpuDevice();
    fprintf('GPU detected: %s\n', gpuDevice.Name);
    fprintf('GPU Memory: %.2f GB\n', gpuDevice.AvailableMemory / 1e9);
else
    execEnv = 'cpu';
    fprintf('No GPU detected. Using CPU for training (this will be slower).\n');
end

% Training options
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...  % Reduced for faster training
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 3, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', execEnv, ...
    'CheckpointPath', 'checkpoints', ...
    'CheckpointFrequency', 2, ...
    'L2Regularization', 0.0001);

% Create checkpoint folder if it doesn't exist
if ~exist('checkpoints', 'dir')
    mkdir('checkpoints');
    fprintf('Created checkpoints folder for saving intermediate models.\n');
end

%% 6. Train the Model
fprintf('\n=== Starting Model Training ===\n');
fprintf('Training Parameters:\n');
fprintf('  - Epochs: %d\n', options.MaxEpochs);
fprintf('  - Batch Size: %d\n', options.MiniBatchSize);
fprintf('  - Initial Learning Rate: %.4f\n', options.InitialLearnRate);
fprintf('  - Execution Environment: %s\n', execEnv);
fprintf('  - Training Images: %d\n', numel(imdsTrain.Files));
fprintf('  - Validation Images: %d\n', numel(imdsVal.Files));
fprintf('\nTraining may take 20-40 minutes depending on your hardware...\n');
fprintf('Training progress will be displayed in a separate window.\n');

% Start timer
startTime = datetime('now');
fprintf('Training started at: %s\n', datestr(startTime));

tic;
aslNet = trainNetwork(augTrain, lgraph, options);
trainingTime = toc;

endTime = datetime('now');
fprintf('Training completed at: %s\n', datestr(endTime));
fprintf('Total training time: %.2f minutes\n', trainingTime/60);

%% 7. Evaluate Model Performance
fprintf('\nEvaluating model performance...\n');

% Predict on validation set
fprintf('Making predictions on validation set...\n');
[predictedLabels, scores] = classify(aslNet, augVal);
trueLabels = imdsVal.Labels;

% Calculate accuracy
accuracy = mean(predictedLabels == trueLabels);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

% Calculate per-class accuracy
classAccuracy = zeros(numClasses, 1);
for i = 1:numClasses
    classIdx = (trueLabels == classNames(i));
    if any(classIdx)
        classAccuracy(i) = mean(predictedLabels(classIdx) == trueLabels(classIdx)) * 100;
    end
end

fprintf('\nPer-class accuracy (top 10):\n');
[~, idx] = sort(classAccuracy, 'descend');
for i = 1:min(10, numClasses)
    fprintf('  %s: %.1f%%\n', char(classNames(idx(i))), classAccuracy(idx(i)));
end

% Display confusion matrix
figure('Name', 'Confusion Matrix', 'Position', [100, 100, 900, 700]);
confusionchart(trueLabels, predictedLabels, ...
    'Title', sprintf('Confusion Matrix - ASL Alphabet (Overall Accuracy: %.2f%%)', accuracy * 100), ...
    'ColumnSummary', 'column-normalized', ...
    'RowSummary', 'row-normalized');

% Display some sample predictions with images
figure('Name', 'Sample Predictions', 'Position', [100, 100, 1400, 500]);
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
    confidence = max(scores(idx, :)) * 100;
    
    if strcmp(predLabel, trueLabel)
        titleColor = [0, 0.7, 0]; % Green
        result = '✓ Correct';
    else
        titleColor = [0.9, 0, 0]; % Red
        result = '✗ Wrong';
    end
    
    title({sprintf('True: %s', trueLabel), ...
           sprintf('Pred: %s', predLabel), ...
           sprintf('Conf: %.1f%%', confidence), ...
           result}, ...
           'Color', titleColor, 'FontSize', 9);
end
sgtitle('Sample Validation Predictions');

%% 8. Save Model Properly
fprintf('\nSaving trained model...\n');

% Save current directory for reference
currentDir = pwd;
fprintf('Current directory: %s\n', currentDir);

% Method 1: Save as MAT file with -v7.3 for large files
modelPath = fullfile(currentDir, 'asl_cnn_model.mat');
save(modelPath, 'aslNet', 'inputSize', 'classNames', 'accuracy', '-v7.3');
fprintf('✓ Main model saved: %s\n', modelPath);

% Method 2: Also save with simpler name for easy loading
backupPath = fullfile(currentDir, 'trained_asl_network.mat');
save(backupPath, 'aslNet', 'inputSize', 'classNames', '-v7.3');
fprintf('✓ Backup model saved: %s\n', backupPath);

% Method 3: Save network information for debugging
networkInfo = struct();
networkInfo.Layers = aslNet.Layers;
networkInfo.Connections = aslNet.Connections;
networkInfo.InputSize = inputSize;
networkInfo.ClassNames = classNames;
networkInfo.TrainingTime = trainingTime;
networkInfo.Accuracy = accuracy;
networkInfo.TrainingStartTime = startTime;
networkInfo.TrainingEndTime = endTime;
networkInfo.DatasetPath = dataFolder;

infoPath = fullfile(currentDir, 'asl_network_info.mat');
save(infoPath, 'networkInfo', '-v7.3');
fprintf('✓ Network info saved: %s\n', infoPath);

% Save a simple test script
testScript = sprintf(['%% Test script for ASL model\n' ...
                      'clc; clear; close all;\n' ...
                      'load(''%s'');\n' ...
                      'fprintf(''Model loaded successfully!\\n'');\n' ...
                      'fprintf(''Accuracy: %.2f%%%%\\n'', accuracy*100);\n' ...
                      'fprintf(''Number of classes: %%d\\n'', length(classNames));\n'], ...
                      'asl_cnn_model.mat', accuracy*100);
fid = fopen('test_model.m', 'w');
fprintf(fid, '%s', testScript);
fclose(fid);
fprintf('✓ Test script created: test_model.m\n');

%% 9. Test the Model with Sample Image
fprintf('\nTesting model with a sample image...\n');

% Try to classify one image from validation set
testIdx = randi(numel(imdsVal.Files));
testImg = readimage(imdsVal, testIdx);
testImg = imresize(testImg, inputSize(1:2));

% Make sure image is RGB
if size(testImg, 3) == 1
    testImg = cat(3, testImg, testImg, testImg);
end

[testPred, testScores] = classify(aslNet, testImg);
testConfidence = max(testScores) * 100;
trueLabel = imdsVal.Labels(testIdx);

figure('Name', 'Model Test', 'Position', [100, 100, 600, 500]);
imshow(testImg);

if strcmp(char(testPred), char(trueLabel))
    titleColor = 'g';
    resultText = 'CORRECT';
else
    titleColor = 'r';
    resultText = 'INCORRECT';
end

title({sprintf('ASL Sign Recognition Test'), ...
       sprintf('True Label: %s', char(trueLabel)), ...
       sprintf('Predicted: %s (%.1f%% confidence)', char(testPred), testConfidence), ...
       resultText}, ...
       'FontSize', 14, 'Color', titleColor);

fprintf('Test prediction: %s (True: %s, Confidence: %.1f%%)\n', ...
    char(testPred), char(trueLabel), testConfidence);

%% 10. Display Final Summary
fprintf('\n=== TRAINING COMPLETE ===\n');
fprintf('Model Summary:\n');
fprintf('  - Network: Modified ResNet-18\n');
fprintf('  - Input Size: %d x %d x %d\n', inputSize(1), inputSize(2), inputSize(3));
fprintf('  - Number of Classes: %d\n', numClasses);
fprintf('  - Number of Layers: %d\n', numel(aslNet.Layers));
fprintf('  - Validation Accuracy: %.2f%%\n', accuracy * 100);
fprintf('  - Training Time: %.2f minutes\n', trainingTime/60);
fprintf('  - Dataset: %s\n', dataFolder);
fprintf('  - Model Files Saved:\n');
fprintf('     1. asl_cnn_model.mat (main model)\n');
fprintf('     2. trained_asl_network.mat (backup)\n');
fprintf('     3. asl_network_info.mat (info)\n');
fprintf('     4. test_model.m (test script)\n');

fprintf('\nNext Steps:\n');
fprintf('  1. Run your main GUI script to use the trained model\n');
fprintf('  2. Test with: run_live_webcam_detection_testingv2(aslNet, inputSize)\n');
fprintf('  3. Test with: image_file_prediction(aslNet, inputSize)\n');

%% 11. Clean up checkpoint files (optional)
if exist('checkpoints', 'dir')
    fprintf('\nCleaning up checkpoint files...\n');
    checkpointFiles = dir(fullfile('checkpoints', '*.mat'));
    if ~isempty(checkpointFiles)
        fprintf('Found %d checkpoint files. Deleting...\n', length(checkpointFiles));
        for i = 1:length(checkpointFiles)
            delete(fullfile('checkpoints', checkpointFiles(i).name));
        end
        rmdir('checkpoints');
        fprintf('Checkpoints cleaned up.\n');
    end
end

fprintf('\n=== ALL DONE! ===\n');