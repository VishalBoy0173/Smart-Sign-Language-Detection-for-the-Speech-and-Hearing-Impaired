%% ASL Alphabet Recognition Model Training (Enhanced & Final)
clc; clear; close all;

fprintf('=== ASL Alphabet Recognition Model Training ===\n');

%% 1. Dataset Loading
dataFolder = 'G:\SEM 5\BAXI 3533 ARTIFICIAL INTELLIGENCE PROJECT MANAGEMENT\Project\Sign Language\Sign Language\asl_alphabet_train';

if ~exist(dataFolder, 'dir')
    dataFolder = uigetdir(pwd, 'Select ASL Alphabet Dataset Folder');
    if dataFolder == 0
        error('Dataset folder not selected.');
    end
end

fprintf('Using dataset from: %s\n', dataFolder);

imds = imageDatastore(dataFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

labelTbl = countEachLabel(imds);
disp(labelTbl);

%% 2. Train / Validation Split
inputSize = [224 224 3];
rng(1);
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

fprintf('Training images: %d\n', numel(imdsTrain.Files));
fprintf('Validation images: %d\n', numel(imdsVal.Files));

classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);

%% 3. Data Augmentation
augmenter = imageDataAugmenter( ...
    'RandRotation', [-15 15], ...
    'RandXTranslation', [-20 20], ...
    'RandYTranslation', [-20 20], ...
    'RandScale', [0.8 1.2], ...
    'RandXReflection', true);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', augmenter);

augVal = augmentedImageDatastore(inputSize, imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

%% 4. Load & Modify ResNet-18
net = resnet18;
lgraph = layerGraph(net);

lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});

newLayers = [
    fullyConnectedLayer(512, 'Name','fc512', ...
        'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    reluLayer('Name','relu1')
    dropoutLayer(0.5,'Name','drop1')

    fullyConnectedLayer(256, 'Name','fc256', ...
        'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    reluLayer('Name','relu2')
    dropoutLayer(0.3,'Name','drop2')

    fullyConnectedLayer(numClasses,'Name','fc_final', ...
        'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
];

% Class weights for imbalance
classWeights = max(labelTbl.Count) ./ labelTbl.Count;

newClassLayer = classificationLayer( ...
    'Name','classoutput', ...
    'Classes', classNames, ...
    'ClassWeights', classWeights);

lgraph = addLayers(lgraph, [newLayers newClassLayer]);
lgraph = connectLayers(lgraph, 'pool5', 'fc512');

%% 5. Freeze Early Layers
layers = lgraph.Layers;
connections = lgraph.Connections;

for i = 1:20
    if isprop(layers(i),'WeightLearnRateFactor')
        layers(i).WeightLearnRateFactor = 0;
        layers(i).BiasLearnRateFactor = 0;
    end
end

lgraph = createLgraphUsingConnections(layers, connections);

%% 6. Training Options
if canUseGPU()
    execEnv = 'gpu';
    g = gpuDevice;
    fprintf('GPU: %s (%.2f GB)\n', g.Name, g.AvailableMemory/1e9);
else
    execEnv = 'cpu';
    fprintf('Using CPU\n');
end

options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 3e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.3, ...
    'LearnRateDropPeriod',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment', execEnv, ...
    'L2Regularization',1e-4);

%% 7. Train Network
fprintf('\n=== Training Started ===\n');
tic;
aslNet = trainNetwork(augTrain, lgraph, options);
trainingTime = toc;

fprintf('Training time: %.2f minutes\n', trainingTime/60);

%% 8. Evaluation
[predictedLabels, scores] = classify(aslNet, augVal);
trueLabels = imdsVal.Labels;

accuracy = mean(predictedLabels == trueLabels);
fprintf('Validation Accuracy: %.2f%%\n', accuracy*100);

% Top-5 Accuracy
top5Correct = 0;
for i = 1:size(scores,1)
    [~, idx] = sort(scores(i,:), 'descend');
    if any(classNames(idx(1:5)) == trueLabels(i))
        top5Correct = top5Correct + 1;
    end
end
top5Accuracy = top5Correct / numel(trueLabels) * 100;
fprintf('Top-5 Accuracy: %.2f%%\n', top5Accuracy);

% Confusion Matrix
figure;
confusionchart(trueLabels, predictedLabels, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
title(sprintf('Confusion Matrix (Acc: %.2f%%)', accuracy*100));

%% 9. Save Model
modelVersion = datestr(now,'yyyy-mm-dd_HHMM');

save(sprintf('asl_model_%s.mat', modelVersion), ...
    'aslNet','inputSize','classNames','accuracy','trainingTime','-v7.3');

save('asl_cnn_model.mat', ...
    'aslNet','inputSize','classNames','accuracy','-v7.3');

fprintf('Model saved successfully.\n');

%% 10. Quick Test Prediction
idx = randi(numel(imdsVal.Files));
img = readimage(imdsVal, idx);
img = imresize(img, inputSize(1:2));

if size(img,3)==1
    img = cat(3,img,img,img);
end

[pred, sc] = classify(aslNet, img);

figure;
imshow(img);
title(sprintf('True: %s | Pred: %s (%.1f%%)', ...
    char(trueLabels(idx)), char(pred), max(sc)*100));

%% 11. Final Summary
fprintf('\n=== TRAINING COMPLETE ===\n');
fprintf('Classes           : %d\n', numClasses);
fprintf('Validation Acc    : %.2f%%\n', accuracy*100);
fprintf('Top-5 Accuracy    : %.2f%%\n', top5Accuracy);
fprintf('Training Time     : %.2f min\n', trainingTime/60);
fprintf('Model Ready for:\n');
fprintf(' - image_file_prediction(aslNet, inputSize)\n');
fprintf(' - live webcam GUI\n');
fprintf(' - project presentation\n');
