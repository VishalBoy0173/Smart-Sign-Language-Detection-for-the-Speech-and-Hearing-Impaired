function image_file_prediction(aslNet, inputSize)
    % IMAGE_FILE_PREDICTION
    % Predict ASL sign from an image file using a trained CNN
    
    % Select image
    [file, path] = uigetfile({'*.jpg;*.png;*.jpeg'}, 'Select an Image File');
    if isequal(file, 0)
        disp('No image selected.');
        return;
    end

    try
        % Read image
        img = imread(fullfile(path, file));
    catch
        error('Unable to read the selected image file.');
    end

    % Resize image to network input size
    img = imresize(img, inputSize(1:2));

    % Convert grayscale to RGB if needed
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end

    % Convert image to single precision if required
    if ~isa(img, 'single')
        img = im2single(img);
    end

    % Classify image
    [prediction, scores] = classify(aslNet, img);

    % Get confidence score
    confidence = max(scores) * 100;

    % Get top-3 predictions
    [sortedScores, idx] = sort(scores, 'descend');
    classNames = aslNet.Layers(end).Classes;
    top3Classes = classNames(idx(1:3));
    top3Scores = sortedScores(1:3) * 100;

    % Display image with prediction
    figure('Name', 'ASL Image Prediction', 'NumberTitle', 'off');
    imshow(img);
    title(sprintf('Prediction: %s (%.2f%%)', ...
        char(prediction), confidence), 'FontSize', 16);

    % Console output
    disp('------------------------------------');
    disp(['Predicted Sign : ', char(prediction)]);
    disp(['Confidence     : ', num2str(confidence, '%.2f'), '%']);
    disp('Top-3 Predictions:');
    for i = 1:3
        fprintf('%d) %s - %.2f%%\n', i, char(top3Classes(i)), top3Scores(i));
    end
    disp('------------------------------------');
end
