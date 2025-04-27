function trainAutoencoderForXSS()
    % Train and evaluate an autoencoder on the XSS dataset.
    %
    % This function loads the labeled CSV dataset, preprocesses the features,
    % trains an autoencoder on "normal" samples only, computes an anomaly threshold,
    % and evaluates detection performance via ROC curves (AUC).

    % Dataset configuration
    filePath = '../datasets/xss_opensource.csv';
    attackName = 'XSS Attack';
    maxSamples = 20000; % Cap on samples

    % Autoencoder training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 64, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'training-progress', ...
        'Verbose', false);

    fprintf('\n=== Processing %s ===\n', attackName);

    % Check file existence
    if ~isfile(filePath)
        warning('File not found: %s', filePath);
        return;
    end

    try
        % 1. Load data with original column names
        raw = readtable(filePath, 'VariableNamingRule', 'preserve');
        labels = raw.(raw.Properties.VariableNames{end}); % Last column is 'Class'
        isNormal = (labels == 0); % Normal = 0

        % 2. Subsample if too large
        if height(raw) > maxSamples
            idx = randperm(height(raw), maxSamples);
            raw = raw(idx, :);
            isNormal = isNormal(idx);
            fprintf('Subsampled to %d samples\n', maxSamples);
        end

        % 3. Extract numeric features (exclude non-numeric columns)
        nonNumericCols = {'Class'}; % Adjust if more non-numeric columns exist
        numericCols = ~ismember(raw.Properties.VariableNames, nonNumericCols);
        X = table2array(raw(:, numericCols));

        % 4. Handle missing values
        X(isnan(X)) = 0;

        % 5. Train/test split (70/30)
        cv = cvpartition(isNormal, 'HoldOut', 0.3);
        XTrain = X(training(cv), :);
        XTest = X(test(cv), :);
        yTrain = isNormal(training(cv));
        yTest = isNormal(test(cv));

        % 6. Normalize (Z-score)
        [XTrain, mu, sigma] = zscore(XTrain);
        sigma(sigma == 0) = 1;
        XTest = (XTest - mu) ./ sigma;

        % 7. Select only normal samples for training
        XNormal = XTrain(yTrain, :);

        if isempty(XNormal)
            warning('No normal samples found for %s', attackName);
            return;
        end

        % 8. Define autoencoder architecture
        inputSize = size(XNormal, 2);
        layers = [
                  featureInputLayer(inputSize, 'Name', 'input')
                  fullyConnectedLayer(64, 'Name', 'encoder1')
                  reluLayer('Name', 'relu1')
                  fullyConnectedLayer(32, 'Name', 'bottleneck')
                  reluLayer('Name', 'relu2')
                  fullyConnectedLayer(64, 'Name', 'decoder1')
                  reluLayer('Name', 'relu3')
                  fullyConnectedLayer(inputSize, 'Name', 'output')
                  regressionLayer('Name', 'mse')
                  ];

        % 9. Train the autoencoder
        net = trainNetwork(XNormal, XNormal, layers, options);

        % 10. Compute anomaly threshold (mean + 3*std)
        reconNormal = predict(net, XNormal);
        errTrain = mean((XNormal - reconNormal) .^ 2, 2);
        threshold = mean(errTrain) + 3 * std(errTrain);
        fprintf('Threshold for %s: %.4f\n', attackName, threshold);

        % 11. Evaluate on test set
        reconTest = predict(net, XTest);
        errTest = mean((XTest - reconTest) .^ 2, 2);
        [fpr, tpr, ~, AUC] = perfcurve(~yTest, errTest, 1); % Invert labels for anomaly detection
        fprintf('AUC for %s: %.4f\n', attackName, AUC);

        % 12. Compute metrics
        yPred = errTest > threshold;
        yTestBinary = ~yTest; % Anomaly = not 'Normal' (label = 1)

        accuracy = sum(yPred == yTestBinary) / numel(yTestBinary);
        precision = sum(yPred & yTestBinary) / (sum(yPred) + eps);
        recall = sum(yPred & yTestBinary) / (sum(yTestBinary) + eps);
        f1_score = 2 * (precision * recall) / (precision + recall + eps);

        fprintf('\nАтака: %s\n', attackName);
        fprintf('Accuracy:  %.4f\n', accuracy);
        fprintf('Precision: %.4f\n', precision);
        fprintf('Recall:    %.4f\n', recall);
        fprintf('F1-score:  %.4f\n\n', f1_score);

        % 13. Plot reconstruction errors
        figure('Name', [attackName ' Reconstruction Errors']);
        scatter(1:length(errTest), errTest, 'filled');
        hold on;
        yline(threshold, 'r--', 'LineWidth', 1.5);
        title([attackName ': Reconstruction Errors']);
        xlabel('Sample Index');
        ylabel('Reconstruction Error');
        grid on;

        % 14. Plot ROC curve
        figure('Name', [attackName ' ROC Curve']);
        plot(fpr, tpr, 'LineWidth', 2);
        title([attackName ' ROC (AUC = ' num2str(AUC, '%.2f') ')']);
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        grid on;

    catch ME
        fprintf('Error processing %s: %s\n', attackName, ME.message);
    end

end

% Run the function
trainAutoencoderForXSS();
