function trainAndEvaluateANN_XSS(maxSamples)
    % Define the architecture of the network
    layers = [
              featureInputLayer(100, "Name", "featureinput")
              fullyConnectedLayer(64, "Name", "fc")
              reluLayer("Name", "relu")
              fullyConnectedLayer(32, "Name", "fc_1")
              reluLayer("Name", "relu_1")
              fullyConnectedLayer(2, "Name", "fc_2")
              softmaxLayer("Name", "softmax")
              classificationLayer("Name", "classification")
              ];

    % Configure training
    options = trainingOptions('adam', ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 1e-3, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'training-progress', ...
        'Verbose', true);

    % Dataset configuration
    filePath = '../datasets/xss_opensource.csv';
    attackName = 'XSS Attack';

    % Set default if not provided
    if nargin < 1
        maxSamples = 20000; % Default from autoencoder_xss.m
    end

    fprintf('\n=== Processing %s (max %d samples) ===\n', attackName, maxSamples);

    try
        % 1. Load data with original column names (as in svm_xss.m)
        raw = readtable(filePath, 'VariableNamingRule', 'preserve');
        fprintf('Original dataset size: %d rows, %d columns\n', height(raw), width(raw));

        % 2. Subsample if too large (consistent with autoencoder_xss.m)
        if height(raw) > maxSamples
            idx = randperm(height(raw), maxSamples);
            raw = raw(idx, :);
            fprintf('Subsampled to %d samples\n', maxSamples);
        end

        % 3. Prepare labels (last column, as in all reference files)
        labels_col = raw.Properties.VariableNames{end};
        labels = raw.(labels_col);

        % Convert to binary if needed (like in logistic_regression_xss.m)
        if iscell(labels) || isstring(labels)
            labels = double(strcmpi(labels, 'malicious')); % Adapt based on actual labels
        end

        % 4. Prepare features (all except last column, like svm_xss.m)
        features = raw(:, 1:end - 1);
        X = table2array(features);

        % 5. Handle missing values (as in all reference files)
        X(isnan(X)) = 0;
        fprintf('Using %d numeric features\n', size(X, 2));

        % 6. Normalize (Z-score, consistent with other models)
        [X, mu, sigma] = zscore(X);
        sigma(sigma == 0) = 1;

        % 7. Dimensionality adjustment (ANN-specific)
        if size(X, 2) > 100
            fprintf('Reducing features to 100 using PCA\n');
            [~, X] = pca(X, 'NumComponents', 100);
        elseif size(X, 2) < 100
            fprintf('Padding features to 100\n');
            X = [X, zeros(size(X, 1), 100 - size(X, 2))];
        end

        % 8. Train/test split (70/30, consistent with all references)
        cv = cvpartition(labels, 'HoldOut', 0.3);
        XTrain = X(training(cv), :);
        XTest = X(test(cv), :);
        yTrain = categorical(labels(training(cv))); % For classificationLayer
        yTest = labels(test(cv));

        % 9. Train the network
        fprintf('Training network for %s...\n', attackName);
        net = trainNetwork(XTrain, yTrain, layers, options);

        % 10. Predict and evaluate
        [~, scores] = classify(net, XTest);
        yPred = scores(:, 2) > 0.5; % Binary predictions

        % 11. Metrics (consistent format with other models)
        TP = sum((yTest == 1) & yPred);
        TN = sum((yTest == 0) & ~yPred);
        FP = sum((yTest == 0) & yPred);
        FN = sum((yTest == 1) & ~yPred);

        accuracy = (TP + TN) / numel(yTest);
        precision = TP / (TP + FP + eps);
        recall = TP / (TP + FN + eps);
        f1_score = 2 * (precision * recall) / (precision + recall + eps);

        fprintf('\nАтака: %s\n', attackName);
        fprintf('Accuracy:  %.4f\n', accuracy);
        fprintf('Precision: %.4f\n', precision);
        fprintf('Recall:    %.4f\n', recall);
        fprintf('F1-score:  %.4f\n\n', f1_score);

        % 12. ROC curve (like in logistic_regression_xss.m)
        [fpr, tpr, ~, AUC] = perfcurve(yTest, scores(:, 2), 1);
        figure('Name', [attackName ' ROC Curve']);
        plot(fpr, tpr, 'LineWidth', 2);
        title(sprintf('ROC Curve (%s, AUC = %.4f)', attackName, AUC));
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        grid on;

        % 13. Save model (consistent naming)
        save('trained_ANN_XSS_opensource.mat', 'net', 'mu', 'sigma');

    catch ME
        fprintf('Error processing %s: %s\n', attackName, ME.message);
    end

    fprintf('Training completed\n');
end

trainAndEvaluateANN_XSS(50000);
