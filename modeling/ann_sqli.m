function trainAndEvaluateANN_SQLi(maxSamples)
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
    filePath = '../datasets/sql_injections_opensource.csv';
    attackName = 'SQL Injection';
    textColumn = 'Query'; % As in autoencoder_sqli.m and svm_sqli.m

    % Set default if not provided
    if nargin < 1
        maxSamples = 20000; % Consistent with other models
    end

    fprintf('\n=== Processing %s (max %d samples) ===\n', attackName, maxSamples);

    try
        % 1. Load data with text column (as in autoencoder_sqli.m)
        raw = readtable(filePath, 'TextType', 'string');
        fprintf('Original dataset size: %d rows, %d columns\n', height(raw), width(raw));

        % 2. Subsample if too large (consistent with other models)
        if height(raw) > maxSamples
            idx = randperm(height(raw), maxSamples);
            raw = raw(idx, :);
            fprintf('Subsampled to %d samples\n', maxSamples);
        end

        % 3. Prepare labels (last column, as in logistic_regression_sqli.m)
        labels = raw.Label;

        % 4. Tokenize and create TF-IDF (as in svm_sqli.m)
        docs = tokenizedDocument(raw.(textColumn));
        bag = bagOfWords(docs);
        bag = removeInfrequentWords(bag, 2); % Keep words appearing ≥2 times (as in logistic_regression_sqli.m)
        X = full(tfidf(bag, docs)); % Convert to full matrix for ANN

        % 5. Normalize (L2 norm per row, as in autoencoder_sqli.m)
        rowNorms = sqrt(sum(X .^ 2, 2)) + eps;
        X = X ./ rowNorms;

        % 6. Dimensionality adjustment (ANN-specific)
        if size(X, 2) > 100
            fprintf('Reducing features to 100 using PCA\n');
            [~, X] = pca(X, 'NumComponents', 100);
        elseif size(X, 2) < 100
            fprintf('Padding features to 100\n');
            X = [X, zeros(size(X, 1), 100 - size(X, 2))];
        end

        % 7. Train/test split (70/30, consistent with all references)
        cv = cvpartition(labels, 'HoldOut', 0.3);
        XTrain = X(training(cv), :);
        XTest = X(test(cv), :);
        yTrain = categorical(labels(training(cv))); % For classificationLayer
        yTest = labels(test(cv));

        % 8. Train the network
        fprintf('Training network for %s...\n', attackName);
        net = trainNetwork(XTrain, yTrain, layers, options);

        % 9. Predict and evaluate
        [~, scores] = classify(net, XTest);
        yPred = scores(:, 2) > 0.5; % Binary predictions

        % 10. Metrics (consistent format with other models)
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

        % 11. ROC curve (as in logistic_regression_sqli.m)
        [fpr, tpr, ~, AUC] = perfcurve(yTest, scores(:, 2), 1);
        figure('Name', [attackName ' ROC Curve']);
        plot(fpr, tpr, 'LineWidth', 2);
        title(sprintf('ROC Curve (%s, AUC = %.4f)', attackName, AUC));
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        grid on;

        % 12. Save model (consistent naming)
        save('trained_ANN_SQLi_opensource.mat', 'net', 'bag'); % Save bag for future text processing

    catch ME
        fprintf('Error processing %s: %s\n', attackName, ME.message);
    end

    fprintf('Training completed\n');
end

trainAndEvaluateANN_SQLi();
