function trainAutoencodersForAttacks()
    % Train and evaluate an autoencoder on multiple attack datasets.
    %
    % This function loads four labeled CSV datasets (SQL Injection, XSS, DoS, Path Traversal),
    % vectorizes text fields using TF-IDF, normalizes features, trains an autoencoder on
    % "normal" samples only, computes an anomaly threshold, and evaluates detection performance
    % via ROC curves (AUC). All comments are in English.

    % Define datasets: {filepath, display name, text column name}
    datasets = {
                '../datasets/sql_injection_dataset.csv', 'SQL Injection', 'query';
                '../datasets/xss_dataset.csv', 'XSS', 'payload';
                '../datasets/dos_dataset.csv', 'DoS', 'request';
                '../datasets/path_traversal_dataset.csv', 'Path Traversal', 'path'
                };

    % Training options for the autoencoder
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 64, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'training-progress', ...
        'Verbose', false);

    maxSamples = 20000; % cap on number of samples per dataset

    % Process datasets one by one
    for i = 1:size(datasets, 1)
        filePath = datasets{i, 1};
        attackName = datasets{i, 2};
        textCol = datasets{i, 3};

        fprintf('\n=== Processing %s ===\n', attackName);

        % Check for file existence
        if ~isfile(filePath)
            warning('  File not found: %s', filePath);
            continue;
        end

        try
            % 1. Load data
            T = readtable(filePath, 'TextType', 'string');

            % 2. Subsample if too large
            if strcmp(attackName, 'DoS')
                sampleSize = 10000;
            else
                sampleSize = maxSamples;
            end

            if height(T) > sampleSize
                idx = randperm(height(T), sampleSize);
                T = T(idx, :);
                fprintf('  Subsampled to %d samples\n', sampleSize);
            end

            % 3. Extract labels
            labels = T.label;

            % 4. Separate numeric columns (all except textCol & label)
            varNames = T.Properties.VariableNames;
            isNumeric = ~strcmp(varNames, textCol) & ~strcmp(varNames, 'label');
            numericData = table2array(T(:, isNumeric));

            % 5. Convert text column to TF-IDF features (sparse for memory)
            docs = tokenizedDocument(T.(textCol));
            bag = bagOfWords(docs);
            bag = removeInfrequentWords(bag, 2);
            Xtxt = sparse(tfidf(bag, docs)); % Use sparse matrix

            % 6. Combine numeric + text features, remove rows with NaNs
            X = [numericData, full(Xtxt)]; % Convert to full only for final matrix
            valid = all(~isnan(X), 2);
            X = X(valid, :);
            labels = labels(valid);

            % Clear large intermediate variables
            clear T docs bag Xtxt;

            % 7. Split into train (70%) and test (30%)
            cv = cvpartition(labels, 'HoldOut', 0.3);
            XTrain = X(training(cv), :);
            yTrain = labels(training(cv));
            XTest = X(test(cv), :);
            yTest = labels(test(cv));

            % 8. Compute normalization parameters from training data
            mu = mean(XTrain, 1);
            sigma = std(XTrain, 0, 1);
            sigma(sigma == 0) = 1; % prevent division by zero

            % 9. Apply z-score normalization
            XTrainN = (XTrain - mu) ./ sigma;
            XTestN = (XTest - mu) ./ sigma;

            % 10. Select only normal samples (label == 0) for autoencoder training
            normalIdx = (yTrain == 0);
            XNormal = XTrainN(normalIdx, :);

            if isempty(XNormal)
                warning('  No normal samples found for %s', attackName);
                continue;
            end

            % 11. Define autoencoder architecture
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

            % 12. Train the autoencoder on normal data only
            net = trainNetwork(XNormal, XNormal, layers, options);

            % 13. Compute anomaly threshold = mean + 3*std of reconstruction error
            reconNormal = predict(net, XNormal);
            errTrain = mean((XNormal - reconNormal) .^ 2, 2);
            threshold = mean(errTrain) + 3 * std(errTrain);
            fprintf('  Threshold for %s: %.4f\n', attackName, threshold);

            % 14. Evaluate on the test set
            reconTest = predict(net, XTestN);
            errTest = mean((XTestN - reconTest) .^ 2, 2);
            [fpr, tpr, ~, AUC] = perfcurve(yTest, errTest, 1);
            fprintf('  AUC for %s: %.4f\n', attackName, AUC);

            % 15.1 Compute binary predictions based on threshold
            yPred = errTest > threshold;
            yTestBinary = (yTest == 1);

            % 15.2 Compute metrics
            accuracy = sum(yPred == yTestBinary) / numel(yTestBinary);
            precision = sum(yPred & yTestBinary) / sum(yPred);
            recall = sum(yPred & yTestBinary) / sum(yTestBinary);
            f1_score = 2 * (precision * recall) / (precision + recall);

            % 15.3 Print metrics
            fprintf('\nАтака: %s\n', attackName);
            fprintf('Accuracy:  %.4f\n', accuracy);
            fprintf('Precision: %.4f\n', precision);
            fprintf('Recall:    %.4f\n', recall);
            fprintf('F1-score:  %.4f\n\n', f1_score);

            % 16. Plot reconstruction errors (fixed scatter call)
            figure('Name', [attackName ' Reconstruction Errors']);
            scatter(1:length(errTest), errTest, 'filled');
            hold on;
            yline(threshold, 'r--', 'LineWidth', 1.5);
            title([attackName ': Reconstruction Errors']);
            xlabel('Sample Index');
            ylabel('Reconstruction Error');
            grid on;

            % 17. Plot ROC curve
            figure('Name', [attackName ' ROC Curve']);
            plot(fpr, tpr, 'LineWidth', 2);
            title([attackName ' ROC (AUC = ' num2str(AUC, '%.2f') ')']);
            xlabel('False Positive Rate');
            ylabel('True Positive Rate');
            grid on;

        catch ME
            fprintf('  Error processing %s: %s\n', attackName, ME.message);
            continue;
        end

    end

end

% Call the function
trainAutoencodersForAttacks();
