function trainAutoencoderForSQLi()
    % Train and evaluate an autoencoder on SQL Injection dataset.
    %
    % This function loads the labeled CSV dataset, converts text to TF-IDF,
    % trains an autoencoder on "normal" samples only, computes an anomaly threshold,
    % and evaluates detection performance via ROC curves (AUC).

    % Dataset configuration
    filePath = '../datasets/sql_injections_opensource.csv';
    attackName = 'SQL Injection';
    textColumn = 'Query'; % Column with SQL queries
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
        % 1. Load data and extract text/labels
        raw = readtable(filePath, 'TextType', 'string');
        texts = raw.(textColumn);
        labels = raw.Label;

        % 2. Subsample if too large
        if height(raw) > maxSamples
            idx = randperm(height(raw), maxSamples);
            texts = texts(idx);
            labels = labels(idx);
            fprintf('Subsampled to %d samples\n', maxSamples);
        end

        % 3. Tokenize and create bag-of-words
        docs = tokenizedDocument(texts);
        bag = bagOfWords(docs);
        bag = removeInfrequentWords(bag, 2); % Keep words appearing ≥2 times

        % 4. Convert to TF-IDF matrix (sparse for memory)
        X = tfidf(bag, docs);
        X = full(X); % Autoencoder requires full matrix

        % 5. Train/test split (70/30)
        cv = cvpartition(labels, 'HoldOut', 0.3);
        XTrain = X(training(cv), :);
        XTest = X(test(cv), :);
        yTrain = labels(training(cv));
        yTest = labels(test(cv));

        % 6. Normalize (L2 norm per row)
        rowNorms = sqrt(sum(XTrain .^ 2, 2)) + eps;
        XTrain = XTrain ./ rowNorms;
        rowNorms = sqrt(sum(XTest .^ 2, 2)) + eps;
        XTest = XTest ./ rowNorms;

        % 7. Select only normal samples (label=0) for training
        XNormal = XTrain(yTrain == 0, :);

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

        % 9. Train autoencoder
        net = trainNetwork(XNormal, XNormal, layers, options);

        % 10. Compute anomaly threshold (mean + 3*std)
        reconNormal = predict(net, XNormal);
        errTrain = mean((XNormal - reconNormal) .^ 2, 2);
        threshold = mean(errTrain) + 3 * std(errTrain);
        fprintf('Threshold for %s: %.4f\n', attackName, threshold);

        % 11. Evaluate on test set
        reconTest = predict(net, XTest);
        errTest = mean((XTest - reconTest) .^ 2, 2);
        [fpr, tpr, ~, AUC] = perfcurve(yTest, errTest, 1);
        fprintf('AUC for %s: %.4f\n', attackName, AUC);

        % 12. Compute metrics
        yPred = errTest > threshold;
        yTestBinary = yTest;

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
trainAutoencoderForSQLi();
