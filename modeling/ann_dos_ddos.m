function trainAndEvaluateANN_DosDdos(maxSamples)
    % Loading parameters from the "Deep Network Designer" tool
    params = load("params_ann.mat");

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

    % Dataset path and attack name
    filePath = '../datasets/dos_ddos_opensource.csv';
    attackName = 'DoS/DDOS';

    fprintf('\n=== Processing %s ===\n', attackName);

    try
        % 1. Load data with original column names
        raw = readtable(filePath, 'VariableNamingRule', 'preserve');
        fprintf('Original dataset size: %d rows, %d columns\n', height(raw), width(raw));

        % 2. Subsample if dataset is too large
        if height(raw) > maxSamples
            idx = randperm(height(raw), maxSamples);
            raw = raw(idx, :);
            fprintf('Subsampled to %d samples\n', maxSamples);
        end

        % 2. Prepare labels (last column is 'Label')
        labels_col = raw.Properties.VariableNames{end};
        labels = raw.(labels_col);

        % Convert string labels to binary (BENIGN=0, attack=1)
        if iscell(labels) || isstring(labels)
            fprintf('Converting string labels to binary...\n');
            is_attack = ~strcmpi(labels, 'BENIGN');
            labels = double(is_attack);
        end

        % 3. Select numeric features (exclude non-numeric columns)
        nonNumericCols = {'Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', ...
                              'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', 'Label'};
        numericCols = ~ismember(raw.Properties.VariableNames, nonNumericCols);
        X = table2array(raw(:, numericCols));

        % 4. Handle missing values
        X(isnan(X)) = 0;
        fprintf('Using %d numeric features\n', size(X, 2));

        % 5. Normalize (Z-score)
        [X, mu, sigma] = zscore(X);
        sigma(sigma == 0) = 1;

        % 6. Reduce dimensionality to 100 using PCA
        if size(X, 2) > 100
            fprintf('Reducing features to 100 using PCA\n');
            [~, X] = pca(X, 'NumComponents', 100);
        elseif size(X, 2) < 100
            fprintf('Padding features to 100\n');
            X = [X, zeros(size(X, 1), 100 - size(X, 2))];
        end

        % 7. Train/test split (70/30)
        cv = cvpartition(labels, 'HoldOut', 0.3);
        XTrain = X(training(cv), :);
        XTest = X(test(cv), :);
        yTrain = categorical(labels(training(cv)));
        yTest = labels(test(cv));

        % 8. Train the network
        fprintf('Training network for %s attack...\n', attackName);
        net = trainNetwork(XTrain, yTrain, layers, options);

        % 9. Predict and evaluate
        [~, scores] = classify(net, XTest);
        yPred = scores(:, 2); % Probability of class 1
        yPredBinary = yPred > 0.5;

        % 10. Metrics
        TP = sum((yTest == 1) & (yPredBinary == 1));
        TN = sum((yTest == 0) & (yPredBinary == 0));
        FP = sum((yTest == 0) & (yPredBinary == 1));
        FN = sum((yTest == 1) & (yPredBinary == 0));

        accuracy = (TP + TN) / numel(yTest);
        precision = TP / (TP + FP + eps);
        recall = TP / (TP + FN + eps);
        f1_score = 2 * (precision * recall) / (precision + recall + eps);

        fprintf('\nАтака: %s\n', attackName);
        fprintf('Accuracy:  %.4f\n', accuracy);
        fprintf('Precision: %.4f\n', precision);
        fprintf('Recall:    %.4f\n', recall);
        fprintf('F1-score:  %.4f\n\n', f1_score);

        % 11. ROC curve
        [fpr, tpr, ~, AUC] = perfcurve(yTest, yPred, 1);
        figure('Name', [attackName ' ROC Curve']);
        plot(fpr, tpr, 'LineWidth', 2);
        title(sprintf('ROC Curve (%s, AUC = %.4f)', attackName, AUC));
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        grid on;

        % 12. Save the model
        save('trained_ANN_DoS_DDoS_opensource.mat', 'net');

    catch ME
        fprintf('Error processing %s: %s\n', attackName, ME.message);
    end

    fprintf('Training completed\n');
end

% Call the function
trainAndEvaluateANN_DosDdos(20000);
