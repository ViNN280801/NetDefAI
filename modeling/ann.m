function trainAndEvaluateANN()
    % Loading parameters from the "Deep Network Designer" tool
    params = load("params_ann.mat");

    % Define the architecture of the network directly without loading parameters
    layers = [
              featureInputLayer(100, "Name", "featureinput")
              fullyConnectedLayer(64, "Name", "fc")
              reluLayer("Name", "relu")
              fullyConnectedLayer(32, "Name", "fc_1")
              reluLayer("Name", "relu_1")
              fullyConnectedLayer(2, "Name", "fc_2")
              softmaxLayer("Name", "softmax")
              classificationLayer("Name", "classification") % Needed for trainNetwork
              ];

    % Configure training
    options = trainingOptions('adam', ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 1e-3, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'training-progress', ...
        'Verbose', true);

    % Process datasets
    datasets = {
                %'../datasets/sql_injection_dataset.csv', 'SQL Injection', 'query';
               % '../datasets/xss_dataset.csv', 'XSS', 'payload';
                '../datasets/dos_dataset.csv', 'DoS', 'request';
               % '../datasets/path_traversal_dataset.csv', 'Path Traversal', 'path'
                };

    for i = 1:size(datasets, 1)
        filePath = datasets{i, 1};
        attackName = datasets{i, 2};
        textCol = datasets{i, 3};

        fprintf('\n=== Processing %s ===\n', attackName);

        try
            % Loading and processing data
            if strcmp(attackName, 'DoS')
                numSamples = 10000; % Limit for DoS
                minWordCount = 5; % Increase threshold for rare words
            else
                numSamples = 20000; % Standard value for other attacks
                minWordCount = 2;
            end

            T = readtable(filePath, 'TextType', 'string');
            T = T(1:numSamples, :);
            texts = T.(textCol);
            labels = T.label;

            % Tokenization and TF-IDF with adjusted minWordCount for DoS
            docs = tokenizedDocument(texts);
            bag = bagOfWords(docs);
            bag = removeInfrequentWords(bag, minWordCount); % Dynamic threshold
            X = full(tfidf(bag, docs));

            % Force PCA for DoS if feature size is too large
            if strcmp(attackName, 'DoS') && size(X, 2) > 100
                fprintf('Reducing features to 100 using PCA (DoS-specific)\n');
                [~, X] = pca(X, 'NumComponents', 100);
            else
                % Standard logic for other attacks
                if size(X, 2) < 100
                    fprintf('Padding features to 100\n');
                    X = [X, zeros(size(X, 1), 100 - size(X, 2))];
                elseif size(X, 2) > 100
                    fprintf('Reducing features to 100 using PCA\n');
                    [~, X] = pca(X, 'NumComponents', 100);
                end

            end

            % Splitting into train/test
            cv = cvpartition(labels, 'HoldOut', 0.3);
            XTrain = X(training(cv), :);
            yTrain = categorical(labels(training(cv))); % Categorical labels for classification
            XTest = X(test(cv), :);
            yTest = labels(test(cv));

            % Training
            fprintf('Training network for %s attack...\n', attackName);
            net = trainNetwork(XTrain, yTrain, layers, options);

            % Prediction - for classification returns scores
            [~, scores] = classify(net, XTest);
            yPred = scores(:, 2); % Probability of class 1
            yPredBinary = yPred > 0.5;

            % Metrics
            TP = sum((yTest == 1) & (yPredBinary == 1));
            TN = sum((yTest == 0) & (yPredBinary == 0));
            FP = sum((yTest == 0) & (yPredBinary == 1));
            FN = sum((yTest == 1) & (yPredBinary == 0));

            accuracy = (TP + TN) / numel(yTest);
            precision = TP / (TP + FP + eps);
            recall = TP / (TP + FN + eps);
            f1_score = 2 * (precision * recall) / (precision + recall + eps);

            % Printing metrics
            fprintf('\nАтака: %s\n', attackName);
            fprintf('Accuracy:  %.4f\n', accuracy);
            fprintf('Precision: %.4f\n', precision);
            fprintf('Recall:    %.4f\n', recall);
            fprintf('F1-score:  %.4f\n\n', f1_score);

            % ROC curve
            [fpr, tpr, ~, AUC] = perfcurve(yTest, yPred, 1);
            figure('Name', [attackName ' ROC Curve']);
            plot(fpr, tpr, 'LineWidth', 2);
            title(sprintf('ROC Curve (%s, AUC = %.4f)', attackName, AUC));
            xlabel('False Positive Rate');
            ylabel('True Positive Rate');
            grid on;

            % Saving the model for the current dataset
            save(sprintf('trained_ANN_%s.mat', attackName), 'net');

        catch ME
            fprintf('Error processing %s: %s\n', attackName, ME.message);
        end

    end

    fprintf('Training completed\n');
end

trainAndEvaluateANN();
