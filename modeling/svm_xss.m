function evaluate_svm_xss(filename, attackName)
    %----------------------------------------------------------------------
    %  Reads a CSV file with numeric features and binary labels,
    %  splits data into train/test, and evaluates SVM classifier
    %  Results (metrics) are printed
    %
    %  filename     – path to CSV file
    %  attackName   – string used in printed output
    %----------------------------------------------------------------------

    % 1. Load data with original column names
    fprintf('Loading data from %s...\n', filename);
    raw = readtable(filename, 'VariableNamingRule', 'preserve');
    fprintf('Dataset size: %d rows, %d columns\n', height(raw), width(raw));

    % 2. Prepare labels (last column)
    labels_col = raw.Properties.VariableNames{end};
    labels = raw.(labels_col);

    % 3. Prepare features (all columns except last)
    features = raw(:, 1:end - 1);
    X = table2array(features);

    % 4. Handle missing values
    X(isnan(X)) = 0;

    % 5. Train/test split
    fprintf('Splitting data (70%% train, 30%% test)...\n');
    cv = cvpartition(labels, 'HoldOut', 0.3);
    XTrain = X(training(cv), :);
    XTest = X(test(cv), :);
    yTrain = labels(training(cv));
    yTest = labels(test(cv));

    % 6. Normalize (Z-score)
    [XTrain, mu, sigma] = zscore(XTrain);
    sigma(sigma == 0) = 1;
    XTest = (XTest - mu) ./ sigma;

    % 7. Train SVM
    fprintf('Training SVM model...\n');
    model = fitcsvm(XTrain, yTrain, 'KernelFunction', 'linear', ...
        'Standardize', false);

    % 8. Predict and evaluate
    [yPred, ~] = predict(model, XTest);

    % 9. Metrics
    TP = sum((yTest == 1) & (yPred == 1));
    TN = sum((yTest == 0) & (yPred == 0));
    FP = sum((yTest == 0) & (yPred == 1));
    FN = sum((yTest == 1) & (yPred == 0));

    accuracy = (TP + TN) / numel(yTest);
    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    f1_score = 2 * (precision * recall) / (precision + recall + eps);

    fprintf('\nАтака: %s\n', attackName);
    fprintf('Accuracy:  %.4f\n', accuracy);
    fprintf('Precision: %.4f\n', precision);
    fprintf('Recall:    %.4f\n', recall);
    fprintf('F1-score:  %.4f\n\n', f1_score);

    % 10. Feature Importance
    try
        fprintf('Calculating feature importance...\n');
        weights = model.Beta;
        [~, idx] = sort(abs(weights), 'descend');
        top_features = features.Properties.VariableNames(idx(1:5));
        disp('Top 5 important features:');
        disp(top_features');
    catch ME
        fprintf('Feature importance analysis failed: %s\n', ME.message);
    end

    % 11. 2D Visualization using PCA
    try
        fprintf('Creating 2D visualization...\n');

        % Combine train and test for consistent scaling
        XCombined = [XTrain; XTest];
        yCombined = [yTrain; yTest];

        % Sample max 10,000 points randomly
        num_points = min(10000, size(XCombined, 1));
        idx = randperm(size(XCombined, 1), num_points);
        XSampled = XCombined(idx, :);
        ySampled = yCombined(idx);

        % Apply PCA on sampled data
        [~, scorePCA] = pca(XSampled);
        projected = scorePCA(:, 1:2);

        % Plot
        figure('Name', 'XSS Attack 2D Projection');
        gscatter(projected(:, 1), projected(:, 2), ySampled, 'br', '..', [15 15]);
        title(sprintf('2D PCA Projection (%s)\n(Sampled %d points)', attackName, num_points));
        xlabel('PCA 1');
        ylabel('PCA 2');
        legend({'Normal', 'Attack'}, 'Location', 'best');
        grid on;
    catch ME
        fprintf('2D visualization failed: %s\n', ME.message);
    end

end

%-----------------------------------------------------------------------
%  Run evaluation
%-----------------------------------------------------------------------
dataset_path = '../datasets/xss_opensource.csv';

if ~exist(dataset_path, 'file')
    dataset_path = '../dataset/xss_opensource.csv';

    if ~exist(dataset_path, 'file')
        error('XSS dataset not found. Please check the path.');
    end

end

evaluate_svm_xss(dataset_path, 'XSS Attack');
