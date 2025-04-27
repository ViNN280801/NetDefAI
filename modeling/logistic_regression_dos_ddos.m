function evaluate_logistic_regression_dos(filename, attackName)
    %----------------------------------------------------------------------
    %  Reads a CSV file with DoS/DDoS features and binary labels, splits data into
    %  train / test, and evaluates a logistic regression classifier.
    %  Results (metrics + ROC curve) are printed and plotted.
    %
    %  filename     – path to CSV file
    %  attackName   – string used in printed output and plot title
    %----------------------------------------------------------------------

    % 1. Load data with original column names
    fprintf('Loading data from %s...\n', filename);
    raw = readtable(filename, 'VariableNamingRule', 'preserve');

    % Display some information about the data
    fprintf('Dataset size: %d rows, %d columns\n', height(raw), width(raw));

    % 2. Prepare features and labels
    % Get the last column as labels
    labels_col = raw.Properties.VariableNames{end};
    labels = raw.(labels_col);

    % Check if labels are strings and convert to binary if needed
    if iscell(labels) || isstring(labels)
        fprintf('Converting string labels to binary...\n');
        % Assume "BENIGN" is 0 and all attacks are 1
        is_attack = ~strcmpi(labels, 'BENIGN');
        labels = double(is_attack);
    end

    % Get all numeric features
    features = raw(:, 1:end - 1);

    % Create a logical array marking numeric columns
    numericCols = varfun(@isnumeric, features, 'OutputFormat', 'uniform');

    % Keep only numeric columns for analysis
    numericFeatures = features(:, numericCols);
    fprintf('Using %d numeric features out of %d total columns\n', sum(numericCols), width(features));

    % Convert to array
    X = table2array(numericFeatures);

    % 3. Handle missing values
    % Replace NaN with column means
    nanCols = any(isnan(X));
    fprintf('Columns with NaN values: %d\n', sum(nanCols));

    if any(nanCols)
        fprintf('Replacing NaN values with column means...\n');
        colMeans = nanmean(X);

        for j = 1:size(X, 2)
            nanIdx = isnan(X(:, j));
            X(nanIdx, j) = colMeans(j);
        end

    end

    % Check for remaining NaNs or Infs
    if any(isnan(X(:))) || any(isinf(X(:)))
        fprintf('Removing rows with remaining NaN or Inf values...\n');
        validRows = all(isfinite(X), 2);
        X = X(validRows, :);
        labels = labels(validRows);
        fprintf('Remaining dataset size: %d rows\n', size(X, 1));
    end

    % 4. Train / test split ----------------------------------------------
    cv = cvpartition(labels, 'HoldOut', 0.3);
    XTrain = X(training(cv), :);
    XTest = X(test(cv), :);
    yTrain = labels(training(cv));
    yTest = labels(test(cv));

    fprintf('Training set: %d samples\n', size(XTrain, 1));
    fprintf('Test set: %d samples\n', size(XTest, 1));

    % 5. Normalize features (Z-score)
    [XTrain, mu, sigma] = zscore(XTrain);

    % Handle potential zero standard deviations
    sigma(sigma == 0) = 1;
    XTest = (XTest - mu) ./ sigma;

    % 6. Train logistic regression ---------------------------------------
    fprintf('Training logistic regression model...\n');
    model = fitclinear(XTrain, yTrain, 'Learner', 'logistic');
    [yPred, scr] = predict(model, XTest); % scr(:,2) – score for class 1

    % 7. Metrics ----------------------------------------------------------
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

    % 8. ROC curve --------------------------------------------------------
    [rocX, rocY, ~, AUC] = perfcurve(yTest, scr(:, 2), 1);
    figure;
    plot(rocX, rocY, 'LineWidth', 2);
    title(sprintf('ROC кривая (%s, AUC = %.4f)', attackName, AUC));
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    grid on;
end

%-----------------------------------------------------------------------
%  Run evaluation
%-----------------------------------------------------------------------
% Check if dataset path exists, if not try different locations
dataset_path = '../datasets/dos_ddos_opensource.csv';

if ~exist(dataset_path, 'file')
    dataset_path = '../dataset/dos_ddos_opensource.csv';

    if ~exist(dataset_path, 'file')
        dataset_path = 'dos_ddos_opensource.csv';

        if ~exist(dataset_path, 'file')
            error('Dataset file not found. Please check the path.');
        end

    end

end

evaluate_logistic_regression_dos(dataset_path, 'DoS/DDoS Attack');
