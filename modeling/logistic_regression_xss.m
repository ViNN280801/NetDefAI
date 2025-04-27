function evaluate_logistic_regression_xss(filename, attackName)
    %----------------------------------------------------------------------
    %  Reads a CSV file with XSS features and binary labels, splits data into
    %  train / test, and evaluates a logistic regression classifier.
    %  Results (metrics + ROC curve) are printed and plotted.
    %
    %  filename     – path to CSV file
    %  attackName   – string used in printed output and plot title
    %----------------------------------------------------------------------

    % 1. Load data with original column names
    raw = readtable(filename, 'VariableNamingRule', 'preserve');
    features = raw(:, 1:end - 1);
    labels = raw.(raw.Properties.VariableNames{end}); % Let's take the last column by name

    % 2. Convert features to matrix
    X = table2array(features);

    % 3. Train / test split ----------------------------------------------
    cv = cvpartition(labels, 'HoldOut', 0.3);
    XTrain = X(training(cv), :);
    XTest = X(test(cv), :);
    yTrain = labels(training(cv));
    yTest = labels(test(cv));

    % 4. Normalize features (Z-score)
    [XTrain, mu, sigma] = zscore(XTrain);
    XTest = (XTest - mu) ./ sigma;

    % 5. Train logistic regression ---------------------------------------
    model = fitclinear(XTrain, yTrain, 'Learner', 'logistic');
    [yPred, scr] = predict(model, XTest); % scr(:,2) – score for class 1

    % 6. Metrics ----------------------------------------------------------
    TP = sum((yTest == 1) & (yPred == 1));
    TN = sum((yTest == 0) & (yPred == 0));
    FP = sum((yTest == 0) & (yPred == 1));
    FN = sum((yTest == 1) & (yPred == 0));

    accuracy = (TP + TN) / numel(yTest);
    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    f1_score = 2 * (precision * recall) / (precision + recall + eps);

    fprintf('Атака: %s\n', attackName);
    fprintf('Accuracy:  %.4f\n', accuracy);
    fprintf('Precision: %.4f\n', precision);
    fprintf('Recall:    %.4f\n', recall);
    fprintf('F1-score:  %.4f\n\n', f1_score);

    % 7. ROC curve --------------------------------------------------------
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
evaluate_logistic_regression_xss('../datasets/xss_opensource.csv', 'XSS Attack');
