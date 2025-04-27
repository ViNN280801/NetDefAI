function evaluate_logistic_regression_sqli(filename, attackName)
    %----------------------------------------------------------------------
    %  Reads a CSV file with SQLi queries and binary labels, splits data into
    %  train/test, builds a TF-IDF representation, and evaluates logistic regression.
    %  Results (metrics + ROC curve) are printed and plotted.
    %
    %  filename     – path to CSV file
    %  attackName   – string used in printed output and plot title
    %----------------------------------------------------------------------

    % 1. Load data --------------------------------------------------------
    fprintf('Loading data from %s...\n', filename);
    raw = readtable(filename, 'TextType', 'string');
    texts = raw.Query;
    labels = raw.Label;

    % 2. Tokenize queries -------------------------------------------------
    documents = tokenizedDocument(texts);

    % 3. Train/test split -------------------------------------------------
    cv = cvpartition(labels, 'HoldOut', 0.3);
    docsTrain = documents(training(cv));
    docsTest = documents(test(cv));
    yTrain = labels(training(cv));
    yTest = labels(test(cv));

    % 4. Build vocabulary and TF-IDF on training set ----------------------
    bag = bagOfWords(docsTrain);
    bag = removeInfrequentWords(bag, 2); % Keep words appearing at least twice
    XTrain = tfidf(bag, docsTrain);
    XTest = tfidf(bag, docsTest);

    % 5. Row-wise L2-normalization ---------------------------------------
    rowNorms = sqrt(sum(XTrain .^ 2, 2)) + eps;
    XTrain = bsxfun(@rdivide, XTrain, rowNorms);
    rowNorms = sqrt(sum(XTest .^ 2, 2)) + eps;
    XTest = bsxfun(@rdivide, XTest, rowNorms);

    % 6. Train logistic regression ----------------------------------------
    fprintf('Training model...\n');
    model = fitclinear(XTrain, yTrain, 'Learner', 'logistic');
    [yPred, scr] = predict(model, XTest);

    % 7. Metrics ----------------------------------------------------------
    TP = sum((yTest == 1) & (yPred == 1));
    TN = sum((yTest == 0) & (yPred == 0));
    FP = sum((yTest == 0) & (yPred == 1));
    FN = sum((yTest == 1) & (yPred == 0));

    accuracy = (TP + TN) / numel(yTest);
    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    f1_score = 2 * (precision * recall) / (precision + recall + eps);

    fprintf('\nAttack: %s\n', attackName);
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
dataset_path = '../datasets/sql_injections_opensource.csv';

if ~exist(dataset_path, 'file')
    dataset_path = '../dataset/sql_injections_opensource.csv';

    if ~exist(dataset_path, 'file')
        error('SQLi dataset not found. Please check the path.');
    end

end

evaluate_logistic_regression_sqli(dataset_path, 'SQL Injection');
