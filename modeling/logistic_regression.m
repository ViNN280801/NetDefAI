function evaluate_logistic_regression(filename, attackName, textColumn)
    %----------------------------------------------------------------------
    %  Reads a CSV file with texts and binary labels, splits data into
    %  train / test, builds a TF‑IDF representation only on the training
    %  set (to avoid data leakage) and evaluates a logistic regression
    %  classifier.  Results (metrics + ROC curve) are printed and plotted.
    %
    %  filename     – path to CSV file
    %  attackName   – string used in printed output and plot title
    %  textColumn   – name of the column that contains raw text
    %----------------------------------------------------------------------

    % 1. Load data --------------------------------------------------------
    raw    = readtable(filename, 'TextType', 'string');
    texts  = raw.(textColumn);
    labels = raw.label;

    % 2. Tokenise ---------------------------------------------------------
    documents = tokenizedDocument(texts);

    % 3. Train / test split ----------------------------------------------
    cv        = cvpartition(labels, 'HoldOut', 0.3);
    docsTrain = documents(training(cv));
    docsTest  = documents(test(cv));
    yTrain    = labels(training(cv));
    yTest     = labels(test(cv));

    % 4. Build vocabulary and TF‑IDF on training set
    bag    = bagOfWords(docsTrain);
    bag    = removeInfrequentWords(bag, 5);
    XTrain = tfidf(bag);                       % TF‑IDF для train
    XTest  = tfidf(bag, docsTest);             % TF‑IDF для test
    
    % 5. Row‑wise L2‑normalisation вручную (не через normalize)
    rowNorms = sqrt(sum(XTrain.^2, 2)) + eps;
    XTrain   = bsxfun(@rdivide, XTrain, rowNorms);
    rowNorms = sqrt(sum(XTest .^2, 2)) + eps;
    XTest    = bsxfun(@rdivide, XTest, rowNorms);

    % 6. Train logistic regression ---------------------------------------
    model        = fitclinear(XTrain, yTrain, 'Learner', 'logistic');
    [yPred, scr] = predict(model, XTest);  % scr(:,2) – score for class 1

    % 7. Metrics ----------------------------------------------------------
    TP = sum((yTest == 1) & (yPred == 1));
    TN = sum((yTest == 0) & (yPred == 0));
    FP = sum((yTest == 0) & (yPred == 1));
    FN = sum((yTest == 1) & (yPred == 0));

    accuracy  = (TP + TN) / numel(yTest);
    precision = TP / (TP + FP + eps);
    recall    = TP / (TP + FN + eps);
    f1_score  = 2 * (precision * recall) / (precision + recall + eps);

    fprintf('Атака: %s\n', attackName);
    fprintf('Accuracy:  %.4f\n', accuracy);
    fprintf('Precision: %.4f\n', precision);
    fprintf('Recall:    %.4f\n', recall);
    fprintf('F1-score:  %.4f\n\n', f1_score);

    % 8. ROC curve --------------------------------------------------------
    [rocX, rocY, ~, AUC] = perfcurve(yTest, scr(:,2), 1);
    figure;
    plot(rocX, rocY, 'LineWidth', 2);
    title(sprintf('ROC кривая (%s, AUC = %.4f)', attackName, AUC));
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    grid on;
end

%-----------------------------------------------------------------------
%  Multiple datasets evaluation
%-----------------------------------------------------------------------
evaluate_logistic_regression('datasets/dos_dataset.csv',           'DoS Attack',    'request');
evaluate_logistic_regression('datasets/sql_injection_dataset.csv', 'SQL Injection', 'query');
evaluate_logistic_regression('datasets/xss_dataset.csv',           'XSS Attack',    'payload');
evaluate_logistic_regression('datasets/path_traversal_dataset.csv','Path Traversal','path');
