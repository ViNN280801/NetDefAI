function evaluate_svm_sqli(filename, attackName, textColumn)
    %----------------------------------------------------------------------
    %  Reads a CSV file with texts and binary labels, splits data into
    %  train/test, builds TF-IDF representation and evaluates SVM classifier
    %  Results (metrics) are printed (ROC curve is commented out)
    %
    %  filename     – path to CSV file
    %  attackName   – string used in printed output
    %  textColumn   – name of the column that contains raw text
    %----------------------------------------------------------------------

    % 1. Load data --------------------------------------------------------
    fprintf('Loading data from %s...\n', filename);
    raw = readtable(filename, 'TextType', 'string');
    texts = raw.(textColumn);
    labels = raw.Label;

    % 2. Tokenize ---------------------------------------------------------
    fprintf('Tokenizing documents...\n');
    documents = tokenizedDocument(texts);

    % 3. Train/test split ------------------------------------------------
    fprintf('Splitting data (70%% train, 30%% test)...\n');
    cv = cvpartition(labels, 'HoldOut', 0.3);
    docsTrain = documents(training(cv));
    docsTest = documents(test(cv));
    yTrain = labels(training(cv));
    yTest = labels(test(cv));

    % 4. Build vocabulary and TF-IDF on training set ---------------------
    fprintf('Building bag-of-words and TF-IDF...\n');
    bag = bagOfWords(docsTrain);
    bag = removeInfrequentWords(bag, 100);

    % Convert to full matrices (not sparse) for SVM
    XTrain = full(tfidf(bag, docsTrain));
    XTest = full(tfidf(bag, docsTest));

    % 5. Row-wise L2-normalization ---------------------------------------
    fprintf('Normalizing features...\n');
    rowNorms = sqrt(sum(XTrain .^ 2, 2)) + eps;
    XTrain = bsxfun(@rdivide, XTrain, rowNorms);
    rowNorms = sqrt(sum(XTest .^ 2, 2)) + eps;
    XTest = bsxfun(@rdivide, XTest, rowNorms);

    % 6. Train SVM classifier --------------------------------------------
    fprintf('Training SVM model (this may take time)...\n');
    model = fitclinear(XTrain, yTrain, 'Learner', 'svm', 'Solver', 'lbfgs');
    [yPred, score] = predict(model, XTest);

    % 7. Metrics ---------------------------------------------------------
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

    % 8. ROC curve -----------------------------------------------------
    % [rocX, rocY, ~, AUC] = perfcurve(yTest, score(:, model.ClassNames == 1), 1);
    % figure('Name', sprintf('SVM ROC - %s', attackName));
    % plot(rocX, rocY, 'LineWidth', 2);
    % title(sprintf('ROC кривая (%s, AUC = %.4f)', attackName, AUC));
    % xlabel('False Positive Rate');
    % ylabel('True Positive Rate');
    % grid on;

    % 10. Lightweight 2D visualization -----------------------------------
    try
        fprintf('Creating lightweight 2D visualization...\n');

        % Combine train and test for consistent scaling
        XCombined = [XTrain; XTest];
        yCombined = [yTrain; yTest];

        % Sample max 10,000 points randomly
        num_points = min(10000, size(XCombined, 1));
        idx = randperm(size(XCombined, 1), num_points);
        XSampled = XCombined(idx, :);
        ySampled = yCombined(idx);

        % Reduce to 2D using PCA on sampled data
        [~, scorePCA] = pca(XSampled, 'NumComponents', 2);

        % Plot
        figure('Name', sprintf('SVM 2D Projection - %s', attackName));
        scatter(scorePCA(ySampled == 0, 1), scorePCA(ySampled == 0, 2), 10, 'b', '.');
        hold on;
        scatter(scorePCA(ySampled == 1, 1), scorePCA(ySampled == 1, 2), 10, 'r', '.');

        title(sprintf('2D Projection (%s)\n(Sampled %d points)', attackName, num_points));
        xlabel('PCA 1');
        ylabel('PCA 2');
        legend({'Normal', 'Attack'}, 'Location', 'best');
        grid on;
        hold off;
    catch ME
        fprintf('Lightweight visualization failed: %s\n', ME.message);
    end

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

evaluate_svm_sqli(dataset_path, 'SQL Injection', 'Query');
