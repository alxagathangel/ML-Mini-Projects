clear;clc;

load iris_dataset
inputs = iris_dataset';
targets = vec2ind(irisTargets)';

% Διαχωρισμός δεδομένων σε training κ test sets
cv = cvpartition(targets, 'HoldOut', 0.3);
X_train = inputs(training(cv), :);
Y_train = targets(training(cv), :);
X_test = inputs(test(cv), :);
Y_test = targets(test(cv), :);

% Εκπαίδευση SVM με fitcecoc για πολυκατηγορική ταξινόμηση
template = templateSVM('KernelFunction', 'rbf', 'Standardize', true);
SVMModel = fitcecoc(X_train, Y_train, 'Learners', template);

% Πρόβλεψη στο test set
[predictions, scores] = predict(SVMModel, X_test);

% Accuracy
accuracy = sum(predictions == Y_test) / length(Y_test);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Confusion Matrix
confMat = confusionmat(Y_test, predictions);
disp('Confusion Matrix:');
disp(confMat);

% ROC Curves
figure;
hold on;
for i = 1:3
    [Xroc, Yroc, T, AUC] = perfcurve(Y_test == i, scores(:, i), true);
    plot(Xroc, Yroc);
    title('ROC Curve for each class');
    xlabel('False positive rate'); 
    ylabel('True positive rate');
    legend({'Class 1', 'Class 2', 'Class 3'});
end
hold off;
