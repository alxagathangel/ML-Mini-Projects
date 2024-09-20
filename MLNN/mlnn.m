clear;clc;

load iris_dataset
inputs = iris_dataset;
targets = irisTargets;
setdemorandstream(270);

    % NN με 1 Layer
hiddenLayerSize1 = 10;
net1 = patternnet(hiddenLayerSize1);

% Διαχωρισμός δεδομένων
net1.divideParam.trainRatio = 0.7;
net1.divideParam.valRatio = 0.15;
net1.divideParam.testRatio = 0.15;

% Εκπαίδευση δικτύου
[net1, tr1] = train(net1, inputs, targets);

% Εκτίμηση απόδοσης
outputs1 = net1(inputs);
errors1 = gsubtract(targets, outputs1);
performance1 = perform(net1, targets, outputs1);

% Confusion matrix
[c1, cm1] = confusion(targets, outputs1);
fprintf('Confusion Matrix for 1 Layer:\n');
disp(cm1);

% ROC καμπύλη
plotroc(targets, outputs1);
title('ROC Curve for 1 Layer');

    % ΝN με 2 Layers
hiddenLayerSize2 = [10, 5];
net2 = patternnet(hiddenLayerSize2);

% Διαχωρισμός δεδομένων
net2.divideParam.trainRatio = 0.7;
net2.divideParam.valRatio = 0.15;
net2.divideParam.testRatio = 0.15;

% Εκπαίδευση δικτύου
[net2, tr2] = train(net2, inputs, targets);

% Εκτίμηση απόδοσης
outputs2 = net2(inputs);
errors2 = gsubtract(targets, outputs2);
performance2 = perform(net2, targets, outputs2);

% Confusion matrix
[c2, cm2] = confusion(targets, outputs2);
fprintf('Confusion Matrix for 2 Layers:\n');
disp(cm2);

% ROC καμπύλη
figure;
plotroc(targets, outputs2);
title('ROC Curve for 2 Layers');

    % ΝN με 5 Κρυφούς Νευρώνες
hiddenLayerSize3 = 5;
net3 = patternnet(hiddenLayerSize3);

% Διαχωρισμός δεδομένων
net3.divideParam.trainRatio = 0.7;
net3.divideParam.valRatio = 0.15;
net3.divideParam.testRatio = 0.15;

% Εκπαίδευση δικτύου
[net3, tr3] = train(net3, inputs, targets);

% Εκτίμηση απόδοσης
outputs3 = net3(inputs);
errors3 = gsubtract(targets, outputs3);
performance3 = perform(net3, targets, outputs3);

% Confusion matrix
[c3, cm3] = confusion(targets, outputs3);
fprintf('Confusion Matrix for 5 Neurons:\n');
disp(cm3);

% ROC καμπύλη
figure;
plotroc(targets, outputs3);
title('ROC Curve for 5 Neurons');

    % NN με 15 Κρυφούς Νευρώνες
hiddenLayerSize4 = 15;
net4 = patternnet(hiddenLayerSize4);

% Διαχωρισμός δεδομένων
net4.divideParam.trainRatio = 0.7;
net4.divideParam.valRatio = 0.15;
net4.divideParam.testRatio = 0.15;

% Εκπαίδευση δικτύου
[net4, tr4] = train(net4, inputs, targets);

% Εκτίμηση απόδοσης
outputs4 = net4(inputs);
errors4 = gsubtract(targets, outputs4);
performance4 = perform(net4, targets, outputs4);

% Confusion matrix
[c4, cm4] = confusion(targets, outputs4);
fprintf('Confusion Matrix for 15 Neurons:\n');
disp(cm4);

% ROC καμπύλη
figure;
plotroc(targets, outputs4);
title('ROC Curve for 15 Neurons');

    % NN με ποσοστά 60/20/20
hiddenLayerSize5 = 10;
net5 = patternnet(hiddenLayerSize5);

% Διαχωρισμός δεδομένων
net5.divideParam.trainRatio = 0.6;
net5.divideParam.valRatio = 0.2;
net5.divideParam.testRatio = 0.2;

% Εκπαίδευση δικτύου
[net5, tr5] = train(net5, inputs, targets);

% Εκτίμηση απόδοσης
outputs5 = net5(inputs);
errors5 = gsubtract(targets, outputs5);
performance5 = perform(net5, targets, outputs5);

% Confusion matrix
[c5, cm5] = confusion(targets, outputs5);
fprintf('Confusion Matrix for 60/20/20 split:\n');
disp(cm5);

% ROC καμπύλη
figure;
plotroc(targets, outputs5);
title('ROC Curve for 60/20/20 split');

    % NN με ποσοστά 80/10/10
hiddenLayerSize6 = 10;
net6 = patternnet(hiddenLayerSize6);

% Διαχωρισμός δεδομένων
net6.divideParam.trainRatio = 0.8;
net6.divideParam.valRatio = 0.1;
net6.divideParam.testRatio = 0.1;

% Εκπαίδευση δικτύου
[net6, tr6] = train(net6, inputs, targets);

% Εκτίμηση απόδοσης
outputs6 = net6(inputs);
errors6 = gsubtract(targets, outputs6);
performance6 = perform(net6, targets, outputs6);

% Confusion matrix
[c6, cm6] = confusion(targets, outputs6);
fprintf('Confusion Matrix for 80/10/10 split:\n');
disp(cm6);

% ROC καμπύλη
figure;
plotroc(targets, outputs6);
title('ROC Curve for 80/10/10 split');