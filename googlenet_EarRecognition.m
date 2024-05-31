Dataset = imageDatastore('images', 'IncludeSubfolders', true, 'LabelSource', 'foldernames')
im.ReadFcn = @(loc)imresize(imread(loc),[224,224]);
[Train ,Test] = splitEachLabel(im,0.80, "randomized");

net = googlenet;

Input_Layer_Size = net.Layers(1).InputSize

Layer_Graph = layerGraph(net)

Feature_Learner = net.Layers(142)
Output_Classifier = net.Layers(144)

Number_of_Classes = numel(categories(Train.Labels))

New_Feature_Learner = fullyConnectedLayer(Number_of_Classes, 'Name', 'Ear Feature Learner', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)

New_Classifier_Layer = classificationLayer('Name', 'Ear Classifier')

Layer_Graph = replaceLayer(Layer_Graph, Feature_Learner.Name, New_Feature_Learner)

Layer_Graph = replaceLayer(Layer_Graph, Output_Classifier.Name, New_Classifier_Layer)


Size_of_MiniBatch = 5
Validation_Frequency = floor(numel(Train.Files)/Size_of_MiniBatch)
Training_Options = trainingOptions('sgdm', 'MiniBatchSize', Size_of_MiniBatch, 'MaxEpochs', 15, 'InitialLearnRate', 0.0001, 'Shuffle', 'every-epoch', 'ValidationData', Test, 'ValidationFrequency', Validation_Frequency, 'Verbose', true)

newnet = trainNetwork(Train, Layer_Graph, Training_Options);
[predict,scores] = classify(newnet, Test);
names = Test.Labels;
pred = (predict==names);
s = size(pred);
acc = sum(pred)/s(1);


Gen = diag(scores);
Imp = scores(~(eye(size(scores))));
scores1 = [Gen; Imp];
labels = [ones(length(Gen), 1); zeros(length(Imp), 1)];

% Compute ROC
[X, Y, T, AUC] = perfcurve(labels, cast(scores1, "double"), 1);
FMR = X;
TMR = 1 - Y;
threshold = T;

% EER
EER = X(find(abs(X - (1 - Y)) == min(abs(X - (1 - Y))), 1));

% d-prime calculation
meanG = mean(Gen);
meanI = mean(Imp);
stdG = std(Gen);
stdI = std(Imp);
d_prime = abs(meanG - meanI) / sqrt(0.5 * (stdG^2 + stdI^2));

% Find TMR at FMR = 1%
idx = find(X <= 0.01, 1, 'last');
TMR_at_FMR_1_percent = Y(idx);


% Find TMR at FMR = 0.01%
idx = find(X <= 0.0001, 1, 'last');
TMR_at_FMR_0point01_percent = Y(idx);

row = s(1)
range1 = 0:0.02:1;
Gen_hist = histc(Gen,range1);
Imp_hist = histc(Imp,range1);
Gen_pdf = Gen_hist ./row ;
Imp_pdf = Imp_hist ./ (row*row - row);
figure(1),
plot(range1,Gen_pdf,'-r','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10)
hold on
plot(range1,Imp_pdf,'-k','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',10)
xlabel('Match Score');
ylabel('Probability');
title('GENUINE AND IMPOSTER SCORE DISTRIBUTION');
legend('Genuine distribution', 'Imposter Distribution');
hold off

figure(2),
plot(X,Y);
xlabel('False Match Rate');
ylabel('True match Rate');
title('ROC curve');

fprintf('d-prime: %f\n', d_prime);
fprintf('Equal Error Rate (EER): %f\n', EER);
fprintf('TMR at FMR 1%%: %f\n', TMR_at_FMR_1_percent);
fprintf('TMR at FMR 0.01%%: %f\n', TMR_at_FMR_0point01_percent);
fprintf('Identification Rate: %f%%\n', acc*100);
