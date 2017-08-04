% ROI MultiScaleMexicanHat Kingsbury SVM
% Risky CHOICES
% controls VS patients
% Added confusion matrix plot
% Added AUC and ROC
% Note that this program has few dependencies including SPM for spm_read_vols and spm_read_vols
% Neural Network Toolbox for MeanAbsoluteError mae
% Statistical and Machine Learning Toolbox for perfcurve and confusionmat
% libsvm for svmtrain and svmpredict
% Includes loop for all frontal cortex rois

% Clear Workspace
clear all;
% Clear command Line
clc;

% Base directory modify as per your own
basedir = ('/Users/manishdalwani/Desktop/Wavelet_Machine/cts_decout_ar_riskyflip/');
cd(basedir)

roi_list = {'01', '02', '03', '04', '05', '06', '07', '08', '09', '10'};

% List of subjects. Note I save my contrast images as con_0001_$subjectid.img & con_0001_$subjectid.hdr or
% con_0001_$subjectid.nii.
subject_list = {'5014', '5017', '5021', '5022', '5024', '5029', '5030', '5034', '5063', '5068', '5070', '5072', '5073', '5074', '5075', '5076', '5079', '5081', '5082', '5083', '5016', '5025', '5032', '5037', '5099', '5100', '5104', '5106', '5108', '5163', '5174', '5175', '5176', '5190', '5194', '5197', '5203', '5214', '5215', '5216', '5011', '5013', '5018', '5033', '5044', '5047', '5049', '5050', '5051', '5052', '5054', '5056', '5057', '5061', '5062', '5067', '5084', '5085', '5087', '5088', '5009', '5010', '5055', '5066', '5090', '5092', '5093', '5094', '5096', '5097', '5116', '5119', '5134', '5136', '5149', '5155', '5157', '5159', '5171', '5182', '5187'  };

numsubjects = length(subject_list);
numROIs = length(roi_list);
for roi=1:numROIs
rois = roi_list{roi}; % roi_list allows picking one roi at a time
roifmri = ['rfDorsalFrontal_thr25_2mm_' num2str(rois) '.nii']; %specific roi
% ROI mask upload. You can pick your own ROI. Note this particular section has SPM dependency
mask_hdr = spm_vol(roifmri); % load mask and convert to logical indexes w/ SPM tools
% spm_read_vols can helps us read the image and store coordinate information
[mask_vol, mask_xyz] = spm_read_vols(mask_hdr);
%logical function helps binarize the mask
mask_vol = logical(mask_vol);    % create binary mask of T/F values

% ALLsubs is initialization of the feature matrix
ALLsubs=[];

% Initialize the current subject (acts as a serial counter).
CURRENTsub = 0;

%setting up the for loop to upload the contrast maps and setting up the feature matrix
for s=1:numsubjects %numsubject is the total number of subjects
    subject = subject_list{s}; % subject_list allows picking one subject at a time
    fprintf('\n\n\n*** Processing subject %d (%s) ***\n\n\n', s, subject); %notify user about the progress
    subjfmri = ['con_0003_' num2str(subject) '.img']; %specific contrast images
    fmri_hdr = spm_vol(subjfmri);                     % load fMRI volume
    [fmri_vol, fmri_xyz] = spm_read_vols(fmri_hdr);
    fmri_masked_vector = fmri_vol(mask_vol);  % Apply mask from above, resulting vector will only contain voxels included in the mask
    features_org = size(fmri_masked_vector); %dimensions of the mask volume
    CURRENTsub = CURRENTsub + 1;
    ALLsubs(CURRENTsub,:) = reshape(fmri_masked_vector, 1, features_org(1));
end;

%Feauture normalization
[ALLsubs mu sigma] = featureNormalize(ALLsubs);

% controls vs patients: Setting up the labels
X = [ones(40, 1), ALLsubs(1:40,:)]; % Add a column of ones to Allsubs
Y = [-(ones(41, 1)), ALLsubs(41:81,:)]; % Add a column of ones to Allsubs
Z = vertcat(X,Y);

% Nans were dropped from the images. Although it could be replaced with an insigificant number in
% to save the coordinate information
NaNCols = any(isnan(Z));
Z = Z(:,~NaNCols); % Z Matrix without Nans
dataRowNumber = size(Z,1); %obtaining row numbers
rng(2); % insuring a fixed seed value for consist results
randomColumn = rand(dataRowNumber,1); % randomization column to mix the rows eventually
Z2 = [randomColumn Z]; % add the randomization column to the Z matrix
SortedData  = sortrows(Z2,1); % Sort the data as per the randomColumn
features=size(SortedData); % features dimensions

% set up cross validation
crossValidationFolds = 9;
% Number of rows per fold e.g. for 81 subjects a 9 loop CV means each fold has 9 rows
numberOfRowsPerFold = dataRowNumber / crossValidationFolds;

% Not using the next two matrices but it could be used to store the training and test data for investigation
crossValidationTrainData = [];
crossValidationTestData = [];

% initialization of various items to be used subsequently
AccuracyData=[];
MeanAbsErr=[];
correct_controls=[];
incorrect_controls=[];
correct_patients=[];
incorrect_patients=[];
cvloop = 0;
fpr_roc_g1=[];
tpr_roc_g1=[];
t_roc_g1=[];
auc_roc_g1=[];
optrocpt_roc_g1=[];
fpr_roc_g2=[];
tpr_roc_g2=[];
t_roc_g2=[];
auc_roc_g2=[];
optrocpt_roc_g2=[];
grammatrix=[];
determinant = [];
lambda = [];
grammatrix2=[];
determinant2=[];
posdef=[];
groundtruth =[];
SVMpred = [];

% Setting up crossvalidation for loop
for startOfRow = 1:numberOfRowsPerFold:dataRowNumber % middle element represents steps
    testRows = startOfRow:startOfRow+numberOfRowsPerFold-1;
    if (startOfRow == 1)
        trainRows = [testRows+1:dataRowNumber];
    else
        trainRows = [1:startOfRow-1 max(testRows)+1:dataRowNumber];
    end

% Obtaining the dimensions of the training and test sets
numTrain = size(SortedData(trainRows ,3:features(2)) ,1);
numTest = size(SortedData(testRows ,3:features(2)) ,1);


% Setting up grid search for tuning hyperparameter
stepSize = 1;
log2c_list = -10.1:stepSize:10.1;
log2g_list = -20:stepSize:20;
log2g2_list = -20:stepSize:20;

numLog2c = length(log2c_list);
numLog2g = length(log2g_list);
numLog2g2 = length(log2g2_list);
cvMatrix = []; %will save the nested crossvalidation performance

% For loop for grid search
bestcv = 0;
for i = 1:numLog2c
        log2c = log2c_list(i);
        for j = 1:numLog2g
            log2g = log2g_list(j);
                for k = 1:numLog2g2
                    log2g2 = log2g2_list(k);
                    sigma=2^log2g;
                    sigma2=2^log2g2

                    % setting up the MultiScale Mexican Hat Wavelet kernel for training data
                     rbfKernel = @(X,Y) (sigma2/(sigma2-sigma)).*exp(-sigma2.*pdist2(X,Y,'euclidean').^2)-(sigma/(sigma-sigma2)).*exp(-sigma.* pdist2(X,Y,'euclidean').^2); %Multiscale Mexican Hat
                    K =  [ (1:numTrain)', rbfKernel(SortedData(trainRows ,3:features(2)),SortedData(trainRows ,3:features(2)))];
                    cmd = ['-t 4 -q -v 3 -c ', num2str(2^log2c)]; % - v option is the nested CV loop
                    cv = svmtrain(SortedData(trainRows,2),K,cmd); %model fit
                    cvMatrix = [cvMatrix, cv]; %store fit for comparison
                    if (cv >= bestcv)
                       bestcv = cv; bestc = 2^log2c; bestsigma = 2^log2g; bestsigma2 = sigma2; % selection of optimal hyperparameter
                    end
                      fprintf('%g %g %g (best c=%g, g=%g, rate=%g, sigma2=%g)\n', log2c, log2g, cv, bestc, bestsigma,bestcv, bestsigma2);
                end
        end
end

% The modified MultiScaleMexicanHat Wavelet Kernel with the tuned parameters
rbfKernel2 = @(X,Y) (bestsigma2/(bestsigma2-bestsigma)).*exp(-bestsigma2.*pdist2(X,Y,'euclidean').^2)-(bestsigma/(bestsigma-bestsigma2)).*exp(-bestsigma .* pdist2(X,Y,'euclidean').^2); %Multiscale Mexican Hat


% setting up the kernel for training data post optimal hyperparameter selection
K2 =  [ (1:numTrain)', rbfKernel2(SortedData(trainRows ,3:features(2)),SortedData(trainRows ,3:features(2)))];
       
%Regularization
grammatrix=K2(:,2:end); % drop the 1st index column
determinant = [determinant, det(grammatrix)]; %check if determinants are positive
lambda = [lambda, abs(min(eig(grammatrix)))]; %obtain the eigen values of the gram matrix and pick minimum
if min(eig(grammatrix))<0 
    grammatrix2 = grammatrix + (abs(min(eig(grammatrix)))+0.0001)*eye(numTrain);% in case of negative values add Regularization parameter
else 
    grammatrix2 = grammatrix;
end
determinant2 = [determinant2, det(grammatrix2)];% check again for negative values
%The CHOL function provides an optional second output argument "p" which is zero if the matrix is
% found to be positive definite. If the input matrix is not positive definite,
%then p will be a positive integer:
[~,p]=chol(grammatrix2);
posdef = [posdef,p];

% Modify K2 with the regularized gram matrix
K2 = [ (1:numTrain)', grammatrix2];

%setting up the kernel for test data
KK = [ (1:numTest)', (SortedData(testRows,3:features(2))*SortedData(trainRows ,3:features(2))')];

%% train and test using libsvm tools
cmd = ['-t 4 -c ', num2str(bestc)]; % picked the best possible trade-off C
model = svmtrain(SortedData(trainRows,2), K2, cmd);
[predClass, acc, decVals] = svmpredict(SortedData(testRows ,2), KK, model);
 
% Saving the accuracy information
AccuracyData = [AccuracyData, acc(1,1)]

% Plot confusion matrix at each of the cv loop
%h = figure;
ground_truth = SortedData(testRows,2); %true test labels
% plotconfusion from neural network toolbox fails if one label is -ve
ground_truth(ground_truth==-1)=0;
       
%SVM prediction performance
predicted = predClass;
% plotconfusion from neural network toolbox fails if one label is -ve
predicted(predicted==-1)=0;
%plotconfusion(ground_truth',predicted');
%saveas(h,sprintf('FIG%d.png',cvloop)); % will create FIG1, FIG2,...
%figure,plotconfusion(SortedData(testRows,2)',predClass');
       
% Storing Test labels for later use to plot the final confusion matrix
groundtruth = [groundtruth ground_truth];
SVMpred = [SVMpred predicted];

       
% Mean Absolute Error using mae function from the neural network Toolbox
MeanAbsErr=[MeanAbsErr, mae(predicted,ground_truth)]
       
% obtain the confusion matrix using confusionmat command from the ML Toolbox
[C,order] = confusionmat(SortedData(testRows,2),predClass) %order informs how to read the confusion matrix
correct_controls=[correct_controls,C(1,1)];
incorrect_controls=[incorrect_controls,C(1,2)];
correct_patients=[correct_patients,C(2,2)];
incorrect_patients=[incorrect_patients,C(2,1)];
       
cvloop = cvloop + 1; %track the loop and this parameter is used later for indexing
       
% The perfcurve helps in obtain parameters such as FPR and TPR at different thresholds and AUC
% Plot ROC at each loop
% Note model.label(2) = -1  is group 1 and model.label(2) = 1 is group 2
[fpr_g1,tpr_g1,t_g1,auc_g1,optrocpt_g1] = perfcurve(SortedData(testRows,2), decVals(:,1)*model.Label(2),model.Label(2));
[fpr_g2,tpr_g2,t_g2,auc_g2,optrocpt_g2] = perfcurve(SortedData(testRows,2), decVals(:,1)*model.Label(1),model.Label(1));
       
%k=figure;
%plot(fpr,tpr);
%xlabel('False positive rate')
%ylabel('True positive rate')
%title('ROC for Classification by SVM')
%saveas(k,sprintf('Figure%d.png',cvloop));
       
% saving the fpr, tpr, t, auc, and optrocpt at each loop for later mean ROC plot
fpr_roc_g1(1:length(fpr_g1),cvloop) = fpr_g1;
tpr_roc_g1(1:length(tpr_g1),cvloop) = tpr_g1;
t_roc_g1(1:length(t_g1),cvloop) = t_g1;
auc_roc_g1 = [auc_roc_g1, auc_g1];
optrocpt_roc_g1=[optrocpt_roc_g1, optrocpt_g1];
       
fpr_roc_g2(1:length(fpr_g2),cvloop) = fpr_g2;
tpr_roc_g2(1:length(tpr_g2),cvloop) = tpr_g2;
t_roc_g2(1:length(t_g2),cvloop) = t_g2;
auc_roc_g2 = [auc_roc_g2, auc_g2];
optrocpt_roc_g2=[optrocpt_roc_g2, optrocpt_g2];

%precision = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);
%recall = @(confusionMat) diag(confusionMat)./sum(confusionMat,1)';
%f1Scores = @(confusionMat) 2*(precision(confusionMat).*recall(confusionMat))./(precision(confusionMat)+recall(confusionMat));
%meanF1 = @(confusionMat) mean(f1Scores(confusionMat));
end
       
       
datasave=[]; % This matrix will capture useful performance parameters
       
% Take mean of the fpr, tpr from each loop and plot ROC
fpr_roc_final(:,1) = mean(fpr_roc_g1(:,1:cvloop),2);
tpr_roc_final(:,1) = mean(tpr_roc_g1(:,1:cvloop),2);
t_roc_final(:,1) = mean(t_roc_g1(:,1:cvloop),2);
fpr_roc_final(:,2) = mean(fpr_roc_g2(:,1:cvloop),2);
tpr_roc_final(:,2) = mean(tpr_roc_g2(:,1:cvloop),2);
t_roc_final(:,2) = mean(t_roc_g2(:,1:cvloop),2);
for k=1:2
h=figure;
plot(fpr_roc_final(:,k),tpr_roc_final(:,k),'linewidth',3);
ax = gca;
ax.LineWidth = 1.5;
ax.TitleFontSizeMultiplier = 1.2;
ax.LabelFontSizeMultiplier = 2;
ax.XLabel.FontWeight = 'Bold';
ax.YLabel.FontWeight = 'Bold';
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for SVM Classification (Kernel="Multiscale MexicanHat") during Cautious Decision')
saveas(h,sprintf('ROI%s_ROC_MultiscaleMexicanHatSVM_cautious_controlspatients_%d.png',rois,k)); % will create FIG1, FIG2,...
end
       
% AUC statistic --useful in ML
auc_roc_final(:,1) = mean(auc_roc_g1);
auc_roc_final(:,2) = mean(auc_roc_g2);
% Optimal ROC point estimation
optrocpc_final(:,1) = mean(optrocpt_roc_g1);
optrocpc_final(:,2) = mean(optrocpt_roc_g2);

% Next section is geared towards measuring performance
mean_correct_controls = mean(correct_controls)
mean_incorrect_controls = mean(incorrect_controls)
mean_correct_patients = mean(correct_patients)
mean_incorrect_patients = mean(incorrect_patients)
per_corr_controls = mean_correct_controls/(mean_correct_controls+mean_incorrect_controls) % percent correct controls
per_corr_patients = mean_correct_patients/(mean_correct_patients+mean_incorrect_patients)% percent correct patients
       
% Balanced Accuracy http://mvpa.blogspot.com/2015/12/balanced-accuracy-what-and-why.html
bal_Accuracy = (per_corr_controls+per_corr_patients)/2 % Balanced accuracy
       
% http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
% https://ifordata.wordpress.com/tag/confusion-matrix/
% What proportion of the predicted classes are correct? These measures are called Precision and Recall.
% precision = tp/(tp+fp)
precision_controls = mean_correct_controls/(mean_correct_controls+mean_incorrect_patients)
precision_patients = mean_correct_patients/(mean_correct_patients+mean_incorrect_controls)

% https://en.wikipedia.org/wiki/Precision_and_recall
% recall = tp/(tp+fn)
Recall_controls = mean_correct_controls/(mean_correct_controls+mean_incorrect_controls)
Recall_patients = mean_correct_patients/(mean_correct_patients+mean_incorrect_patients)

% https://en.wikipedia.org/wiki/Sensitivity_and_specificity
% sensitivity = recall  = tpr = tp/(tp+fn)
sensitivity_controls = mean_correct_controls/(mean_correct_controls+mean_incorrect_controls)
sensitivity_patients = mean_correct_patients/(mean_correct_patients+mean_incorrect_patients)
       
% https://en.wikipedia.org/wiki/Sensitivity_and_specificity
% specificity = tnr = TN/(TN+FP)
specificity_controls = mean_correct_patients/(mean_incorrect_patients+mean_correct_patients)
specificity_patients = mean_correct_controls/(mean_incorrect_controls+mean_correct_controls)
       
% https://wikimedia.org/api/rest_v1/media/math/render/svg/8b64097b6362d28387a0c4650f2fed2bc5ea9fe9
% F1 = 2*TP(2*TP+FP+FN)
% F1 = 2 * (precision x recall)/(precision + recall)
Fscore_controls = 2*mean_correct_controls/(2*mean_correct_controls+mean_incorrect_patients+mean_incorrect_controls)
Fscore_patients = 2*mean_correct_patients/(2*mean_correct_patients+mean_incorrect_controls+mean_incorrect_patients)
       
% FPR = FP/(FP+TN)
FP_rate_controls = mean_incorrect_patients/(mean_incorrect_patients+mean_correct_patients)
FP_rate_patients = mean_incorrect_controls/(mean_incorrect_controls+mean_correct_controls)

% TPR = sensitivity = recall
TP_rate_controls = mean_correct_controls/(mean_correct_controls+mean_incorrect_controls)
TP_rate_patients = mean_correct_patients/(mean_correct_patients+mean_incorrect_patients)

%http://standardwisdom.com/softwarejournal/2011/12/confusion-matrix-another-single-value-metric-kappa-statistic/
% Random Accuracy also known as expected Accuracy is defined as the accuracy
% that any random classifier would be expected to achieve based on the confusion matrix.
random_Accuracy = ((mean_correct_patients+mean_incorrect_patients)*(mean_correct_patients+mean_incorrect_controls)+(mean_incorrect_controls+mean_correct_controls)*(mean_incorrect_patients+mean_correct_controls))/(mean_correct_controls+mean_correct_patients+mean_incorrect_controls+mean_incorrect_patients)^2

%http://standardwisdom.com/softwarejournal/2011/12/confusion-matrix-another-single-value-metric-kappa-statistic/
mean_Accuracy = mean(AccuracyData) % Overall accuracy

% Kappa is also stated as (observed accuracy-expected accuracy)/(1-expected accuracy)
% http://stats.stackexchange.com/questions/82162/kappa-statistic-in-plain-english
kappa=(mean_Accuracy/100-random_Accuracy)/(1-random_Accuracy)
mean_absolute_error=mean(MeanAbsErr) % Mean absolute error

TP_weighted=(TP_rate_controls+TP_rate_patients)/2
FP_weighted=(FP_rate_controls+FP_rate_patients)/2
Precision_weighted=(precision_controls+precision_patients)/2
Recall_weighted=(Recall_controls+Recall_patients)/2
Fscore_weighted=(Fscore_controls+Fscore_patients)/2 %F score is the harmonic mean of precision and sensitivity

Pred_controls = TP_rate_controls*40 % Predicted number of controls in total sample
Pred_patients = TP_rate_patients*41 % Predicted number of patients in total sample

% print data in an organized manner
datasave=[datasave; mean_Accuracy mean_absolute_error kappa TP_weighted FP_weighted Precision_weighted Recall_weighted Fscore_weighted mean(auc_roc_final) round(Pred_controls) round(Pred_patients)];
fprintf(' Accuracy MeanAbsoluteError Kappa TP FP Precision Recall F1score AUC Predcontrols Predpatients\n')
disp(datasave)

% Detailed measures
extradatasave=[];
extradatasave=[extradatasave; 100*bal_Accuracy precision_controls precision_patients Recall_controls Recall_patients  sensitivity_controls sensitivity_patients specificity_controls specificity_patients Fscore_controls Fscore_patients FP_rate_controls FP_rate_patients TP_rate_controls TP_rate_patients];
fprintf(' BalancedAccuracy Precisioncontrols Precisionpatients Recallcontrols Recallpatients  Sensitivitycontrols Sensitivitypatients Specificitycontrols Specificitypatients Fscorecontrols Fscorepatients FPratecontrols FPratepatients TPratecontrols TPratepatients\n')
disp(extradatasave)

% Save the stats in an excel file
% Store the characters in a cell array and then using cell2table we convert
% it to a table format
T = cell2table({'Accuracy', 'MeanAbsoluteError', 'Kappa', 'TP', 'FP', 'Precision', 'Recall', 'F1score', 'AUC', 'Predcontrols', 'Predpatients'; mean_Accuracy, mean_absolute_error, kappa, TP_weighted, FP_weighted, Precision_weighted, Recall_weighted, Fscore_weighted, mean(auc_roc_final), round(Pred_controls), round(Pred_patients)});
T2 = cell2table({'BalancedAccuracy', 'Precisioncontrols', 'Precisionpatients', 'Recallcontrols', 'Recallpatients',  'Sensitivitycontrols', 'Sensitivitypatients', 'Specificitycontrols', 'Specificitypatients', 'Fscorecontrols', 'Fscorepatients', 'FPratecontrols', 'FPratepatients', 'TPratecontrols', 'TPratepatients';100*bal_Accuracy, precision_controls, precision_patients, Recall_controls, Recall_patients, sensitivity_controls, sensitivity_patients, specificity_controls, specificity_patients, Fscore_controls, Fscore_patients, FP_rate_controls, FP_rate_patients, TP_rate_controls, TP_rate_patients});
T3 = cell2table({'BESTC', 'BESTsigma', 'BESTsigma2'; bestc, bestsigma, bestsigma2});
%writetable helps to save the file into an excel file
filename = 'roi_MultiscaleMexicanHat_SVM_controls_vs_patients_cautious.xlsx';
filename = sprintf('%s_%s',rois, filename);
writetable(T, filename,'sheet',1);
writetable(T2, filename,'sheet',2);
writetable(T3, filename,'sheet',3);
       
% Plot the final Confusion Matrix using plotconfusion from Neural Network
% Toolbox
h=figure;
groundtruth = reshape(groundtruth, dataRowNumber, 1);
SVMpred = reshape(SVMpred, dataRowNumber, 1);
plotconfusion(groundtruth',SVMpred');
saveas(h,sprintf('ROI%sConfusion Matrix Multiscale Mexican Hat SVM Cautious Decision controls vs patients.png',rois));
end
