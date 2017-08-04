% ROI SVM
% LINEAR AND RBF
% BOYS VS GIRLS
% RISKY
% STRIATUM (1124)

clear;
clc;
basedir = ('/Users/manishdalwani/Desktop/Wavelet_Machine/cts_decout_ar_riskyflip/');
cd(basedir)
subject_list = {'5014', '5017', '5021', '5022', '5024', '5029', '5030', '5034', '5063', '5068', '5070', '5072', '5073', '5074', '5075', '5076', '5079', '5081', '5082', '5083', '5016', '5025', '5032', '5037', '5099', '5100', '5104', '5106', '5108', '5163', '5174', '5175', '5176', '5190', '5194', '5197', '5203', '5214', '5215', '5216', '5011', '5013', '5018', '5033', '5044', '5047', '5049', '5050', '5051', '5052', '5054', '5056', '5057', '5061', '5062', '5067', '5084', '5085', '5087', '5088', '5009', '5010', '5055', '5066', '5090', '5092', '5093', '5094', '5096', '5097', '5116', '5119', '5134', '5136', '5149', '5155', '5157', '5159', '5171', '5182', '5187'  };
numsubjects = length(subject_list);

parentfolder = '/Users/manishdalwani/Desktop/Wavelet_Machine/cts_decout_ar_riskyflip/';

mask_hdr = spm_vol('striatum-structural-2mm.nii'); % load mask and convert to logical indexes w/ SPM tools
[mask_vol, mask_xyz] = spm_read_vols(mask_hdr);
mask_vol = logical(mask_vol);    % create binary mask of T/F values

ALLsubs=zeros(20,2847);
%ALLsubs2=zeros(20,2847);
CURRENTsub = 0;

for s=1:numsubjects
    subject = subject_list{s};
    fprintf('\n\n\n*** Processing subject %d (%s) ***\n\n\n', s, subject);
    subjfmri = ['con_0002_' num2str(subject) '.img'];
    fmri_hdr = spm_vol(subjfmri);                     % load fMRI volume
    [fmri_vol, fmri_xyz] = spm_read_vols(fmri_hdr);
    fmri_masked_vector = fmri_vol(mask_vol);  % Apply mask from above, resulting vector will only contain voxels included in the mask
    CURRENTsub = CURRENTsub + 1;
    ALLsubs(CURRENTsub,:) = reshape(fmri_masked_vector, 1, 2847);
end;

X = [ones(20, 1), ALLsubs(1:20,:)]; % Add a column of ones to Allsubs
Y = [-(ones(20, 1)), ALLsubs(21:40,:)]; % Add a column of ones to Allsubs
X2 = [ones(20, 1), ALLsubs(41:60,:)]; % Add a column of ones to Allsubs
Y2 = [-(ones(21, 1)), ALLsubs(61:81,:)]; % Add a column of ones to Allsubs
Z = vertcat(X,Y,X2,Y2);
NaNCols = any(isnan(Z));
Z = Z(:,~NaNCols); 
dataRowNumber = size(Z,1);
rng(2);
randomColumn = rand(dataRowNumber,1);
Z2 = [randomColumn Z];
SortedData  = sortrows(Z2,1);
%[values, order] = sort(Z(:,1));
%SortedData = Z2(order,:);
 

% Split Data
%train_data = vertcat(Z(1:14,2:190002),Z(21:34,2:190002),Z(41:54,2:190002),Z(61:75,2:190002));
%train_label = vertcat(Z(1:14,1),Z(21:34,1),Z(41:54,1),Z(61:75,1));
%test_data = vertcat(Z(15:20,2:190002),Z(35:40,2:190002),Z(55:60,2:190002),Z(76:81,2:190002));
%test_label = vertcat(Z(15:20,1),Z(35:40,1),Z(55:60,1),Z(76:81,1));
crossValidationFolds = 3;
numberOfRowsPerFold = dataRowNumber / crossValidationFolds;

crossValidationTrainData = [];
crossValidationTestData = [];
AccuracyData_L=[];
AccuracyData_RBF = [];
correct_boys=[];
incorrect_boys=[];
correct_girls=[];
incorrect_girls=[];
for startOfRow = 1:numberOfRowsPerFold:dataRowNumber
    testRows = startOfRow:startOfRow+numberOfRowsPerFold-1;
    if (startOfRow == 1)
        trainRows = [testRows+1:dataRowNumber];
        else
        trainRows = [1:startOfRow-1 max(testRows)+1:dataRowNumber];
    end
    
    bestcv = 0;
for log2c = -3.1:3.1,
  for log2g = -20.1:20.1,
    cmd = ['-t 2 -v 3 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    cv = svmtrain(SortedData(trainRows ,2), SortedData(trainRows ,3:1126), cmd);
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
      fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
    end
  end
end
    
    % After finding the best parameter value for C, we train the entire data
% again using this parameter value
    cmd = ['-t 0 -c ', num2str(bestc), ' -g ', num2str(bestg)];
    model_linear = svmtrain(SortedData(trainRows ,2), SortedData(trainRows ,3:1126), cmd);
    w_L = (model_linear.sv_coef' * full(model_linear.SVs));
    bias_L = -model_linear.rho;
    [predict_label_L, accuracy_L, dec_values_L] = svmpredict(SortedData(testRows ,2), SortedData(testRows ,3:1126), model_linear);
    AccuracyData_L = [AccuracyData_L, accuracy_L(1,1)]
    cmd = ['-t 2 -c ', num2str(bestc), ' -g ', num2str(bestg)];
    model_RBF = svmtrain(SortedData(trainRows ,2), SortedData(trainRows ,3:1126), cmd);
    [predict_label_RBF, accuracy_RBF, dec_values_RBF] = svmpredict(SortedData(testRows ,2), SortedData(testRows ,3:1126), model_RBF);
    AccuracyData_RBF = [AccuracyData_RBF, accuracy_RBF(1,1)]
    crossValidationTrainData = [crossValidationTrainData ; SortedData(trainRows ,:)];
    crossValidationTestData = [crossValidationTestData ;SortedData(testRows ,:)];

    C = confusionmat(SortedData(testRows,2),predict_label_RBF)
    correct_boys=[correct_boys,C(1,1)];
    incorrect_boys=[incorrect_boys,C(1,2)];
    correct_girls=[correct_girls,C(2,2)];
     incorrect_girls=[incorrect_girls,C(2,1)];
end
mean_Accuracy_L = mean(AccuracyData_L)
mean_Accuracy_RBF = mean(AccuracyData_RBF)

mean_correct_boys=mean(correct_boys)
mean_incorrect_boys=mean(incorrect_boys)
mean_correct_girls=mean(correct_girls)
mean_incorrect_girls=mean(incorrect_girls)
per_corr_boys=mean_correct_boys/(mean_correct_boys+mean_incorrect_boys) % percent correct boys
per_corr_girls=mean_correct_girls/(mean_correct_girls+mean_incorrect_girls)% percent correct girls
bal_accuracy_RBF=(per_corr_boys+per_corr_girls)/2 % Balanced accuracy
precision_boys=mean_correct_boys/(mean_correct_boys+mean_incorrect_boys)
precision_girls=mean_correct_girls/(mean_correct_girls+mean_incorrect_girls)

sensitivity_boys=mean_correct_boys/(mean_correct_boys+mean_incorrect_girls)
specificity_boys=mean_correct_girls/(mean_incorrect_boys+mean_correct_girls)
sensitivity_girls=mean_correct_girls/(mean_correct_girls+mean_incorrect_boys)
specificity_girls=mean_correct_boys/(mean_incorrect_girls+mean_correct_boys)
Fscore_boys=2*mean_correct_boys/(2*mean_correct_boys+mean_incorrect_girls+mean_incorrect_boys)
Fscore_girls=2*mean_correct_girls/(2*mean_correct_girls+mean_incorrect_boys+mean_incorrect_girls)
Recall_boys=mean_correct_boys/(mean_correct_boys+mean_incorrect_girls)
Recall_girls=mean_correct_girls/(mean_correct_girls+mean_incorrect_boys)
random_Accuracy_RBF = ((mean_correct_girls+mean_incorrect_boys)*(mean_correct_girls+mean_incorrect_girls)+(mean_incorrect_girls+mean_correct_boys)*(mean_incorrect_boys+mean_correct_boys))/(mean_correct_boys+mean_correct_girls+mean_incorrect_boys+mean_incorrect_girls)^2

mean_Accuracy_RBF = mean(AccuracyData_RBF) % Overall accuracy
kappa=(mean_Accuracy_RBF/100-random_Accuracy_RBF)/(1-random_Accuracy_RBF)
%mean_abs_err=
TP_weighted=(mean_correct_boys+mean_correct_girls)/2
FP_weighted=(mean_incorrect_boys+mean_incorrect_girls)/2 %F score is the harmonic mean of precision and sensitivity
Precision_weighted=(precision_boys+precision_girls)/2
Recall_weighted=(Recall_boys+Recall_girls)/2
Fscore_weighted=(Fscore_boys+Fscore_girls)/2
%ROC=((sensitivity_boys+sensitivity_girls)/2)/(1-((specificity_boys+specificity_girls)/2))
%TPR=mean


