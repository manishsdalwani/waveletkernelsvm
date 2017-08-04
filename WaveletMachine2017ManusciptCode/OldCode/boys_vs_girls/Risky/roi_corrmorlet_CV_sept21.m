% ROI SVM
% OLD STRIATUM (1124)
% BOYS VS GIRLS
% RISKY

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

% set up cross validation
crossValidationFolds = 3;
numberOfRowsPerFold = dataRowNumber / crossValidationFolds;
crossValidationTrainData = [];
crossValidationTestData = [];
AccuracyData_RBF=[];
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

numTrain = size(SortedData(trainRows ,3:1126) ,1);
numTest = size(SortedData(testRows ,3:1126) ,1);

 bestcv = 0;
for log2c = -3.1:3.1,
  for log2g = -20.1:20.1,
      sigma=2^log2g;
      rbfKernel = @(X,Y) cos(sqrt(sigma).* pdist2(X,Y,'euclidean')).*exp(-sigma .* pdist2(X,Y,'euclidean').^2)-exp(-0.5).*exp(-sigma .* pdist2(X,Y,'euclidean').^2);     
      K =  [ (1:numTrain)', rbfKernel(SortedData(trainRows ,3:1126),SortedData(trainRows ,3:1126))];
     cmd = ['-t 4 -v 3 -c ', num2str(2^log2c)];
     cv = svmtrain(SortedData(trainRows,2),K,cmd);
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c; bestsigma = 2^log2g;
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestsigma, bestcv);
  end
end

rbfKernel2 = @(X,Y) cos(sqrt(bestsigma).* pdist2(X,Y,'euclidean')).*exp(-bestsigma .* pdist2(X,Y,'euclidean').^2)-exp(-0.5).*exp(-sigma .* pdist2(X,Y,'euclidean').^2);
K2 =  [ (1:numTrain)', rbfKernel2(SortedData(trainRows ,3:1126),SortedData(trainRows ,3:1126))];
KK = [ (1:numTest)', rbfKernel2(SortedData(testRows,3:1126),SortedData(trainRows ,3:1126))];

%# train and test
cmd = ['-t 4 -c ', num2str(bestc)];% fixed bestc
model = svmtrain(SortedData(trainRows,2), K2, cmd);
[predClass, acc, decVals] = svmpredict(SortedData(testRows ,2), KK, model);
 AccuracyData_RBF = [AccuracyData_RBF, acc(1,1)]

 %# confusion matrix
C = confusionmat(SortedData(testRows,2),predClass)
correct_boys=[correct_boys,C(1,1)];
incorrect_boys=[incorrect_boys,C(1,2)];
correct_girls=[correct_girls,C(2,2)];
incorrect_girls=[incorrect_girls,C(2,1)];
%precision = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);
%recall = @(confusionMat) diag(confusionMat)./sum(confusionMat,1)';
%f1Scores = @(confusionMat) 2*(precision(confusionMat).*recall(confusionMat))./(precision(confusionMat)+recall(confusionMat));
%meanF1 = @(confusionMat) mean(f1Scores(confusionMat));
end
%# Accuracy, %classified correctly

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
