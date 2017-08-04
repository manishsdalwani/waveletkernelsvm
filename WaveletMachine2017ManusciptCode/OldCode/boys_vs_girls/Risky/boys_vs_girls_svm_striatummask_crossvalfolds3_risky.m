
% Region of Interest SVM
% ORIGINAL STRIATUM POST NAN's is 1124 voxel
% Risky
% Boys vs Girls

clear all;
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
crossValidationFolds = 27;
numberOfRowsPerFold = dataRowNumber / crossValidationFolds;

crossValidationTrainData = [];
crossValidationTestData = [];
AccuracyData_L=[];
AccuracyData_RBF = [];
for i = 1:3
    if (i == 1)
        trainRows = SortedData(1:54, 1:1126);
        testRows = SortedData(55:81, 1:1126);
    elseif (i == 2)
        trainRows = vertcat(SortedData(1:27, 1:1126), SortedData(55:81, 1:1126));
        testRows = SortedData(28:54, 1:1126);
    else 
        trainRows = SortedData(28:81, 1:1126);
        testRows = SortedData(1:27, 1:1126);        
    end
    model_linear = svmtrain(trainRows(:,2), trainRows(: ,3:1126), '-t 0');
    [predict_label_L, accuracy_L, dec_values_L] = svmpredict(testRows(:,2), testRows(: ,3:1126), model_linear);
    AccuracyData_L=[AccuracyData_L, accuracy_L(1,1)]
    model_RBF = svmtrain(trainRows(:,2), trainRows(: ,3:1126), '-t 2');
    [predict_label_RBF, accuracy_RBF, dec_values_RBF] = svmpredict(testRows(:,2), testRows(: ,3:1126), model_RBF);
    AccuracyData_RBF=[AccuracyData_RBF, accuracy_RBF(1,1)]
    crossValidationTrainData = [crossValidationTrainData ; trainRows];
    crossValidationTestData = [crossValidationTestData ;testRows];

end
mean_Accuracy_L = mean(AccuracyData_L)
mean_Accuracy_RBF = mean(AccuracyData_RBF)
