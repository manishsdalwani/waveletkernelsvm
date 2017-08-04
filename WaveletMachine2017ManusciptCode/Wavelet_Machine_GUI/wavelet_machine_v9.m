function varargout = wavelet_machine_v9(varargin)
% WAVELET_MACHINE_V9 MATLAB code for wavelet_machine_v9.fig
%      WAVELET_MACHINE_V9, by itself, creates a new WAVELET_MACHINE_V9 or raises the existing
%      singleton*.
%
%      H = WAVELET_MACHINE_V9 returns the handle to a new WAVELET_MACHINE_V9 or the handle to
%      the existing singleton*.
%
%      WAVELET_MACHINE_V9('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in WAVELET_MACHINE_V9.M with the given input arguments.
%
%      WAVELET_MACHINE_V9('Property','Value',...) creates a new WAVELET_MACHINE_V9 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before wavelet_machine_v9_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to wavelet_machine_v9_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help wavelet_machine_v9

% Last Modified by GUIDE v2.5 15-May-2017 23:01:35

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @wavelet_machine_v9_OpeningFcn, ...
                   'gui_OutputFcn',  @wavelet_machine_v9_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before wavelet_machine_v9 is made visible.
function wavelet_machine_v9_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to wavelet_machine_v9 (see VARARGIN)

% Choose default command line output for wavelet_machine_v9
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes wavelet_machine_v9 wait for user response (see UIRESUME)
% uiwait(handles.figure1);
kernel = {'Linear', 'RBF', 'Morlet', 'Mexican Hat', 'Multiscale Mexican Hat', 'MorletRBF', 'CustomKernel'};

% load the cell array into the listbox (assumed to be named listbox1
set(handles.kernel,'String',kernel);

% --- Outputs from this function are returned to the command line.
function varargout = wavelet_machine_v9_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in normyes.
function normyes_Callback(hObject, eventdata, handles)
% hObject    handle to normyes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of normyes
if (get(hObject,'Value') == get(hObject,'Max'))
   normyes = 1 %normalization turned on
else
  normyes = 0 %normalization turned off
end
assignin('base','normyes',normyes);

% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1


% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in loadimages.
function loadimages_Callback(hObject, eventdata, handles)
% hObject    handle to loadimages (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

directory = uigetdir;
cd(directory);

% [filename, pathname] = uigetfile( ...
% { '*.img',  'Analyze files (*.img)'; ...
%   '*.nii', 'Nifti File (*.nii)';...
%    '*.*',  'All Files (*.*)'}, ...
%    'Pick a file', ...
%    'MultiSelect', 'on');

% NAME = [pathname,filename]; %path and name
% V = spm_vol(NAME);
% assignin('base','V',V); 
    
%  global im
% [path, user_pick] = imgetfile();
% if user_pick
%     msgbox(sprintf('Error'),'Error','Error');
%     return
% end

% --- Executes on button press in loadlabels.
function loadlabels_Callback(hObject, eventdata, handles)
% hObject    handle to loadlabels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename, pathname] = uigetfile( ...
{ '*.txt',  'text file (*.txt)'; ...
   '*.*',  'All Files (*.*)'}, ...
   'Pick a file');
NAME = [pathname,filename]; %path and name
labels = textread(NAME, '%d');
assignin('base','labels',labels); 
normyes = 0; %normalization turned off at default
assignin('base','normyes',normyes);
regyes = 0; %regularization turned off at default
assignin('base','regyes',regyes);
wholebrain = 0; %regularization turned off at default
assignin('base','wholebrain',wholebrain);

% --- Executes on button press in ROI.
function ROI_Callback(hObject, eventdata, handles)
% hObject    handle to ROI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[filename, pathname] = uigetfile( ...
{  '*.nii', 'Nifti File (*.nii)';...
   '*.img',  'Analyze files (*.img)'; ...
   '*.*',  'All Files (*.*)'}, ...
   'Pick a file', ...
   'MultiSelect', 'on');

NAME = [pathname,filename]; %path and name
mask_hdr = spm_vol(NAME);
assignin('base','mask_hdr',mask_hdr); 


function cvloopparms_Callback(hObject, eventdata, handles)
% hObject    handle to cvloopparms (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of cvloopparms as text
%        str2double(get(hObject,'String')) returns contents of cvloopparms as a double
numsubjects = evalin('base','numsubjects');
cvloop_parmselect = str2double(get(handles.cvloopparms,'String'));
if (mod(numsubjects,cvloop_parmselect)==0)
    assignin('base','cvloop_parmselect',cvloop_parmselect);
else
    msgbox(sprintf('Pick a divisible number'),'Error','Error');
    return
end

% --- Executes during object creation, after setting all properties.
function cvloopparms_CreateFcn(hObject, eventdata, handles)
% hObject    handle to cvloopparms (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function cvloopmodel_Callback(hObject, eventdata, handles)
% hObject    handle to cvloopmodel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of cvloopmodel as text
%        str2double(get(hObject,'String')) returns contents of cvloopmodel as a double
numsubjects = evalin('base','numsubjects');
cvloop_modelperf = str2double(get(handles.cvloopmodel,'String'));
if (mod(numsubjects,cvloop_modelperf)==0)
    assignin('base','cvloop_modelperf',cvloop_modelperf);
else
    msgbox(sprintf('Pick a divisible number'),'Error','Error');
    return
end


% --- Executes during object creation, after setting all properties.
function cvloopmodel_CreateFcn(hObject, eventdata, handles)
% hObject    handle to cvloopmodel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in regularizationyes.
function regularizationyes_Callback(hObject, eventdata, handles)
% hObject    handle to regularizationyes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of regularizationyes
if (get(hObject,'Value') == get(hObject,'Max'))
   regyes = 1 %regularization turned on
else
  regyes = 0 %regularization turned off
end
assignin('base','regyes',regyes);

% --- Executes on button press in classify.
function classify_Callback(hObject, eventdata, handles)
% hObject    handle to classify (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cvloop_parmselect=evalin('base','cvloop_parmselect');
cvloop_modelperf=evalin('base','cvloop_modelperf');
subject_list=evalin('base','subject_list');
%Kernel = evalin('base','Kernel');
%Kernel2 = evalin('base','Kernel2');
labels = evalin('base','labels');
wholebrain = evalin('base','wholebrain');
normyes = evalin('base', 'normyes');
regyes = evalin('base', 'regyes');
KernelChoice = evalin('base','KernelChoice');
numsubjects = length(subject_list);
assignin('base','numsubjects',numsubjects);
if (wholebrain == 0)
% ROI mask upload. Note this particular section has SPM dependency
mask_hdr = evalin('base','mask_hdr');
% spm_read_vols can helps us read the image and store coordinate information
[mask_vol, mask_xyz] = spm_read_vols(mask_hdr);
%logical function helps binarize the mask
mask_vol = logical(mask_vol);    % create binary mask of T/F values

%Next few lines is just a practice to get the dimensions of mask to allow
%preallocation
% testsub = 'con_0002_5014.img';
% read_fMRI = spm_vol(testsub);
% [get_fMRI_vol, get_fMRI_xyz] = spm_read_vols(read_fMRI);
% fMRI_masked = get_fMRI_vol(mask_vol);


% ALLsubs is initialization of the feature matrix
%ALLsubs=zeros(numsubjects, size(fMRI_masked,1));
end

% ALLsubs is initialization of the feature matrix
ALLsubs=[];
datasave = zeros(1,11);
extradatasave = zeros(1,16);
finaldatasave = zeros(1,6);
besthyperparameters = zeros(1,1);
% Initialize the current subject (acts as a serial counter).
CURRENTsub = 0;

%setting up the for loop to upload the contrast maps and setting up the feature matrix
for s=1:numsubjects %numsubject is the total number of subjects
    subject = subject_list{s}; % subject_list allows picking one subject at a time
    %fprintf('\n\n\n*** Processing subject %d (%s) ***\n\n\n', s, subject); %notify user about the progress
    subjfmri = ['con_0002_' num2str(subject) '.img']; %specific contrast images
    fmri_hdr = spm_vol(subjfmri);                     % load fMRI volume
    [fmri_vol, fmri_xyz] = spm_read_vols(fmri_hdr);
    CURRENTsub = CURRENTsub + 1;
    features_org = size(fmri_xyz);
    if (wholebrain == 0)
        fmri_masked_vector = fmri_vol(mask_vol);  % Apply mask from above, resulting vector will only contain voxels included in the mask
        features_org = size(fmri_masked_vector); %dimensions of the mask volume
        ALLsubs(CURRENTsub,:) = reshape(fmri_masked_vector, 1, features_org(1));
        
    else 
        ALLsubs(CURRENTsub,:) = reshape(fmri_vol, 1, features_org(2));
    end
end;
assignin('base','ALLsubs_beforenorm',ALLsubs);
if (normyes == 1)
    %Feauture normalization
    [ALLsubs, mu, sigma] = featureNormalize(ALLsubs);
    sprintf('normalizing data')
end
assignin('base','ALLsubs',ALLsubs);
Z = [labels ALLsubs];
assignin('base','Z',Z);

% Nans were dropped from the images. Although it could be replaced with an insigificant number in
% to save the coordinate information
NaNCols = any(isnan(Z));
Z = Z(:,~NaNCols); % Z Matrix without Nans
dataRowNumber = size(Z,1); %obtaining row numbers
rng(2); % insuring a fixed seed value for consist results
randomColumn = rand(dataRowNumber,1); % randomization column to mix the rows eventually
Z2 = [randomColumn Z]; % add the randomization column to the Z matrix
SortedData  = sortrows(Z2,1); % Sort the data as per the randomColumn
assignin('base','SortedData',SortedData);
features=size(SortedData); % features dimensions
clear ALLsubs;

% set up cross validation
crossValidationFolds = cvloop_modelperf;
% Number of rows per fold e.g. for 81 subjects a 9 loop CV means each fold has 9 rows
numberOfRowsPerFold = dataRowNumber / crossValidationFolds;

% Not using the next two matrices but it could be used to store the training and test data for investigation
%crossValidationTrainData = [];
%crossValidationTestData = [];

% initialization of various items to be used subsequently
AccuracyData=zeros(1,cvloop_modelperf);
MeanAbsErr=zeros(1,cvloop_modelperf);
correct_boys=zeros(1,cvloop_modelperf);
incorrect_boys=zeros(1,cvloop_modelperf);
correct_girls=zeros(1,cvloop_modelperf);
incorrect_girls=zeros(1,cvloop_modelperf);
cvloop = 0;
fpr_roc_g1=zeros(10,cvloop_modelperf);
tpr_roc_g1=zeros(10,cvloop_modelperf);
t_roc_g1=zeros(10,cvloop_modelperf);
auc_roc_g1=zeros(1,cvloop_modelperf);
%optrocpt_roc_g1=[];
fpr_roc_g2=zeros(10,cvloop_modelperf);
tpr_roc_g2=zeros(10,cvloop_modelperf);
t_roc_g2=zeros(10,cvloop_modelperf);
auc_roc_g2=zeros(1,cvloop_modelperf);
%optrocpt_roc_g2=[];
grammatrix=[];
determinant=zeros(1,cvloop_modelperf);
lambda = zeros(1,cvloop_modelperf);
grammatrix2=[];
determinant2=zeros(1,cvloop_modelperf);
posdef=zeros(1,cvloop_modelperf);
%groundtruth = zeros(81,1);
%SVMpred = zeros(81,1);

groundtruth = [];
SVMpred = [];
                                        
switch KernelChoice
  case 1
      % Setting up grid search for tuning hyperparameter
      stepSize = 1;
      log2c_list = -10.1:stepSize:10.1;

      cvMatrix = []; %will save the nested crossvalidation performance
      
      % For loop for grid search note it also includes a CV loop matching the CV for model performance
      % Recommendation is that C specifically depends on sample size hence to keep it simple picket the
      % whole data set
      %bestcv = 0;
      [bestc, bestcv] = LinearHyperPar(log2c_list, dataRowNumber, SortedData(:,3:features(2)), SortedData(:,3:features(2)),cvloop_parmselect, SortedData(:,2)); %
      assignin('base', 'bestc', bestc);
      Kernel2 = @(X,Y) (X*Y');
      
   case 2
       % Setting up grid search for tuning hyperparameter
       
       stepSize = 1;
       log2c_list = -10.1:stepSize:10.1;
       log2g_list = -20:stepSize:20;
       
       %cvMatrix = []; %will save the nested crossvalidation performance
       
      [bestc, bestsigma, bestcv] = RBFHyperPar(log2c_list, log2g_list, dataRowNumber, SortedData(:,3:features(2)), SortedData(:,3:features(2)),cvloop_parmselect, SortedData(:,2)); %#ok<ASGLU> %
      assignin('base','bestc',bestc);
      assignin('base','bestsigma',bestsigma);
       Kernel2 = @(X,Y) exp(-bestsigma .* pdist2(X,Y,'euclidean').^2);
                        
    case 3
    % Setting up grid search for tuning hyperparameter
    
    stepSize = 1;
    log2c_list = -10.1:stepSize:10.1;
    log2g_list = -20:stepSize:20;
    ctrfreq_list = 0.1:4.1
                    
    [bestc, bestsigma, bestctrfreq, bestcv] = MorletHyperPar(log2c_list, log2g_list, ctrfreq_list, dataRowNumber, SortedData(:,3:features(2)), SortedData(:,3:features(2)),cvloop_parmselect, SortedData(:,2));
    assignin('base','bestc',bestc);
    assignin('base','bestsigma',bestsigma);
    assignin('base','bestctrfreq',bestctrfreq);
    Kernel2 = @(X,Y) cos(bestctrfreq*sqrt(bestsigma).* pdist2(X,Y,'euclidean')).*exp(-bestsigma .* pdist2(X,Y,'euclidean').^2)-exp(-0.5*bestctrfreq*bestctrfreq).*exp(-bestsigma .* pdist2(X,Y,'euclidean').^2);
    case 4
        % Setting up grid search for tuning hyperparameter
        stepSize = 1;
        log2c_list = -10.1:stepSize:10.1;
        log2g_list = -20:stepSize:20;
        
        %cvMatrix = zeros(1, 861); %will save the nested crossvalidation performance
        [bestc, bestsigma, bestcv] = MexicanHatHyperPar(log2c_list, log2g_list, dataRowNumber, SortedData(:,3:features(2)), SortedData(:,3:features(2)),cvloop_parmselect, SortedData(:,2));
        assignin('base','bestc',bestc);
        assignin('base','bestsigma',bestsigma);
        Kernel2 = @(X,Y) 2/sqrt(3)*pi^(1/4)*(1-bestsigma*pdist2(X,Y,'euclidean').^2).*exp(-0.5*bestsigma .* pdist2(X,Y,'euclidean').^2);
    case 5
        % Setting up grid search for tuning hyperparameter
        stepSize = 1;
        log2c_list = -10.1:stepSize:10.1;
        log2g_list = -20:stepSize:20;
        log2g2_list = -20:stepSize:20;
        
        %cvMatrix = []; %will save the nested crossvalidation performance
        
        [bestc, bestsigma, bestsigma2, bestcv] = MultiMHHyperPar(log2c_list, log2g_list, log2g2_list, dataRowNumber, SortedData(:,3:features(2)), SortedData(:,3:features(2)),cvloop_parmselect, SortedData(:,2));

        assignin('base','bestc',bestc);
        assignin('base','bestsigma',bestsigma);
        assignin('base','bestsigma2',bestsigma2);
        Kernel2 = @(X,Y) (bestsigma2/(bestsigma2-bestsigma)).*exp(-bestsigma2.*pdist2(X,Y,'euclidean').^2)-(bestsigma/(bestsigma-bestsigma2)).*exp(-bestsigma .* pdist2(X,Y,'euclidean').^2); %Multiscale Mexican Hat
    case 6
        % Setting up grid search for tuning hyperparameter
        
        stepSize = 1;
        log2c_list = -10.1:stepSize:10.1;
        log2g_list = -20:stepSize:20;
        ctrfreq_list = 0.1:4.1;
        gamma_list = 0.1:5.1;
        
        %cvMatrix = []; %will save the nested crossvalidation performance
        
        [bestc, bestsigma, bestctrfreq, bestgamma, bestcv] = MorletRBFHyperPar(log2c_list, log2g_list, ctrfreq_list, gamma_list, dataRowNumber, SortedData(:,3:features(2)), SortedData(:,3:features(2)),cvloop_parmselect, SortedData(:,2));

        assignin('base','bestc',bestc);
        assignin('base','bestsigma',bestsigma);
        assignin('base','bestctrfreq',bestctrfreq);
        assignin('base','bestgamma',bestgamma);
        Kernel2 = @(X,Y) exp(-bestgamma*(2-2*cos(bestctrfreq*sqrt(bestsigma).* pdist2(X,Y,'euclidean')).*exp(-bestsigma .* pdist2(X,Y,'euclidean').^2)));%adaptive wavelet
    
      case 7
          % Input dialog 
    
end

% Setting up crossvalidation for loop
for startOfRow = 1:numberOfRowsPerFold:dataRowNumber % middle element represents steps
    testRows = startOfRow:startOfRow+numberOfRowsPerFold-1;
    if (startOfRow == 1)
        trainRows = max(testRows)+1:dataRowNumber;
    else
        trainRows = [1:startOfRow-1 max(testRows)+1:dataRowNumber];
    end
cvloop = cvloop + 1; %track the loop and this parameter is used later for indexing
% Obtaining the dimensions of the training and test sets
numTrain = size(SortedData(trainRows,3:features(2)) ,1);
numTest = size(SortedData(testRows,3:features(2)) ,1);
POSITION = [10 10 200 20]; % Position of uiwaitbar in pixels.
H = uiwaitbar(POSITION);
uiwaitbar(H,cvloop/9);
pause(.5);
% f = figure('Name', 'Progress Bar Example', 'Position', [100 100 200 20]);
% progressBar = uiProgressBar(f);
% uiProgressBar(f, cvloop/9);
% pause(.5);
% axes(handles.axes6);
% f = figure('Name', 'Progress Bar Example', 'Position', [10 10 200 20]);
% progressBar = uiProgressBar(f);
% uiProgressBar(f, cvloop/9);
% pause(.5);


% setting up the kernel for training data post optimal hyperparameter selection
K2 =  [ (1:numTrain)', Kernel2(SortedData(trainRows ,3:features(2)),SortedData(trainRows ,3:features(2)))];
assignin('base','K2',K2);
if (regyes == 1)
    %Regularization
    grammatrix=K2(:,2:end); % drop the 1st index column
    determinant(:,cvloop) = det(grammatrix); %check if determinants are positive
    lambda(:,cvloop) = abs(min(eig(grammatrix))); %obtain the eigen values of the gram matrix and pick minimum
    if min(eig(grammatrix))<0
        grammatrix2 = grammatrix + (abs(min(eig(grammatrix)))+0.0001)*eye(numTrain);% in case of negative values add Regularization parameter
    else
        grammatrix2 = grammatrix;
    end
    determinant2(:,cvloop) = det(grammatrix2);% check again for negative values
    %The CHOL function provides an optional second output argument "p" which is zero if the matrix is
    % found to be positive definite. If the input matrix is not positive definite,
    %then p will be a positive integer:
    [~,p]=chol(grammatrix2);
    posdef(:,cvloop) = p;
    
    % Modify K2 with the regularized gram matrix
    K2 = [ (1:numTrain)', grammatrix2];
end

%setting up the kernel for test data
KK = [ (1:numTest)', Kernel2(SortedData(testRows,3:features(2)),SortedData(trainRows,3:features(2)))];
assignin('base','KK',KK);
%% train and test using libsvm tools
cmd = ['-t 4 -c ', num2str(bestc)]; % picked the best possible trade-off C
model = svmtrain(SortedData(trainRows,2), K2, cmd);
[predClass, acc, decVals] = svmpredict(SortedData(testRows ,2), KK, model);
 
% Saving the accuracy information
AccuracyData(:,cvloop) = acc(1,1);

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
%groundtruth((1:length(ground_truth))*cvloop,:) = ground_truth;
%SVMpred((1:length(ground_truth))*cvloop,:) = predicted;
groundtruth = [groundtruth ground_truth];
SVMpred = [SVMpred predicted];

% Mean Absolute Error using mae function from the neural network Toolbox
MeanAbsErr(:,cvloop)= mae(predicted,ground_truth);
       
% obtain the confusion matrix using confusionmat command from the ML Toolbox
[C,order] = confusionmat(SortedData(testRows,2),predClass); %order informs how to read the confusion matrix
correct_boys(:,cvloop) = C(1,1);
incorrect_boys(:,cvloop) = C(1,2);
correct_girls(:,cvloop)= C(2,2);
incorrect_girls(:,cvloop) = C(2,1);
       
       
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
auc_roc_g1(:,cvloop) = auc_g1;
%optrocpt_roc_g1=[optrocpt_roc_g1, optrocpt_g1];
       
fpr_roc_g2(1:length(fpr_g2),cvloop) = fpr_g2;
tpr_roc_g2(1:length(tpr_g2),cvloop) = tpr_g2;
t_roc_g2(1:length(t_g2),cvloop) = t_g2;
auc_roc_g2(:,cvloop) = auc_g2;
%optrocpt_roc_g2=[optrocpt_roc_g2, optrocpt_g2];

%precision = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);
%recall = @(confusionMat) diag(confusionMat)./sum(confusionMat,1)';
%f1Scores = @(confusionMat) 2*(precision(confusionMat).*recall(confusionMat))./(precision(confusionMat)+recall(confusionMat));
%meanF1 = @(confusionMat) mean(f1Scores(confusionMat));
end
            
% Take mean of the fpr, tpr from each loop and plot ROC
fpr_roc_final(:,1) = mean(fpr_roc_g1(:,1:cvloop),2);
tpr_roc_final(:,1) = mean(tpr_roc_g1(:,1:cvloop),2);
t_roc_final(:,1) = mean(t_roc_g1(:,1:cvloop),2);
fpr_roc_final(:,2) = mean(fpr_roc_g2(:,1:cvloop),2);
tpr_roc_final(:,2) = mean(tpr_roc_g2(:,1:cvloop),2);
t_roc_final(:,2) = mean(t_roc_g2(:,1:cvloop),2);
for k=1:2
h=figure('visible', 'off');;
plot(fpr_roc_final(:,k),tpr_roc_final(:,k),'linewidth',3);
ax = gca;
ax.LineWidth = 1.5;
ax.TitleFontSizeMultiplier = 1.6;
ax.LabelFontSizeMultiplier = 2;
ax.XLabel.FontWeight = 'Bold';
ax.YLabel.FontWeight = 'Bold';
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for SVM Classification')
saveas(h,sprintf('ROC_SVM_%d.png',k)); % will create FIG1, FIG2,...
end

% AUC statistic --useful in ML
auc_roc_final(:,1) = mean(auc_roc_g1);
auc_roc_final(:,2) = mean(auc_roc_g2);
% Optimal ROC point estimation
%optrocpc_final(:,1) = mean(optrocpt_roc_g1);
%optrocpc_final(:,2) = mean(optrocpt_roc_g2);

% Next section is geared towards measuring performance
mean_correct_boys = mean(correct_boys);
mean_incorrect_boys = mean(incorrect_boys);
mean_correct_girls = mean(correct_girls);
mean_incorrect_girls = mean(incorrect_girls);
per_corr_boys = mean_correct_boys/(mean_correct_boys+mean_incorrect_boys); % percent correct boys
per_corr_girls = mean_correct_girls/(mean_correct_girls+mean_incorrect_girls);% percent correct girls
       
% Balanced Accuracy http://mvpa.blogspot.com/2015/12/balanced-accuracy-what-and-why.html
bal_Accuracy = (per_corr_boys+per_corr_girls)/2; % Balanced accuracy
assignin('base', 'bal_Accuracy', bal_Accuracy);       
% http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
% https://ifordata.wordpress.com/tag/confusion-matrix/
% What proportion of the predicted classes are correct? These measures are called Precision and Recall.
% precision = tp/(tp+fp)
precision_boys = mean_correct_boys/(mean_correct_boys+mean_incorrect_girls);
precision_girls = mean_correct_girls/(mean_correct_girls+mean_incorrect_boys);

% https://en.wikipedia.org/wiki/Precision_and_recall
% recall = tp/(tp+fn)
Recall_boys = mean_correct_boys/(mean_correct_boys+mean_incorrect_boys);
Recall_girls = mean_correct_girls/(mean_correct_girls+mean_incorrect_girls);

% https://en.wikipedia.org/wiki/Sensitivity_and_specificity
% sensitivity = recall  = tpr = tp/(tp+fn)
sensitivity_boys = mean_correct_boys/(mean_correct_boys+mean_incorrect_boys);
sensitivity_girls = mean_correct_girls/(mean_correct_girls+mean_incorrect_girls);
       
% https://en.wikipedia.org/wiki/Sensitivity_and_specificity
% specificity = tnr = TN/(TN+FP)
specificity_boys = mean_correct_girls/(mean_incorrect_girls+mean_correct_girls);
specificity_girls = mean_correct_boys/(mean_incorrect_boys+mean_correct_boys);
       
% https://wikimedia.org/api/rest_v1/media/math/render/svg/8b64097b6362d28387a0c4650f2fed2bc5ea9fe9
% F1 = 2*TP(2*TP+FP+FN)
% F1 = 2 * (precision x recall)/(precision + recall)
Fscore_boys = 2*mean_correct_boys/(2*mean_correct_boys+mean_incorrect_girls+mean_incorrect_boys);
Fscore_girls = 2*mean_correct_girls/(2*mean_correct_girls+mean_incorrect_boys+mean_incorrect_girls);
       
% FPR = FP/(FP+TN)
FP_rate_boys = mean_incorrect_girls/(mean_incorrect_girls+mean_correct_girls);
FP_rate_girls = mean_incorrect_boys/(mean_incorrect_boys+mean_correct_boys);

% TPR = sensitivity = recall
TP_rate_boys = mean_correct_boys/(mean_correct_boys+mean_incorrect_boys);
TP_rate_girls = mean_correct_girls/(mean_correct_girls+mean_incorrect_girls);

%http://standardwisdom.com/softwarejournal/2011/12/confusion-matrix-another-single-value-metric-kappa-statistic/
% Random Accuracy also known as expected Accuracy is defined as the accuracy
% that any random classifier would be expected to achieve based on the confusion matrix.
random_Accuracy = ((mean_correct_girls+mean_incorrect_girls)*(mean_correct_girls+mean_incorrect_boys)+(mean_incorrect_boys+mean_correct_boys)*(mean_incorrect_girls+mean_correct_boys))/(mean_correct_boys+mean_correct_girls+mean_incorrect_boys+mean_incorrect_girls)^2;

%http://standardwisdom.com/softwarejournal/2011/12/confusion-matrix-another-single-value-metric-kappa-statistic/
mean_Accuracy = mean(AccuracyData); % Overall accuracy

% Kappa is also stated as (observed accuracy-expected accuracy)/(1-expected accuracy)
% http://stats.stackexchange.com/questions/82162/kappa-statistic-in-plain-english
kappa=(mean_Accuracy/100-random_Accuracy)/(1-random_Accuracy);
mean_absolute_error=mean(MeanAbsErr); % Mean absolute error

sensitivity = (sensitivity_boys + sensitivity_girls)/2;
specificity = (specificity_boys + specificity_girls)/2;
TP_weighted=(TP_rate_boys+TP_rate_girls)/2;
FP_weighted=(FP_rate_boys+FP_rate_girls)/2;
Precision_weighted=(precision_boys+precision_girls)/2;
Recall_weighted=(Recall_boys+Recall_girls)/2;
Fscore_weighted=(Fscore_boys+Fscore_girls)/2; %F score is the harmonic mean of precision and sensitivity

Pred_boys = TP_rate_boys*40; % Predicted number of boys in total sample
Pred_girls = TP_rate_girls*41; % Predicted number of girls in total sample

% print data in an organized manner
datasave=[mean_Accuracy mean_absolute_error kappa TP_weighted FP_weighted Precision_weighted Recall_weighted Fscore_weighted mean(auc_roc_final) round(Pred_boys) round(Pred_girls)];
%fprintf(' Accuracy MeanAbsoluteError Kappa TP FP Precision Recall F1score AUC Predboys Predgirls\n')
%disp(datasave)

% Detailed measures
extradatasave=[100*bal_Accuracy precision_boys precision_girls Recall_boys Recall_girls  sensitivity_boys sensitivity_girls specificity_boys specificity_girls Fscore_boys Fscore_girls FP_rate_boys FP_rate_girls TP_rate_boys TP_rate_girls];
%fprintf(' BalancedAccuracy Precisionboys Precisiongirls Recallboys Recallgirls  Sensitivityboys Sensitivitygirls Specificityboys Specificitygirls Fscoreboys Fscoregirls FPrateboys FPrategirls TPrateboys TPrategirls\n')
%disp(extradatasave)

%Final Data measures
finaldatasave=[100*bal_Accuracy kappa Precision_weighted Recall_weighted Fscore_weighted auc_roc_final(2)];

%Best Hyperparameters
%besthyperparameters = [bestc, bestsigma];
% Plot the final Confusion Matrix using plotconfusion from Neural Network
% Toolbox
h=figure('visible', 'off');
groundtruth = reshape(groundtruth, dataRowNumber, 1);
SVMpred = reshape(SVMpred, dataRowNumber, 1);
plotconfusion(groundtruth',SVMpred');
saveas(h,sprintf('Confusion Matrix SVM.png'));

% Save the stats in an excel file
% Store the characters in a cell array and then using cell2table we convert
% it to a table format
T = cell2table({'Accuracy', 'MeanAbsoluteError', 'Kappa', 'TP', 'FP', 'Precision', 'Recall', 'F1score', 'AUC', 'Predboys', 'Predgirls'; mean_Accuracy, mean_absolute_error, kappa, TP_weighted, FP_weighted, Precision_weighted, Recall_weighted, Fscore_weighted, mean(auc_roc_final), round(Pred_boys), round(Pred_girls)});
T2 = cell2table({'BalancedAccuracy', 'Precisionboys', 'Precisiongirls', 'Recallboys', 'Recallgirls',  'Sensitivityboys', 'Sensitivitygirls', 'Specificityboys', 'Specificitygirls', 'Fscoreboys', 'Fscoregirls', 'FPrateboys', 'FPrategirls', 'TPrateboys', 'TPrategirls';100*bal_Accuracy, precision_boys, precision_girls, Recall_boys, Recall_girls, sensitivity_boys, sensitivity_girls, specificity_boys, specificity_girls, Fscore_boys, Fscore_girls, FP_rate_boys, FP_rate_girls, TP_rate_boys, TP_rate_girls});
%T3 = cell2table({'BESTC', 'BESTsigma'; bestc, bestsigma});
T3 = cell2table({'BalancedAccuracy', 'Kappa','Precision', 'Recall', 'F1score', 'AUC'; 100*bal_Accuracy,kappa, Precision_weighted, Recall_weighted, Fscore_weighted, auc_roc_final(2)});

%writetable helps to save the file into an excel file
filename = 'SVM_result.xlsx';
writetable(T, filename,'sheet',1);
writetable(T2, filename,'sheet',2);
writetable(T3, filename,'sheet',3);
%writetable(T4, filename,'sheet',4);

% Plot the final Confusion Matrix using plotconfusion from Neural Network
% Toolbox
h=figure('visible', 'off');
groundtruth = reshape(groundtruth, dataRowNumber, 1);
SVMpred = reshape(SVMpred, dataRowNumber, 1);
plotconfusion(groundtruth',SVMpred');
saveas(h,sprintf('Confusion Matrix.png'));

set(handles.BA, 'String', num2str(100*bal_Accuracy));
set(handles.kappa, 'String', num2str(kappa));
set(handles.sensitivity, 'String', num2str(sensitivity));
set(handles.specificity, 'String', num2str(specificity));
set(handles.precision, 'String', num2str(Precision_weighted));
set(handles.recall, 'String', num2str(Recall_weighted));
set(handles.fscore, 'String', num2str(Fscore_weighted));
set(handles.auc, 'String', num2str(auc_roc_final(2)));
axes(handles.axes1);
imshow('ROC_SVM_2.png');
axes(handles.axes2);
imshow('Confusion Matrix SVM.png');




% --- Executes on selection change in kernel.
function kernel_Callback(hObject, eventdata, handles)
% hObject    handle to kernel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns kernel contents as cell array
%        contents{get(hObject,'Value')} returns selected item from kernel
listbox_index=get(hObject, 'Value');
switch listbox_index
    case 1
        %Kernel = @(X,Y) X*Y';
        %Kernel2 = @(X,Y) X*Y';
        KernelChoice = 1;
        %assignin('base','Kernel',Kernel);
        %assignin('base','Kernel2',Kernel2);
        assignin('base','KernelChoice',KernelChoice);
    case 2
        %Kernel = @(X,Y) exp(-sigma .* pdist2(X,Y,'euclidean').^2);
        %Kernel2 = @(X,Y) exp(-bestsigma .* pdist2(X,Y,'euclidean').^2);
        KernelChoice = 2;
        %assignin('base','Kernel',Kernel);
        %assignin('base','Kernel2',Kernel2);
        assignin('base','KernelChoice',KernelChoice);
    case 3 
        %Kernel = @(X,Y) cos(ctrfreq*sqrt(sigma).* pdist2(X,Y,'euclidean')).*exp(-sigma .* pdist2(X,Y,'euclidean').^2)-exp(-0.5*ctrfreq*ctrfreq).*exp(-sigma .* pdist2(X,Y,'euclidean').^2);
        %Kernel2 = @(X,Y) cos(bestctrfreq*sqrt(bestsigma).* pdist2(X,Y,'euclidean')).*exp(-bestsigma .* pdist2(X,Y,'euclidean').^2)-exp(-0.5*bestctrfreq*bestctrfreq).*exp(-bestsigma .* pdist2(X,Y,'euclidean').^2);
        KernelChoice = 3;
        %assignin('base','Kernel',Kernel);
        %assignin('base','Kernel2',Kernel2);
        assignin('base','KernelChoice',KernelChoice);
    case 4
        %Kernel = @(X,Y) 2/sqrt(3)*pi^(1/4)*(1-sigma*pdist2(X,Y,'euclidean').^2).*exp(-0.5*sigma .* pdist2(X,Y,'euclidean').^2);
        %Kernel2 = @(X,Y) 2/sqrt(3)*pi^(1/4)*(1-bestsigma*pdist2(X,Y,'euclidean').^2).*exp(-0.5*bestsigma .* pdist2(X,Y,'euclidean').^2);
        KernelChoice = 4;
        %assignin('base','Kernel',Kernel);
        %assignin('base','Kernel2',Kernel2);
        assignin('base','KernelChoice',KernelChoice);
    case 5
        %Kernel = @(X,Y) (sigma2/(sigma2-sigma)).*exp(-sigma2.*pdist2(X,Y,'euclidean').^2)-(sigma/(sigma-sigma2)).*exp(-sigma.* pdist2(X,Y,'euclidean').^2); %Multiscale Mexican Hat 
        %Kernel2 = @(X,Y) (bestsigma2/(bestsigma2-bestsigma)).*exp(-bestsigma2.*pdist2(X,Y,'euclidean').^2)-(bestsigma/(bestsigma-bestsigma2)).*exp(-bestsigma .* pdist2(X,Y,'euclidean').^2); %Multiscale Mexican Hat
        KernelChoice = 5;
        %assignin('base','Kernel',Kernel);
        %assignin('base','Kernel2',Kernel2);
        assignin('base','KernelChoice',KernelChoice);
    case 6    
        %Kernel = @(X,Y) exp(-gamma*(2-2*cos(ctrfreq*sqrt(sigma).* pdist2(X,Y,'euclidean')).*exp(-sigma .* pdist2(X,Y,'euclidean').^2)));%adaptive wavelet
        %Kernel2 = @(X,Y) exp(-bestgamma*(2-2*cos(bestctrfreq*sqrt(bestsigma).* pdist2(X,Y,'euclidean')).*exp(-bestsigma .* pdist2(X,Y,'euclidean').^2)));%adaptive wavelet
        KernelChoice = 6;
        %assignin('base','Kernel',Kernel);
        %assignin('base','Kernel2',Kernel2);
        assignin('base','KernelChoice',KernelChoice);
    case 7
        KernelChoice = 7;
        assignin('base', 'KernelChoice', KernelChoice);
        prompt = {'Customized Kernel', 'Hyperparameter 1 Name', 'Hyperparameter 2 Name', 'Default C (in case of no CV)', 'Hyperparameter 1 (in case of no CV)', 'Hyperparameter 2 (in case of no CV)', 'Grid Range for C in logbase2', 'Grid Range for Hyperparameter 1 in logbase2','Grid Range for Hyperparameter 2 in logbase2'};
        dlg_title = 'Input Custom Kernel';
        num_lines = 1;
        defaultans = {'exp(-sigma .* pdist2(X,Y).^2)','sigma' , '', '1', '0.001', '', '-10.1:1:9.9', '-20:1:20', ''};
        answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
        assignin('base', 'Kernel', answer(1));
        assignin('base', 'Hyperparm1name', answer(2));
        assignin('base', 'Hyperparm2name', answer(3));
        assignin('base', 'bestc', answer(4));
        assignin('base', 'besthyper1', answer(5));
        assignin('base','besthyper2', answer(6));
        assignin('base','gridc', answer(7));
        assignin('base', 'gridhyper1', answer(8));
        assignin('base', 'gridhyper2', answer(9));
end
        % --- Executes during object creation, after setting all properties.
function kernel_CreateFcn(hObject, eventdata, handles)
% hObject    handle to kernel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function BA_Callback(hObject, eventdata, handles)
% hObject    handle to BA (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of BA as text
%        str2double(get(hObject,'String')) returns contents of BA as a double
%bal_Accuracy = evalin('base', 'bal_Accuracy');



% --- Executes during object creation, after setting all properties.
function BA_CreateFcn(hObject, eventdata, handles)
% hObject    handle to BA (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function kappa_Callback(hObject, eventdata, handles)
% hObject    handle to kappa (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of kappa as text
%        str2double(get(hObject,'String')) returns contents of kappa as a double


% --- Executes during object creation, after setting all properties.
function kappa_CreateFcn(hObject, eventdata, handles)
% hObject    handle to kappa (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function sensitivity_Callback(hObject, eventdata, handles)
% hObject    handle to sensitivity (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of sensitivity as text
%        str2double(get(hObject,'String')) returns contents of sensitivity as a double


% --- Executes during object creation, after setting all properties.
function sensitivity_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sensitivity (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function specificity_Callback(hObject, eventdata, handles)
% hObject    handle to specificity (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of specificity as text
%        str2double(get(hObject,'String')) returns contents of specificity as a double


% --- Executes during object creation, after setting all properties.
function specificity_CreateFcn(hObject, eventdata, handles)
% hObject    handle to specificity (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function precision_Callback(hObject, eventdata, handles)
% hObject    handle to precision (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of precision as text
%        str2double(get(hObject,'String')) returns contents of precision as a double


% --- Executes during object creation, after setting all properties.
function precision_CreateFcn(hObject, eventdata, handles)
% hObject    handle to precision (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function recall_Callback(hObject, eventdata, handles)
% hObject    handle to recall (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of recall as text
%        str2double(get(hObject,'String')) returns contents of recall as a double


% --- Executes during object creation, after setting all properties.
function recall_CreateFcn(hObject, eventdata, handles)
% hObject    handle to recall (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function fscore_Callback(hObject, eventdata, handles)
% hObject    handle to fscore (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of fscore as text
%        str2double(get(hObject,'String')) returns contents of fscore as a double


% --- Executes during object creation, after setting all properties.
function fscore_CreateFcn(hObject, eventdata, handles)
% hObject    handle to fscore (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function auc_Callback(hObject, eventdata, handles)
% hObject    handle to auc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of auc as text
%        str2double(get(hObject,'String')) returns contents of auc as a double


% --- Executes during object creation, after setting all properties.
function auc_CreateFcn(hObject, eventdata, handles)
% hObject    handle to auc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on key press with focus on loadimages and none of its controls.
function loadimages_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to loadimages (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in subjectlist.
function subjectlist_Callback(hObject, eventdata, handles)
% hObject    handle to subjectlist (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename, pathname] = uigetfile( ...
{ '*.txt',  'text file (*.txt)'; ...
   '*.*',  'All Files (*.*)'}, ...
   'Pick a file');
NAME = [pathname,filename]; %path and name
subject_list = textread(NAME, '%s');
assignin('base','subject_list',subject_list);
numsubjects = length(subject_list);
assignin('base','numsubjects',numsubjects);


% --- Executes on key press with focus on normyes and none of its controls.
function normyes_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to normyes (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over normyes.
function normyes_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to normyes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in wholebrain.
function wholebrain_Callback(hObject, eventdata, handles)
% hObject    handle to wholebrain (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of wholebrain
if (get(hObject,'Value') == get(hObject,'Max'))
   wholebrain = 1 %wholebrain turned on
else
  wholebrain = 0 %wholebrain turned off
end
assignin('base','wholebrain',wholebrain);
