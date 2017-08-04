function[bestc, bestcv] = LinearHyperPar(Range,dataRowNumber,FeatMat1,FeatMat2,cvloop_parmselect,labels)
%This function calculates the optimal hyperparameters
% Inputs include Range of grid search and FeatMat1 and FeatMat2 are feature matrices
% Typical Range for log2c is -10.1:9.9
bestcv = 0;
for i = 1:length(Range)
    log2c = Range(i)
    K =  [ (1:dataRowNumber)', FeatMat1*FeatMat2'];
    cmd = ['-t 4 -q -v ', num2str(cvloop_parmselect), ' -c ', num2str(2^log2c)]; % - v option is the nested CV loop
    cv = svmtrain(labels,K,cmd);
    if (cv >= bestcv)
        bestcv = cv; bestc = 2^log2c;
    end
    fprintf('%g (best c=%g,rate=%g)\n', log2c, cv, bestc, bestcv);
    progressbar % Create figure and set starting time
    pause(0.01) % Do something important
    progressbar(i/length(Range)) % Update figure
end
