function[bestc, bestsigma, bestcv] = MexicanHatHyperPar(Range1,Range2,dataRowNumber,FeatMat1,FeatMat2,cvloop_parmselect,labels)
%This function calculates the optimal hyperparameters
% Inputs include Range of grid search and FeatMat1 and FeatMat2 are feature matrices
% Typical Range for log2c is -10.1:9.9
bestcv = 0;
for i = 1:length(Range1)
    log2c = Range1(i);
    for j = 1:length(Range2)
        log2g = Range2(j);
        sigma=2^log2g;
        % setting up the Mexican Hat Wavelet kernel for training data
        Kernel = @(X,Y) 2/sqrt(3)*pi^(1/4)*(1-sigma*pdist2(X,Y,'euclidean').^2).*exp(-0.5*sigma .* pdist2(X,Y,'euclidean').^2);
        K =  [ (1:dataRowNumber)', Kernel(FeatMat1,FeatMat2)];
        cmd = ['-t 4 -q -v ', num2str(cvloop_parmselect), ' -c ', num2str(2^log2c)]; % - v option is the nested CV loop
        cv = svmtrain(labels,K,cmd); %model fit
        %cvMatrix = [cvMatrix, cv]; %store fit for comparison
        if (cv >= bestcv)
            bestc = 2^log2c; bestsigma = 2^log2g; bestcv = cv; % selection of optimal hyperparameter
        end
        fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestsigma, bestcv);
        progressbar % Create figure and set starting time
        pause(0.01) % Do something important
        progressbar(i/length(Range1)) % Update figure
    end
end
