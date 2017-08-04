function[bestc, bestsigma, bestsigma2, bestcv] = MultiMHHyperPar(Range1,Range2,Range3,dataRowNumber,FeatMat1,FeatMat2,cvloop_parmselect,labels)
%This function calculates the optimal hyperparameters
% Inputs include Range of grid search and FeatMat1 and FeatMat2 are feature matrices
% Typical Range for log2c is -10.1:9.9
bestcv = 0;
for i = 1:length(Range1)
    log2c = Range1(i);
    for j = 1:length(Range2)
        log2g = Range2(j);
        sigma=2^log2g;
        for k = 1:length(Range3)
            log2g2 = Range3(k);
            sigma2=2^log2g2;
            % setting up the MultiMH kernel for training data
            Kernel = @(X,Y) (sigma2/(sigma2-sigma)).*exp(-sigma2.*pdist2(X,Y,'euclidean').^2)-(sigma/(sigma-sigma2)).*exp(-sigma.* pdist2(X,Y,'euclidean').^2); %Multiscale Mexican Hat
            K =  [ (1:dataRowNumber)', Kernel(FeatMat1,FeatMat2)];
            cmd = ['-t 4 -q -v ', num2str(cvloop_parmselect), ' -c ', num2str(2^log2c)]; % - v option is the nested CV loop
            cv = svmtrain(labels,K,cmd); %model fit
            %cvMatrix = [cvMatrix, cv]; %store fit for comparison
            if (cv >= bestcv)
                bestc = 2^log2c; bestsigma = 2^log2g; bestcv = cv; bestsigma2 = 2^log2g2; % selection of optimal hyperparameter
            end
            fprintf('%g %g %g (best c=%g, g=%g, rate=%g, sigma2=%g)\n', log2c, log2g, cv, bestc, bestsigma,bestcv, bestsigma2);
            progressbar % Create figure and set starting time
            pause(0.01) % Do something important
            progressbar(i/length(Range1)) % Update figure
        end
    end
end
