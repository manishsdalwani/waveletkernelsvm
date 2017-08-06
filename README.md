# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

This respository is for the application of wavelet kernels in conjuction with SVM to classify groups based on fMRI data

### How do I get set up? ###

Note the code for this paper uses SPM functions, Matlab Machine Learning commands and Matlab Neural Network commands

Dependencies: SPM, Matlab recent version (2016), ML toolbox and Neural Network toolbox and most importantly libSVM for svmtrain and svmpredict functions 

How to run tests: You can organize contrast maps in one folder change the parent directory and change the individual subjects id to run classification between groups

Deployment instructions: Make sure scripts like feature normalization, contrast images are hopefully in the same folder along with the code itself

### Contribution guidelines ###

Note a major credit for this code is given to https://sites.google.com/site/kittipat/projects. 

There are excellent snippets of using SVM and many of wavelet machine code is inspired from the Kittipat's homepage. 

Writing tests: Insure in the code that labels are assigned properly to groups and that number of cross-validation loops make sense Code review : Code may be updated with optimization of several routines and functions  

Other guidelines: The code is open source and authors encourage optimization by others as long as the work done by the authors are cited properly.

### Who do I talk to? ###

Manish Dalwani (manish.dalwani@UCDenver.edu) or Debashis Ghosh (Debashis.Ghosh@UCDenver.edu) 
