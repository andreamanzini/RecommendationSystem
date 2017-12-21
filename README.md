# Project Recommender System

####TODO: introduction

## Setting up the environment

* Install custom surprise library:
	* Clone this repository: https://github.com/manzo94/Surprise/tree/integrated_model
	* Install requirements: pip install -r <path_of_Surprise_folder>/requirements.txt
	* Install library:  pip install <path_of_Surprise_folder>
	* ### Throubleshooting:
		* The library may ask the installation of the Visual Studio 2015 compiler.
		  If asked follow the link which shows up in the error message and install the software.
	    * If the library can't be installed try to update all the python modules with 
		  "conda upgrade --all" and "pip install -U <modules>"
        * If you are using a mac laptop, make sure to you have xcode downloaded. Also, command line tools should be installed (you can install them by running xcode-select --install). Note that command line tools should be up to date.

* Install requirements for the project
	* Run: pip install -r <folder_of_project>/requirements.txt  
	  (Install required modules using conda if you prefer)

* Data Sets:
    * Download the data_train.csv and sample_submission.csv files from the competition page on kaggle.
	  (https://www.kaggle.com/c/epfml17-rec-sys). The files should be put in the folder ../data, with
	  respect to the code folder.
	
## Description

* Files:
     * run.py is the main code which generates dumps (if not already generated) and the final predictions file (after the blending)
     * models.py contains all the models used in the blending. Both custom and Surprise models are included in this file.
	   Each model function creates a pickle dump of the predictions and during the blending all the predictions are collected again from the dump files. Default location for the dumps are: ../predictions and ../test. The folders are automatically created if missing.
     * *_helpers are files which contain helper functions. The description is inside the files. Take a look there add stuff here

	 
## Performances

	The code takes approximately 4 hours on a Intel i7 6700HQ when executed from scratch (no pickle dump provided) and less than 2 minutes when all the model predictions have already been dumped (only blending). The algorithms are not parallelized, so only a fraction of the total computational power of the	CPU is used. No GPU is required since it is not exploited. The RAM usage is always lower than 4GB. In our tests, the code ran perfectly on a 8GB system.
	
	These are the minutes the models take separately with data splitted in 93.3% training, 6.7% testing. The models not listed take less than 1 minute.
    matrix_factorization_SGD : 13
    matrix_factorization_ALS : 13
    KNN_user                 : 50
    KNN_item                 : 5
    slope_one                : 4
    svdpp                    : 3
    svdpp                    : 140
	

 

