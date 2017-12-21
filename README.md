# Project Recommender System

####TODO: introduction

## Setting up the environment

* Install custom surprise library:
	* Clone this repository: https://github.com/manzo94/Surprise/tree/integrated_model
	* Install requirements: pip install -r <folder_of_Surprise>/requirements.txt
	* Install library:  pip install <folder_of_Surprise>
	* ### Throubleshooting:
		* The library may ask the installation of the Visual Studio 2015 compiler.
		  If asked follow the link which shows up in the error message and install the software.
	    * If the library can't be installed try to update all the python modules with 
		  "conda upgrade --all" and "pip install -U <modules>"

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
     * *_helpers are files which contain helper functions. The description is inside the files. Take a look there add stuff here

	 
## Performances
	####TODO: add final score on Kaggle and position in the leaderboard. Add the table with computation time for algorithms
	

 

