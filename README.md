# DKVMN
Tensorflow implementation of Dynamic Key-Value Memory Networks for Knowledge Tracing 

Reference
---------
* Paper
  * [Dynamic Key Valuy Memory Network](https://arxiv.org/abs/1611.08108)
  * [Deep Knowledge Tracing](https://arxiv.org/abs/1506.05908)


Utiles:
It is recommendable to install anaconda to manage packages and environments, the miniconda version can be download here:
https://conda.io/projects/conda/en/latest/user-guide/install/index.html

Once installed:
	"conda init", to start conda 
	"conda list", list of installed packages. 
	"conda env list", list of configured environments.

To create a new environment digit:

	"conda create --name DKVMNtest python=3.7 " 
	"conda activate DKVMNtest" it allow us to activate the selected environment
 	"conda deactivate"

For Windows needs to run:
	"conda prompt"

To export the environment use:
	"conda env export > environment.yml"

To create the environment using the file :
	"conda env create --file environment.yml"


Run Instructions:
The code is ready to be run at the moment with a configuration of the parameters acording to the coments
Since the last run was done with 50 epoch , in the check points the fiiles contain the trainning parmeter for 
for an accuracy of 72.11 and AUC: 0.74756

the parameter train:  yes=trainning or no=testing


