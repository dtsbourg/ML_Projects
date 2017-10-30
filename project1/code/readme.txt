README for the project 1 in Machine Learning
Dylan Bourgeois
Antoine Mougeot
Philippe Verbist

/!\ Please use python3 to run the scripts

Where to place the data-sets:
————————————————————————————

You can simply put the two data-sets (train.csv and test.csv) in this folder.



Mandatory functions:
———————————————————

You will find the implementations of the six mandatory algorithms in the file implementations.py. 
The file called implementations_test.py tests the six mandatory functions on the train data-set. (The results are the one showed in our report)
Simply run the following command:

    - implementation_test.py


Best result on Kaggle:
—————————————————————

The script run.py produces the selected result on Kaggle. 
It can be run like this:
    - run.py -h  -> Does not produce the output. Show the usage for the function
    - run.py     -> Produces the output, without displaying any messages
    - run.py -v  -> Produces the output, and shows the mains steps on the screen
    - run.py -V  -> Produces the output, displays a maximum of information

We recommend you tu use: run.py -V
The output will be: submission_run.csv

Optimization file:
——————————————————

The script reg_log_reg_opti.py generates a (long) optimisation on the different parameters of the reg_logistic_regression function. Again, the 
It can be run like this:
    - optimization.py -h  -> Does not produce the output. Show the usage for the function
    - optimization.py     -> Produces the output, without displaying any messages
    - optimization.py -v  -> Produces the output, and shows the mains steps on the screen
    - optimization.py -V  -> Produces the output, displays a maximum of information
    - optimization.py -q  -> Perform the optimisation on a reduced range of parameters (but it does not give an optimal solution). The -q run takes ~20 minutes.
The normal run takes several hours

We recommend you tu use: optimization.py -q -V

