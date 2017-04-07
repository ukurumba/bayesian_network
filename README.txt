Bayesian Network Project
------------------------

Unni Kurumbail (ukurumba@u.rochester.edu)
CSC 242
Spring 2017



This project implements the enumeration method as an Exact Inference method and the weighted likelihood method as an Approximate Inference method. 

To execute this project:

1) Navigate to the top level directory (there should be a directory 'src', a directory 'examples', and a script 'main.py').
2) Input the requisite commands similar to the examples Prof. Ferguson provided in the following manner:

	python3 main.py your_file_name OPTIONAL_NUM_ITERATIONS query_var evidence_var_1 evidence_var_1_state evidence_var_2 evidence_var_2_state ...


^ If OPTIONAL_NUM_ITERATIONS is provided, the Weighted Likelihood method is ran instead of the Exact Inference enumeration method.
Please note that, for some reason, the dictionary printing method makes it so that the same state isn't always printed first in python3.
i.e. if you run the first example below you may get either:
 
			{'true' : 0.95, 'false' : 0.05}
						   OR
			{'false' : 0.05, 'true' : 0.95}

3) If you'd like to use some of the built-in examples simply specify the relative path like in the 4th example below.

4) Please note I did not include a "classNames.txt" file as the inference algorithms I implemented aren't directly callable. There's a bit of wrangling that needed to be done b/c of the XMBLIF and BIF parsers and so the only way to run the functions is from the 'main.py' file in the aforementioned way. Sorry for any troubles on your part. 

##############################################################################################

Examples: 

python3 main.py aima-alarm.xml B J true M true
>>> {'true' : 0.95, 'false' : 0.05}

python3 main.py aima-alarm.xml 1000 B J true M true 
>>> {'true' : 0.95, 'false' : 0.05}

python3 main.py alarm.bif 10000 BP CO LOW
>>> {'LOW': 0.95918910868536877, 'NORMAL': 0.031500008760709408, 'HIGH': 0.0093108825539219928}

python3 main.py ./examples/alarm.bif 10000 BP CO LOW
>>> {'LOW': 0.95918910868536877, 'NORMAL': 0.031500008760709408, 'HIGH': 0.0093108825539219928}

###############################################################################################

PLEASE SPECIFY PYTHON3 AS THIS CODE DOES NOT RUN ON PYTHON2. THANKS!