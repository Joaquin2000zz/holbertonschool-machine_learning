0x03. Hyperparameter Tuning
 By: Alexa Orrico, Software Engineer at Holberton School
 Weight: 5
 Project will start Nov 28, 2022 12:00 AM, must end by Dec 2, 2022 12:00 AM
 will be released at Nov 30, 2022 9:36 AM
 Manual QA review must be done (request it when you are done with the project)
 An auto review will be launched at the deadline


Resources
Read or watch:

Hyperparameter Tuning in Practice
Orthogonalization
Single Number Evaluation Metric
Satisficing and Optimizing Metrics
Gaussian process
Kriging
Machine learning - Introduction to Gaussian processes
Machine learning - Gaussian processes
Quick Start to Gaussian Process Regression
Gaussian processes
Machine learning - Bayesian optimization and multi-armed bandits
Bayesian Optimization
Bayesian Optimization
A Tutorial on Bayesian Optimization
GPy documentation
GPy.kern.src
GPy.plotting.gpy_plot
GPyOpt documentation
GPyOpt.methods.bayesian_optimization
GPyOpt.core.task.space
Learning Objectives
What is Hyperparameter Tuning?
What is random search? grid search?
What is a Gaussian Process?
What is a mean function?
What is a Kernel function?
What is Gaussian Process Regression/Kriging?
What is Bayesian Optimization?
What is an Acquisition function?
What is Expected Improvement?
What is Knowledge Gradient?
What is Entropy Search/Predictive Entropy Search?
What is GPy?
What is GPyOpt?
Requirements
General
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
Your files will be executed with numpy (version 1.15)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.4), but can ignore error E741
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
Unless otherwise noted, you are not allowed to import any module except import numpy as np
All your files must be executable
Install GPy and GPyOpt
pip install --user GPy
pip install --user gpyopt