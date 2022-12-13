0x0D. RNNs
 By: Alexa Orrico, Software Engineer at Holberton School
 Weight: 4
 Project will start Dec 12, 2022 12:00 AM, must end by Dec 16, 2022 12:00 AM
 will be released at Dec 15, 2022 4:48 AM
 An auto review will be launched at the deadline


Resources
Read or watch:

MIT 6.S191: RNN
Introduction to RNNs
Illustrated Guide to RNNs
Illustrated Guide to LSTM’s and GRU’s: A step by step explanation
RNNs Tutorial, Parts 1
RNNs Tutorial, Parts 2
RNNs Tutorial, Parts 3
RNNs Tutorial, Parts 4
NOTE: There is a slight mistake in the last equation for the GRU cell. It should instead be: s_t = (1 - z) * s_t-1 + z * h
Bidirectional RNN Indepth Intuition- Deep Learning Tutorial
Deep RNN
Definitions to Skim:

RNN
GRU
LSTM
BRNN
Requirements
General
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
Your files will be executed with numpy (version 1.15)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.4)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
Unless otherwise noted, you are not allowed to import any module except import numpy as np
