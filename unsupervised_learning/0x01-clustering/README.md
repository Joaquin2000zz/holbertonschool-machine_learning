0x01. Clustering
 By: Alexa Orrico, Software Engineer at Holberton School
 Weight: 5
 Project will start Nov 14, 2022 12:00 AM, must end by Nov 18, 2022 12:00 AM
 was released at Nov 14, 2022 12:00 AM
 Manual QA review must be done (request it when you are done with the project)
 An auto review will be launched at the deadline


Resources
Read or watch:

Understanding K-means Clustering in Machine Learning
K-means clustering: how it works
How many clusters?
Bimodal distribution
Gaussian Mixture Model
EM algorithm: how it works
Expectation Maximization: how it works
Mixture Models 4: multivariate Gaussians
Mixture Models 5: how many Gaussians?
Gaussian Mixture Model (GMM) using Expectation Maximization (EM) Technique
What is Hierarchical Clustering?
Hierarchical Clustering
Definitions to skim:

Cluster analysis
K-means clustering
Multimodal distribution
Mixture_model
Expectation–maximization algorithm
Hierarchical clustering
Ward’s method
Cophenetic
References:

scikit-learn
Clustering
sklearn.cluster.KMeans
Gaussian mixture models
sklearn.mixture.GaussianMixture
scipy
scipy.cluster.hierarchy
scipy.cluster.hierarchy.linkage
scipy.cluster.hierarchy.fcluster
scipy.cluster.hierarchy.dendrogram
Learning Objectives
What is a multimodal distribution?
What is a cluster?
What is cluster analysis?
What is “soft” vs “hard” clustering?
What is K-means clustering?
What are mixture models?
What is a Gaussian Mixture Model (GMM)?
What is the Expectation-Maximization (EM) algorithm?
How to implement the EM algorithm for GMMs
What is cluster variance?
What is the mountain/elbow method?
What is the Bayesian Information Criterion?
How to determine the correct number of clusters
What is Hierarchical clustering?
What is Agglomerative clustering?
What is Ward’s method?
What is Cophenetic distance?
What is scikit-learn?
What is scipy?
Requirements
General
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
Your files will be executed with numpy (version 1.15), sklearn (version 0.21), and scipy (version 1.3)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.4)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
Unless otherwise noted, you are not allowed to import any module except import numpy as np
All your files must be executable
Your code should use the minimum number of operations
Installing Scikit-Learn 0.21.x
pip install --user scikit-learn==0.21
Installing Scipy 1.3.x
scipy should have already been installed with matplotlib and numpy, but just in case:

pip install --user scipy==1.3