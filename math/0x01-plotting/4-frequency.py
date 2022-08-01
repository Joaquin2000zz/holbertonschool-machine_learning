#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, ax = plt.subplots()
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title("Project A")
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.hist(student_grades, edgecolor="black", bins=range(0, 101, 10))
plt.show()
