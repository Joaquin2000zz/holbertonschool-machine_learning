#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
names = ["apples", "bananas", "oranges", "peaches"]
persons = ['Farrah', 'Fred', 'Felicia']
colors = ["red", "yellow", "#ff8000", "#ffe5b4"]
n = len(fruit)
width = .50

fig, ax = plt.subplots()

aux = None

for i in range(n):
    if not i:
        ax.bar(persons, fruit[i], width=width, label=names[i], color=colors[i])
    else:
        ax.bar(persons, fruit[i], width=width, bottom=aux, label=names[i], color=colors[i])
    if isinstance(aux, type(None)):
        aux = fruit[i]
    else:
        aux += fruit[i]

ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.legend()
ax.set_ylim(0, 80)
plt.show()
