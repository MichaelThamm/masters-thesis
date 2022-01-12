# Linear Induction Motor Optimization with Genetic Algorithm
Title: **_2D Hybrid Magnetic Field Model Performance Optimization for Linear Induction Motors_**

This repository includes the relevant Python code for my Ma.Sc. motor simulation and optimization software.
An attached [PDF](https://github.com/MichaelThamm/Masters/blob/main/ProjectExplanation_GitHub.pdf) file is provided in this repo that highlights the top-down foundation of the coding project.

Python modules required to run the code:

* math [documentation](https://docs.python.org/3/library/math.html)
* cmath [documentation](https://docs.python.org/3/library/cmath.html)
* timeit [documentation](https://docs.python.org/3/library/timeit.html)
* contextlib [documentation](https://docs.python.org/3/library/contextlib.html)
* collections [documentation](https://docs.python.org/3/library/collections.html)
* tkinter [documentation](https://docs.python.org/3/library/tk.html)
* json [documentation](https://docs.python.org/3/library/json.html)
* numpy [documentation](https://numpy.org/doc/)
* matplotlib [documentation](https://matplotlib.org/)
* inquirer [documentation](https://python-inquirer.readthedocs.io/en/latest/)

Note: [LIM_Platypus.py](https://github.com/MichaelThamm/Masters/blob/main/LIM_Platypus.py) is the main file that calls the other .py files in this order:

* [LIM_SlotPoleCalculation.py](https://github.com/MichaelThamm/Masters/blob/main/LIM_SlotPoleCalculation.py)
* [LIM_Grid.py](https://github.com/MichaelThamm/Masters/blob/main/LIM_Grid.py)
* [LIM_Compute.py](https://github.com/MichaelThamm/Masters/blob/main/LIM_Compute.py)
* [LIM_Show.py](https://github.com/MichaelThamm/Masters/blob/main/LIM_Show.py)
* [LIM_ShowFromJSON.py](https://github.com/MichaelThamm/Masters/blob/main/LIM_ShowFromJSON.py)
  *  Note: This file pulls the stored JSON results and creates a field plot without having to run the entire simulation

----------VERY IMPORTANT----------

Make sure to run LIM_Platypus.py from a terminal or the user input functionality will not work
You must also change to the correct directory that contains these files and execute the LIM_Platypus.py using "python LIM_Platypus.py"
