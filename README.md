# Linear Induction Motor Optimization with Genetic Algorithm
Title: **_2D Hybrid Magnetic Field Model Performance Optimization for Linear Induction Motors_**

This repository includes the relevant Python code for my Ma.Sc. motor simulation and optimization software.
An attached [PDF](https://github.com/MichaelThamm/Masters/blob/main/ProjectExplanation_GitHub.pdf) file is provided in this repo that highlights the top-down foundation of the coding project.

## Python modules required to run the code:

[Requirements.txt](https://github.com/MichaelThamm/Masters/blob/main/requirements.txt)

This can be used with *pip install -r requirements.txt* to install all requirements.

<p>&nbsp;</p>

Note: [LIM.Platypus.py](https://github.com/MichaelThamm/Masters/blob/main/LIM/Platypus.py) is the main file that calls the other .py files in this order:

* LIM.SlotPoleCalculation.py
* LIM.Grid.py
* LIM.Compute.py
* LIM.Show.py
* LIM.ShowFromJSON.py
  *  Note: This file pulls the stored JSON results and creates a field plot without having to run the entire simulation

## ----------VERY IMPORTANT----------

Make sure to run LIM.Platypus.py from a terminal or the user input functionality will not work
You must also change to the correct directory that contains these files and execute the LIM.Platypus.py using "python LIM.Platypus.py"

