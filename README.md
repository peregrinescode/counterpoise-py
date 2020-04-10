# counterpoise-py

For calculating the transfer integral between two molecules using the counterpoise method. There are two scripts:

** 1. generateCOMS-CP.py**

> This script creates three Gaussian input files (.com) of: molecule A with ghost atoms of molecule B, molecule B with ghost atoms of molecule A, and the pair of molecules with no ghost atoms (moleculeAB). The script can also perform translation and rotation from the initial geometry files (.xyz)

**2)	calc-CP.py**

> After running the .com files produced in the first script through Gaussian, this program performs the counterpoise calculation on the Gaussian output files (.log).

Code based on: [https://github.com/peregrinescode/counterpoise-J](https://github.com/peregrinescode/counterpoise-J)