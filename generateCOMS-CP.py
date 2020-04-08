#!/usr/bin/env python3
#coding:utf-8
"""
  Author:   peregrine.warren<at>physics.ox.ac.uk
  Purpose:  Generates Gaussian input files (.com) from geometry files (.xyz) of 
            two molecules for a Counterpoise calculation.
            This is a rewrite of the bash scripts in https://github.com/Jenny-Nelson-Group/counterpoise-J
  Created: 07/04/20
"""

import os
import numpy as np
import pandas as pd
from inspect import cleandoc # to ignore indents when printing to file
#import sys

# Set variables for top of com file
nprocshared = 1
mem = '500MB'
method = 'b3lyp/6-31g*'

def read_geom(file_in):
    """read .xzy to dataframe."""
    mol_xyz = pd.read_csv(file_in, sep=' ', skiprows=0, skipinitialspace=1, header=None, names=['atom', 'x', 'y', 'z'])
    return mol_xyz

def write_com(geom_in, com_name):
    """generate com file from geom file"""
    with open(com_name, "w") as text_file:
        print(cleandoc(
                f'''
                    %nprocshared={nprocshared}
                    %mem={mem}
                    #p SCF(Tight,Conver=8) Integral(Grid=UltraFine) IOp(6/7=3) {method} nosymm

                    autogen

                    0 1'''), file=text_file)
        print(geom_in.to_string(header=False, index=False), file=text_file)
    return

def write_CP_com(geom_molA, geom_molB, job_name):
    """generate com files for counterpoise calculation."""

    # Make directories
    try:
        os.makedirs('job_files/' + job_name, exist_ok=True)
        print(f"Directory job_files/{job_name} created successfully")
        dir = 'job_files/' + job_name + '/'
    except OSError:
        print(f"Directories job_files/{job_name} can not be created")

    # Molecule A with ghost atoms of molecule B
    with open(dir + job_name + '-part1.com', "w") as text_file:
        print(cleandoc(
            f'''
            %nprocshared={nprocshared}
            %mem={mem}
            #p SCF(Tight,Conver=8) Integral(Grid=UltraFine) IOp(6/7=3) {method} nosymm

            CP calculation part 1 - mol A with ghost atoms of mol B

            0 1'''), file=text_file)
        print(geom_molA.to_string(header=False, index=False), file=text_file)
        ghost_molB = geom_molB.copy() # Turn molecule B into a ghost
        ghost_molB['atom'] = ghost_molB['atom'] + '-Bq' # Add -Bq to each atom, marking it as a ghost
        print(ghost_molB.to_string(header=False, index=False), file=text_file)

    # Molecule B with ghost atoms of molecule A
    with open(dir + job_name + '-part2.com', "w") as text_file:
        print(cleandoc(
            f'''
            %nprocshared={nprocshared}
            %mem={mem}
            #p SCF(Tight,Conver=8) Integroal(Grid=UltraFine) IOp(6/7=3) {method} nosymm

            CP calculation part 2 - mol B with ghost atoms of mol A

            0 1'''), file=text_file)
        ghost_molA = geom_molA.copy() # Turn molecule A into a ghost
        ghost_molA['atom'] = ghost_molA['atom'] + '-Bq' # Add -Bq to each atom, marking it as a ghost
        print(ghost_molA.to_string(header=False, index=False), file=text_file)
        print(geom_molB.to_string(header=False, index=False), file=text_file)

    # Molecule AB - the pair/dimer
    with open(dir + job_name + '-pair.com', "w") as text_file:
        print(cleandoc(
            f'''
            %nprocshared={nprocshared}
            %mem={mem}
            #p SCF(Tight,Conver=8) Integral(Grid=UltraFine) IOp(6/7=3) {method} nosymm

            CP calculation part 3 - mol AB 

            0 1'''), file=text_file)
        print(geom_molA.to_string(header=False, index=False), file=text_file)
        print(geom_molB.to_string(header=False, index=False), file=text_file, end='\n\n') # add extra line break
        # The molecular pair / dimer calc has a Link1 (Gaussian restart) to separately perform IOp(3/33=1)
        print(cleandoc(
            f'''

            --Link1--
            %chk=dimer.chk
            %nprocshared={nprocshared}
            %mem={mem}
            #p geom(allcheck) guess(read,only) IOp(3/33=1) {method} nosymm

            '''), file=text_file)

    return

def translate_molecule(mol_xyz, d, dr):
    """translate a molecule in x, y or z by distance d in direction dr."""
    mol_xyz[dr] = mol_xyz[dr] + d
    return mol_xyz

def rotate_molecule(mol_xyz, d, dr, cen):
    """rotate a molecule by degress deg in a clockwise direction around center cen."""
    mol_xyz[dr] = mol_xyz[dr] + d
    return mol_xyz

# define paths
geoms = 'geom_files/'

# Read in molecules
znpc_xyz = geoms + 'ZnPc.xyz'
f6_xyz = geoms + 'F6TCNNQ.xyz'
znpc_geom = read_geom(znpc_xyz)
f6_geom = read_geom(f6_xyz)

# Write a com file
# job_name = 'znpc-f6tcnnq'
# write_CP_com(znpc_geom, f6_geom, job_name)

# Translate one molecule in Z-direction
for Z in np.arange(3,8,1):	
        print (Z)
        znpc_Z = znpc_geom.copy() # restore orignal coordinates
        znpc_Z = translate_molecule(znpc_Z, Z, 'z')
        job_name = 'znpc-f6tcnnq-translateZ-' + str(Z)
        write_CP_com(znpc_Z, f6_geom, job_name)

