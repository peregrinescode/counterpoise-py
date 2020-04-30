#!/usr/bin/env python3
#coding:utf-8
"""
  Author:   peregrine<dot>warren<at>physics<dot>ox<dot>ac<dot>uk
  Purpose:  Generates Gaussian input files (.com) from geometry files (.xyz) of 
            two molecules for a Counterpoise calculation.
            This is a rewrite of the bash scripts in https://github.com/Jenny-Nelson-Group/counterpoise-J
  Created: 07/04/20
"""

import os
#import sys
import numpy as np
import pandas as pd
from inspect import cleandoc  # to ignore indents when printing to file
from scipy.spatial.transform import Rotation as R  # for rotating molecules in 3D


# Set variables for top of com file
nprocshared = 16
mem = '14GB'
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
        print(ghost_molB.to_string(header=False, index=False), file=text_file, end='\n\n')

    # Molecule B with ghost atoms of molecule A
    with open(dir + job_name + '-part2.com', "w") as text_file:
        print(cleandoc(
            f'''
            %nprocshared={nprocshared}
            %mem={mem}
            #p SCF(Tight,Conver=8) Integral(Grid=UltraFine) IOp(6/7=3) {method} nosymm

            CP calculation part 2 - mol B with ghost atoms of mol A

            0 1'''), file=text_file)
        ghost_molA = geom_molA.copy() # Turn molecule A into a ghost
        ghost_molA['atom'] = ghost_molA['atom'] + '-Bq' # Add -Bq to each atom, marking it as a ghost
        print(ghost_molA.to_string(header=False, index=False), file=text_file)
        print(geom_molB.to_string(header=False, index=False), file=text_file, end='\n\n')

    # Molecule AB - the pair/dimer
    with open(dir + job_name + '-pair.com', "w") as text_file:
        print(cleandoc(
            f'''
            %chk=dimer.chk
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

def rotate_molecule(mol_xyz, orgin=(0, 0), degrees=90):
    """rotate a molecule by degress deg in a clockwise direction around center cen."""
    # Make a np array out of coordinates
    p = np.array(mol_xyz[['x', 'y', 'z']])
    # Convert degrees to radians
    rad = np.deg2rad(degrees)
    # Create rotation
    r = R.from_quat([0, 0, np.sin(rad / 2), np.cos(rad / 2)])
    # print('Transformation: ', r.as_euler('xyz', degrees=True))
    # Apply the rotation to the molecule array
    p_rotated = r.apply(p)
    # Put rotation back into the geom style dataframe
    p_rotated_geom = mol_xyz.copy()
    p_rotated_geom[['x', 'y', 'z']] = pd.DataFrame(p_rotated, columns=['x', 'y', 'z'])
    return p_rotated_geom

# define paths
geoms = 'geom_files/'

# Read in molecules
znpc_xyz = geoms + 'ZnPc.xyz'
f4znpc_xyz = geoms + 'F4ZnPc.xyz'
f4znpc2_xyz = geoms + 'F4ZnPc-2.xyz'
f8znpc_xyz = geoms + 'F8ZnPc.xyz'
f16znpc_xyz = geoms + 'F16ZnPc.xyz'
f6_xyz = geoms + 'F6TCNNQ.xyz'
ethene_xyz = geoms + 'ethene.xyz'
znpc_geom = read_geom(znpc_xyz)
f4znpc_geom = read_geom(f4znpc_xyz)
f4znpc2_geom = read_geom(f4znpc2_xyz)
f8znpc_geom = read_geom(f8znpc_xyz)
f16znpc_geom = read_geom(f16znpc_xyz)
f6_geom = read_geom(f6_xyz)
ethene_geom = read_geom(ethene_xyz)

# Rotate a square and write a com file
#square_xzy = geoms + 'square.xyz'
#square_geom = read_geom(square_xzy)
#job_name = 'square-rotate'
#square_rot = rotate_molecule(square_geom, degrees=90)
#write_com(square_geom, job_name)

# 00-ethene
# for Z in np.linspace(3, 15, 25):
#         print (Z)
#         ethene_Z = ethene_geom.copy()  # restore orignal coordinates
#         ethene_Z = translate_molecule(ethene_Z, Z, 'z')
#         job_name = 'ethene-translateZ-' + str(Z).replace('.', 'p')
#         write_CP_com(ethene_Z, ethene_geom, job_name)
        

# Translate one molecule in Z-direction
for Z in np.linspace(0.5, 15, 30):
        print (Z)
        f4znpc2_Z = f4znpc2_geom.copy()  # restore orignal coordinates
        f4znpc2_Z = translate_molecule(f4znpc2_Z, Z, 'z')
        job_name = 'f4znpc2-f6tcnnq-translateZ-' + str(Z).replace('.', 'p')
        write_CP_com(f4znpc2_Z, f6_geom, job_name)

# Rotate one molecule around z:
# for theta in np.linspace(0, 90, num=10):
#         print (theta)
#         # Place one molecule away in Z-direction
#         f16znpc_Z = f16znpc_geom.copy()  # restore orignal coordinates
#         f16znpc_Z = translate_molecule(f16znpc_Z, 3.5, 'z')
#         f6tcnnq_theta = f6_geom.copy()  # restore orignal coordinates
#         f6tcnnq_theta = rotate_molecule(f6tcnnq_theta, degrees=theta)
#         job_name = 'f16znpc-f6tcnnq-rotateZ-' + str(theta).replace('.', 'p')
#         write_CP_com(f16znpc_Z, f6tcnnq_theta, job_name)

