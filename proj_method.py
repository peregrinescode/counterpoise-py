'''
Changes made: - update to python3
              - update cclib.parser
'''

import numpy as np
import scipy as sp
import sys
from scipy import linalg
from cclib.parser import ccopen

DATA = '/data/phys-prw17/phys1470/'
path = DATA + 'job_files/10-znpc-znpc-translate-z/'
#jobtitle = 'znpc-f6tcnnq-translateZ-3p0'
jobtitle = sys.argv[1]


#MOLA_CP = path + jobtitle + '/' + jobtitle + "-part1.log"
#MOLB_CP = path + jobtitle + '/' + jobtitle + "-part2.log"
MOLAB_CP = path + jobtitle + '/' + jobtitle + "-pair.log"

MOLA_CP = DATA + 'job_files/optimisations/znpc-opt-SP.log'
MOLB_CP = DATA + 'job_files/optimisations/znpc-opt-SP.log'

def ProJ():
    # Read in molecule log files for projective method. Requires iop(3/33=1,6/7=3) in Gaussian header for calculation on each molecule + the pair
    #MOLA_proj=sys.argv[1]
    #MOLB_proj=sys.argv[2]
    #Degeneracy_HOMO=int(sys.argv[4])
    #Degeneracy_LUMO=int(sys.argv[5])    # =0 for non-degenerate, 2,3 etc for doubly, triply etc

    # Open the log files
    molA_parser = ccopen("%s" % (MOLA_CP))
    molB_parser = ccopen("%s" % (MOLB_CP))
    molAB_parser = ccopen("%s" % (MOLAB_CP))
    
    # Parse the relevant data
    molA = molA_parser.parse()
    molB = molB_parser.parse()
    molAB = molAB_parser.parse()

    print ("Parsed...")

    # Size of basis sets
    nbasisA = molA.nbasis
    nbasisB = molB.nbasis
    nbasisAB = molAB.nbasis

    print ("nbasisA: ", nbasisA)
    print ("nbasisB: ", nbasisB)
    print ("nbasisAB: ", nbasisAB)

    Nbasis = nbasisA + nbasisB

    # Position of HOMO
    nhomoA = molA.homos
    nhomoB = molB.homos
    nhomoAB = molAB.homos


    # Get molecular orbitals. Need the transpose for our purposes.
    MOsA=(molA.mocoeffs[0]).T
    MOsB=(molB.mocoeffs[0]).T
    MOsAB=(molAB.mocoeffs[0]).T

    # Get eigenvalues of pair
    EvalsAB = molAB.moenergies[0]
    
    # Check degeneracy
    deg_homo=1
    deg_lumo=1

    for i in range(1, 10):
        if (np.absolute(EvalsAB[nhomoAB] - EvalsAB[nhomoAB - i]) < 0.005):
            deg_homo+=1
        if (np.absolute(EvalsAB[nhomoAB + 1 + i] - EvalsAB[nhomoAB + 1]) < 0.005):
            deg_lumo+=1

    print ("Deg HOMO: ", deg_homo)
    print ("Deg LUMO: ", deg_lumo)    

    # Get overlaps. These are symmetric so transpose not required in this case
    SA = molA.aooverlaps
    SB = molB.aooverlaps
    SAB = molAB.aooverlaps

    # Set up matrices for MOs and S
    MOs = np.zeros((Nbasis, Nbasis))
    S = np.zeros((Nbasis, Nbasis))

    # First N/2 columns correspond to MO of molA
    MOs[:nbasisA, :nbasisA] = MOsA
    #MOs[:nbasisA, :MOsA.shape[1]] = MOsA
    
    # Second N/2 columns correspond to MO of molB
    MOs[nbasisA:Nbasis, nbasisA:Nbasis] = MOsB
    #MOs[nbasisA:Nbasis, MOsA.shape[1]:MOsA.shape[1] + MOsB.shape[1]] = MOsB
    
    # Same for overlaps:
    S[0:nbasisA, 0:nbasisA] = SA
    S[nbasisA:Nbasis, nbasisA:Nbasis] = SB

    # Calculate upper diagonal matrix D, such that S=D.T*D for Lowdin orthogonalisation.
    D=sp.linalg.cholesky(S) 
    Dpair=sp.linalg.cholesky(SAB)

    # Orthogonalise MOs matrix and MOsAB matrix
    MOsorth = np.dot(D, MOs)
    MOspairorth=np.dot(Dpair,MOsAB)

    # Calculate the Fock matrix
    B = np.dot(MOsorth.T, MOspairorth)
    Evals = np.diagflat(EvalsAB)
    F = np.dot(np.dot(B, Evals), B.T)


    # Output the HOMO-HOMO and LUMO-LUMO coupling elements fromt the Fock matrix
    if deg_homo == 1:
        print ("HOMO-HOMO coupling: ", F[int(nhomoB + nbasisA), int(nhomoA)])

    if deg_lumo == 1:
        print ("LUMO-LUMO coupling: ", F[int(nhomoB + nbasisA + 1), int(nhomoA + 1)])

    # Degeneracies
    # If orbitals are degenerate, take root mean square.
    # e.g. RMS of HOMO-HOMO, HOMO-HOMO-1, HOMO-1-HOMO and HOMO-1-HOMO-1
    if deg_homo == 2:
        j0 = F[int(nhomoB + nbasisA), int(nhomoA)]
        print ("HOMO-HOMO coupling: ", j0)
        j1 = F[int(nhomoB + nbasisA - 1), int(nhomoA)]
        print ("HOMO-HOMO-1 coupling: ", j1)
        j2 = F[int(nhomoB + nbasisA), int(nhomoA - 1)]
        print ("HOMO-1-HOMO coupling: ", j2)
        j3 = F[int(nhomoB + nbasisA - 1), int(nhomoA - 1)]
        print ("HOMO-1-HOMO-1 coupling: ", j3)
        print('------------------------------')
        print (' HOMO-HOMO RMS equals', np.sqrt(np.mean(np.square([j0, j1, j2, j3]))))
        print('------------------------------')

        #Degeneracy_HOMO = deg_homo
        #F_deg_HOMO = F[int(nhomoB + nbasisA - Degeneracy_HOMO + 1):int(nhomoB + nbasisA + 1), int(nhomoA - Degeneracy_HOMO):int(nhomoA)]
        #F_deg_HOMO = F[int(nhomoB + nbasisA - deg_homo): int(nhomoB + nbasisA), int(nhomoA - deg_homo):int(nhomoA)]
        # print(F_deg_HOMO)
        # print ("HOMO-HOMO coupling", (np.sum(np.absolute(F_deg_HOMO**2))) / deg_homo**2) # Strange way to get RMS?
        #print ("Degenerate HOMO-HOMO coupling: ", np.sqrt(np.mean(np.square(F_deg_HOMO))))

    if deg_lumo == 2:
        j0 = F[int(nhomoB + nbasisA + 1), int(nhomoA + 1)]
        print ("LUMO-LUMO coupling: ", j0)
        j1 = F[int(nhomoB + nbasisA + 1), int(nhomoA + 1 + 1)]
        print ("LUMO-LUMO+1 coupling: ", j1)
        j2 = F[int(nhomoB + nbasisA + 1 + 1), int(nhomoA + 1)]
        print ("LUMO+1-LUMO coupling: ", j2)
        j3 = F[int(nhomoB + nbasisA + 1 + 1), int(nhomoA + 1 + 1)]
        print ("LUMO+1-LUMO+1 coupling: ", j3)
        print('------------------------------')
        print ('LUMO-LUMO-RMS equals', np.sqrt(np.mean(np.square([j0, j1, j2, j3]))))
        print('------------------------------')

        #print ("LUMO-LUMO coupling: ", F[int(nhomoB + nbasisA + 1), int(nhomoA + 1)])
        #F_deg_LUMO = F[int(nhomoB + nbasisA):int(nhomoB + nbasisA + deg_lumo - 1), int(nhomoA + 1):int(nhomoA + 1 + deg_lumo - 1)]
        #print(F_deg_LUMO)
        ## print ("LUMO-LUMO coupling", (np.sum(np.absolute(F_deg_LUMO**2))) / deg_lumo**2)
        #print ("Degenerate LUMO-LUMO coupling: ", np.sqrt(np.mean(np.square(F_deg_LUMO))))
        pass
    
if __name__ == '__main__':
    ProJ()