import numpy as np
#import scipy as sp
#from scipy import linalg
import sys
from cclib.parser import ccopen

DATA = '/data/phys-prw17/phys1470/'
path = DATA + 'job_files/02-znpc-f6tcnnq-rotate-z/'
#jobtitle = 'znpc-f6tcnnq-translateZ-3p0'
jobtitle = sys.argv[1]

#path = './job_files/'

MOLA_CP = path + jobtitle + '/' + jobtitle + "-part1.log"
MOLB_CP = path + jobtitle + '/' + jobtitle + "-part2.log"
MOLAB_CP = path + jobtitle + '/' + jobtitle + "-pair.log"


def CountPJ():
        # Read in molecule log files for counterpoise method. Requires IOp(6/7=3) in Gaussian header + ghost atoms to make up basis sets in individual com files
        # MOLA_CP=sys.argv[1]
        # MOLB_CP=sys.argv[2]
        # MOLAB_CP=sys.argv[3]

        # Open the log files
        molA_parser = ccopen("%s" % (MOLA_CP))
        molB_parser = ccopen("%s" % (MOLB_CP))
        molAB_parser = ccopen("%s" % (MOLAB_CP))

        # Parse the relevant data
        molA = molA_parser.parse()
        molB = molB_parser.parse()
        molAB = molAB_parser.parse()

        print ("Parsed...")

        # HOMO and LUMO index 
        nhomoA = molA.homos
        nhomoB = molB.homos
        nhomoAB = molAB.homos

        nlumoA = nhomoA + 1
        nlumoB = nhomoB + 1
        #print ("HOMO AB: ", nhomoAB)

        # Every basis set should have the same size (the size of the pair)
        if molA.nbasis != molB.nbasis:
                print ("Count of basis functions doesn't match. Failing.")

        # Get molecular orbitals
        MOsA = molA.mocoeffs[0]
        MOsB = molB.mocoeffs[0]
        MOsAB = molAB.mocoeffs[0]

        # Get eigenvalues of pair
        EvalsAB = molAB.moenergies[0]

        # print ("Energies: ", EvalsAB)
        print ("Gap: ", EvalsAB[nhomoAB + 1] - EvalsAB[nhomoAB])

        # Check degeneracy
        deg_homo=1
        deg_lumo=1

        for i in range(1, 10):
                if (np.absolute(EvalsAB[nhomoAB]-EvalsAB[nhomoAB-i])<0.005):
                        deg_homo+=1
                if (np.absolute(EvalsAB[nhomoAB+1+i]-EvalsAB[nhomoAB+1])<0.005):
                        deg_lumo+=1

        print ("Deg HOMO: ", deg_homo)
        print ("Deg LUMO: ", deg_lumo)

        # Find HOMO and LUMO from energy splitting in dimer
        #print ("ESID HOMO-HOMO coupling", 0.5 * (EvalsAB[nhomoAB] - EvalsAB[nhomoAB - 1]))
        #print ("ESID LUMO-LUMO coupling", 0.5 * (EvalsAB[nhomoAB + 2] - EvalsAB[nhomoAB + 1]))

        # Calculate the molecular orbitals of A and B in the AB basis set
        SAB = molAB.aooverlaps
        MolAB_Pro = (np.dot(MOsAB, SAB)).T
        PsiA_AB_BS = np.dot(MOsA, MolAB_Pro)
        PsiB_AB_BS = np.dot(MOsB, MolAB_Pro)

        # Calculate the matrix of transfer integrals
        JAB = np.dot(np.dot(PsiA_AB_BS, np.diagflat(EvalsAB)), PsiB_AB_BS.T)
        JAA = np.dot(np.dot(PsiA_AB_BS, np.diagflat(EvalsAB)), PsiA_AB_BS.T)
        JBB = np.dot(np.dot(PsiB_AB_BS, np.diagflat(EvalsAB)), PsiB_AB_BS.T)
        S = np.dot(PsiA_AB_BS, PsiB_AB_BS.T)

        # Symmetric Lowdin transformation
        J_eff = (JAB - 0.5 * (JAA + JBB) * S) / (1.0 - S ** 2)

        # Energy eigenvalues
        eA_eff = 0.5*( (JAA + JBB) - 2 * JAB * S + (JAA - JBB) * np.sqrt(1 - S**2 ))/ ( 1 - S**2 )
        eB_eff = 0.5*( (JAA + JBB) - 2 * JAB * S - (JAA - JBB) * np.sqrt(1 - S**2 ))/ ( 1 - S**2 )


        # Print the HOMO-HOMO and LUMO-LUMO coupling
        print('\n ---------------------------- \n')
        with open(path + jobtitle + '/' + jobtitle + '-CP-calc.txt', "w") as text_file:
                print ("Job title: ", jobtitle)
                print (f"Job title: {jobtitle}", file=text_file)
                if deg_homo == 1:
                        print ("HOMO A: ", eA_eff[nhomoA,nhomoA])
                        print (f"HOMO A: {eA_eff[nhomoA,nhomoA]}", file=text_file)
                        print ("LUMO B: ", eB_eff[nlumoB,nlumoB])
                        print (f"LUMO B: {eB_eff[nlumoB,nlumoB]}", file=text_file)
                        # print ("ESID HOMO-HOMO coupling", 0.5 * (EvalsAB[nhomoAB] - EvalsAB[nhomoAB - 1]))
                        # print (f"ESID HOMO-HOMO coupling {0.5 * (EvalsAB[nhomoAB] - EvalsAB[nhomoAB - 1])}", file=text_file)
                        #print ("HOMO-HOMO coupling: ", J_eff[nhomoA,nhomoB])
                        #print (f"HOMO-HOMO coupling: {J_eff[nhomoA,nhomoB]}", file=text_file)
                        print ("HOMO-LUMO coupling: ", J_eff[nhomoA, nlumoB])
                        print (f"HOMO-LUMO coupling: {J_eff[nhomoA,nlumoB]}", file=text_file)
                if deg_lumo == 1:
                        # print ("ESID LUMO-LUMO coupling", 0.5 * (EvalsAB[nhomoAB + 2] - EvalsAB[nhomoAB + 1]))
                        #print (f"ESID LUMO-LUMO coupling {0.5 * (EvalsAB[nhomoAB + 2] - EvalsAB[nhomoAB + 1])}", file=text_file)
                        #print ("LUMO-LUMO coupling: ", J_eff[nlumoA,nlumoB])
                        #print (f"LUMO-LUMO coupling: {J_eff[nlumoA,nlumoB]}", file=text_file)
                        #print ("LUMO-HOMO coupling: ", J_eff[nlumoA,nhomoB])
                        # print (f"LUMO-HOMO coupling: {J_eff[nlumoA,nhomoB]}", file=text_file)
                        pass
                if deg_homo == 2:
                        j0 = J_eff[nhomoA, nhomoB]
                        print ("HOMO-HOMO coupling: ", j0)
                        j1 = J_eff[nhomoA, nhomoB - 1]
                        print ("HOMO-HOMO-1 coupling: ", j1)
                        j2 = J_eff[nhomoA - 1, nhomoB]
                        print ("HOMO-1-HOMO coupling: ", j2)
                        j3 = J_eff[nhomoA - 1, nhomoB - 1]
                        print ("HOMO-1-HOMO-1 coupling: ", j3)
                        print('------------------------------')
                        print (' HOMO-HOMO RMS equals', np.sqrt(np.mean(np.square([j0, j1, j2, j3]))))
                        print('------------------------------')
                        
                        deg_homo_J = J_eff[int(nhomoA - deg_homo + 1):int(nhomoA + 1), int(nhomoB - deg_homo + 1):int(nhomoB + 1)]
                        print ("Degenerate HOMO-HOMO coupling: ", np.sqrt(np.mean(np.square(deg_homo_J))))
                        print (f"Degenerate HOMO-HOMO coupling: {np.sqrt(np.mean(np.square(deg_homo_J)))}", file=text_file)
                if deg_lumo == 2:
                        j0 = J_eff[nlumoA, nlumoB]
                        print ("LUMO-LUMO coupling: ", j0)
                        j1 = J_eff[nlumoA, nlumoB + 1]
                        print ("LUMO-LUMO+1 coupling: ", j1)
                        j2 = J_eff[nlumoA + 1, nlumoB]
                        print ("LUMO+1-LUMO coupling: ", j2)
                        j3 = J_eff[nlumoA + 1, nlumoB + 1]
                        print ("LUMO+1-LUMO+1 coupling: ", j3)
                        print('------------------------------')
                        print ('LUMO-LUMO-RMS equals', np.sqrt(np.mean(np.square([j0, j1, j2, j3]))))
                        print('------------------------------')
                        deg_lumo_J = J_eff[int(nlumoA):int(nlumoA + deg_lumo), int(nlumoB):int(nlumoB + deg_lumo)]
                        print ("Degenerate LUMO-LUMO coupling: ", np.sqrt(np.mean(np.square(deg_lumo_J))))
                        print (f"Degenerate LUMO-LUMO coupling: {np.sqrt(np.mean(np.square(deg_lumo_J)))}", file=text_file)

CountPJ()

