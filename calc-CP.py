import numpy as np
#import scipy as sp
#from scipy import linalg
import sys
from cclib.parser import ccopen

DATA = '/data/phys-prw17/phys1470/'
path = DATA + 'job_files/01-znpc-f6tcnnq-translate-z/'
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
        print('----------------------------')
        print ('molA HOMO-3: \t', nhomoA-3, molA.moenergies[0][nhomoA - 3])
        print ('molA HOMO-2: \t', nhomoA-2, molA.moenergies[0][nhomoA - 2])
        print ('molA HOMO-1: \t', nhomoA-1, molA.moenergies[0][nhomoA - 1])
        print ('molA HOMO: \t', nhomoA, molA.moenergies[0][nhomoA])
        print ('molA LUMO: \t', nhomoA + 1, molA.moenergies[0][nhomoA + 1])

        print('----------------------------')

        print ('molB HOMO: \t', nhomoB, molB.moenergies[0][nhomoB])
        print ('molB LUMO: \t', nhomoB + 1, molB.moenergies[0][nhomoB + 1])
        print ('molB LUMO+1: \t', nhomoB + 2, molB.moenergies[0][nhomoB + 2])
        print ('molB LUMO+2: \t', nhomoB + 3, molB.moenergies[0][nhomoB + 3])
        print ('molB LUMO+3: \t', nhomoB + 4, molB.moenergies[0][nhomoB + 4])
        
        print('----------------------------')

        print ('molAB HOMO-4: \t', nhomoAB -4, molAB.moenergies[0][nhomoAB -4])
        print ('molAB HOMO-3: \t', nhomoAB -3, molAB.moenergies[0][nhomoAB -3])
        print ('molAB HOMO-2: \t', nhomoAB -2, molAB.moenergies[0][nhomoAB -2])
        print ('molAB HOMO-1: \t', nhomoAB -1, molAB.moenergies[0][nhomoAB -1])
        print ('molAB HOMO: \t', nhomoAB, molAB.moenergies[0][nhomoAB])
        print ('molAB LUMO: \t', nhomoAB + 1, molAB.moenergies[0][nhomoAB + 1])
        print ('molAB LUMO+1: \t', nhomoAB + 2, molAB.moenergies[0][nhomoAB + 2])
        print ('molAB LUMO+2: \t', nhomoAB + 3, molAB.moenergies[0][nhomoAB + 3])
        print ('molAB LUMO+3: \t', nhomoAB + 4, molAB.moenergies[0][nhomoAB + 4])
        print ('molAB LUMO+5: \t', nhomoAB + 5, molAB.moenergies[0][nhomoAB + 5])

        print('----------------------------')

        print ("Gap in isolation: ", molB.moenergies[0][nhomoB + 1] - molA.moenergies[0][nhomoA])
        print ("Gap in pair: ", EvalsAB[nhomoAB + 1] - EvalsAB[nhomoAB])

        deg_homo = 1
        deg_lumo = 1

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

        # print ('PsiA_AB_BS = ',PsiA_AB_BS[nhomoA, nlumoB])
        # print ('PsiB_AB_BS = ',PsiB_AB_BS[nhomoA, nlumoB])
        # temp = np.dot(PsiA_AB_BS, np.diagflat(EvalsAB))
        # print ('PsiA_AB_BS dot E_lk = ', temp[nhomoA,nlumoB])
        # temp2 = PsiB_AB_BS.T
        # print ('PsiB_AB_BS^T = ',temp2[nhomoA, nlumoB])
        # temp3 = np.dot(temp, temp2)
        # print ('PsiA_AB_BS dot E_lk dot PsiA_AB_BS^T = ', temp3[nhomoA,nlumoB])

        # Calculate the matrix of transfer integrals
        JAB = np.dot(np.dot(PsiA_AB_BS, np.diagflat(EvalsAB)), PsiB_AB_BS.T)
        JAA = np.dot(np.dot(PsiA_AB_BS, np.diagflat(EvalsAB)), PsiA_AB_BS.T)
        JBB = np.dot(np.dot(PsiB_AB_BS, np.diagflat(EvalsAB)), PsiB_AB_BS.T)
        S = np.dot(PsiA_AB_BS, PsiB_AB_BS.T)

        # Symmetric Lowdin transformation
        J_eff = (JAB - 0.5 * (JAA + JBB) * S) / (1.0 - S ** 2)

        # print ('j_AB = ',JAB[nhomoA, nlumoB])
        # print ('j_AA = ',JAA[nhomoA, nlumoB])
        # print ('j_BB = ',JBB[nhomoA, nlumoB])
        # print ('S_AB = ',S[nhomoA, nlumoB])

        # Energy eigenvalues
        # eA_eff = 0.5*( (JAA + JBB) - 2 * JAB * S + (JAA - JBB) * np.sqrt(1 - S**2 ))/ ( 1 - S**2 )
        # eB_eff = 0.5*( (JAA + JBB) - 2 * JAB * S - (JAA - JBB) * np.sqrt(1 - S**2 ))/ ( 1 - S**2 )


        # Print the HOMO-HOMO and LUMO-LUMO coupling
        print('\n ---------------------------- \n')
        with open(path + jobtitle + '/' + jobtitle + '-CP-calc.txt', "w") as text_file:
                print ("Job title: ", jobtitle)
                print (f"Job title: {jobtitle}", file=text_file)
                if deg_homo == 1:
                        # print ("nHOMO A: ", nhomoA)
                        # print ("nHOMO B: ", nlumoB)
                        # print ("HOMO A: ", nhomoA, eA_eff[nhomoA,nhomoA])
                        # print (f"HOMO A: {eA_eff[nhomoA,nhomoA]}", file=text_file)
                        # print ("LUMO B: ", nlumoB, eB_eff[nlumoB,nlumoB])
                        # print (f"LUMO B: {eB_eff[nlumoB,nlumoB]}", file=text_file)
                        # print ("ESID HOMO-HOMO coupling", 0.5 * (EvalsAB[nhomoAB] - EvalsAB[nhomoAB - 1]))
                        # print (f"ESID HOMO-HOMO coupling {0.5 * (EvalsAB[nhomoAB] - EvalsAB[nhomoAB - 1])}", file=text_file)
                        # print ("HOMO-HOMO coupling: ", J_eff[nhomoA,nhomoB])
                        # print (f"HOMO-HOMO coupling: {J_eff[nhomoA,nhomoB]}", file=text_file)
                        print ("HOMO", nhomoA, "-LUMO", nlumoB," coupling: ", J_eff[nhomoA, nlumoB])
                        print (f"HOMO-LUMO coupling: {J_eff[nhomoA,nlumoB]}", file=text_file)
                if deg_lumo == 1:
                        # print ("ESID LUMO-LUMO coupling", 0.5 * (EvalsAB[nhomoAB + 2] - EvalsAB[nhomoAB + 1]))
                        # print (f"ESID LUMO-LUMO coupling {0.5 * (EvalsAB[nhomoAB + 2] - EvalsAB[nhomoAB + 1])}", file=text_file)
                        # print ("LUMO-LUMO coupling: ", J_eff[nlumoA,nlumoB])
                        # print (f"LUMO-LUMO coupling: {J_eff[nlumoA,nlumoB]}", file=text_file)
                        #print ("LUMO-HOMO coupling: ", J_eff[nlumoA,nhomoB])
                        #print (f"LUMO-HOMO coupling: {J_eff[nlumoA,nhomoB]}", file=text_file)
                        pass
                if deg_homo == 2:
                        # doubly degenerate
                        
                        # %H00=H_eff(nhomo_mon+nbasis,   nhomo_mon  );
                        # %H01=H_eff(nhomo_mon+nbasis,   nhomo_mon-1);
                        # %H10=H_eff(nhomo_mon+nbasis-1, nhomo_mon  );
                        # %H11=H_eff(nhomo_mon+nbasis-1, nhomo_mon-1);

                        # %L00=H_eff(nhomo_mon+nbasis+1, nhomo_mon+1);
                        # %L01=H_eff(nhomo_mon+nbasis+1, nhomo_mon);
                        # %L10=H_eff(nhomo_mon+nbasis,   nhomo_mon+1);
                        # %L11=H_eff(nhomo_mon+nbasis,   nhomo_mon  );

                        # %Doubly_degenerate_HOMO_coupling=(abs(H00)+abs(H01)+abs(H10)+abs(H11))/4
                        # %Doubly_degenerate_LUMO_coupling=(abs(L00)+abs(L01)+abs(L10)+abs(L11))/4
                        j0 = J_eff[nhomoA, nhomoB]
                        print ("HOMO-HOMO",nhomoA, nhomoB, "coupling: ", j0)
                        j1 = J_eff[nhomoA, nhomoB - 1]
                        print ("HOMO-HOMO-1 coupling: ", j1)
                        j2 = J_eff[nhomoA - 1, nhomoB]
                        print ("HOMO-1-HOMO coupling: ", j2)
                        j3 = J_eff[nhomoA - 1, nhomoB - 1]
                        print ("HOMO-1-HOMO-1 coupling: ", j3)
                        print('------------------------------')
                        print (' HOMO-HOMO RMS equals', np.sqrt(np.mean(np.square([j0, j3]))))
                        print('------------------------------')
                        
                        # deg_homo_J = J_eff[int(nhomoA - deg_homo + 1):int(nhomoA + 1), int(nhomoB - deg_homo + 1):int(nhomoB + 1)]
                        # print ("Degenerate HOMO-HOMO coupling: ", np.sqrt(np.mean(np.square(deg_homo_J))))
                        # print (f"Degenerate HOMO-HOMO coupling: {np.sqrt(np.mean(np.square(deg_homo_J)))}", file=text_file)
                if deg_lumo == 2:
                        j0 = J_eff[nlumoA, nlumoB]
                        print ("LUMO-LUMO",nlumoA, nlumoB, " coupling: ", j0)
                        j1 = J_eff[nlumoA, nlumoB + 1]
                        print ("LUMO-LUMO+1 coupling: ", j1)
                        j2 = J_eff[nlumoA + 1, nlumoB]
                        print ("LUMO+1-LUMO coupling: ", j2)
                        j3 = J_eff[nlumoA + 1, nlumoB + 1]
                        print ("LUMO+1-LUMO+1 coupling: ", j3)
                        print('------------------------------')
                        print ('LUMO-LUMO-RMS equals', np.sqrt(np.mean(np.square([j0, j3]))))
                        print('------------------------------')
                        # deg_lumo_J = J_eff[int(nlumoA):int(nlumoA + deg_lumo), int(nlumoB):int(nlumoB + deg_lumo)]
                        # print ("Degenerate LUMO-LUMO coupling: ", np.sqrt(np.mean(np.square(deg_lumo_J))))
                        # print (f"Degenerate LUMO-LUMO coupling: {np.sqrt(np.mean(np.square(deg_lumo_J)))}", file=text_file)

                if deg_homo == 4:
                        # molA HOMO
                        j0 = J_eff[nhomoA, nhomoB]
                        print ("HOMO-HOMO coupling: ", j0)
                        # j1 = J_eff[nhomoA, nhomoB - 1]
                        # j2 = J_eff[nhomoA, nhomoB - 2]
                        # j3 = J_eff[nhomoA, nhomoB - 3]
                        # molA HOMO-1
                        # j4 = J_eff[nhomoA-1, nhomoB]
                        j5 = J_eff[nhomoA-1, nhomoB - 1]
                        # j6 = J_eff[nhomoA-1, nhomoB - 2]
                        # j7 = J_eff[nhomoA-1, nhomoB - 3]
                        # molA HOMO-2
                        # j8 = J_eff[nhomoA-2, nhomoB]
                        # j9 = J_eff[nhomoA-2, nhomoB - 1]
                        j10 = J_eff[nhomoA-2, nhomoB - 2]
                        # j11 = J_eff[nhomoA-2, nhomoB - 3]
                        # molA HOMO-3
                        # j12 = J_eff[nhomoA-3, nhomoB]
                        # j13 = J_eff[nhomoA-3, nhomoB - 1]
                        # j14 = J_eff[nhomoA-3, nhomoB - 2]
                        j15 = J_eff[nhomoA-3, nhomoB - 3]

                        print('------------------------------')
                        # print (' HOMO-HOMO RMS equals', np.sqrt(np.mean(np.square([j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15]))))
                        print (' HOMO-HOMO RMS equals', np.sqrt(np.mean(np.square([j0, j5, j10, j15]))))
                        print('------------------------------')
                        
                        deg_homo_J = J_eff[int(nhomoA - deg_homo + 1):int(nhomoA + 1), int(nhomoB - deg_homo + 1):int(nhomoB + 1)]
                        print ("Degenerate HOMO-HOMO coupling: ", np.sqrt(np.mean(np.square(deg_homo_J))))
                        print (f"Degenerate HOMO-HOMO coupling: {np.sqrt(np.mean(np.square(deg_homo_J)))}", file=text_file)


                if deg_lumo == 4:
                        # molA HOMO
                        j0 = J_eff[nlumoA, nlumoB]
                        # print ("LUMO-LUMO coupling: ", j0)
                        j1 = J_eff[nlumoA, nlumoB + 1]
                        # print ("LUMO-LUMO+1 coupling: ", j1)
                        j2 = J_eff[nlumoA, nlumoB + 2]
                        # print ("LUMO-LUMO+2 coupling: ", j2)
                        j3 = J_eff[nlumoA, nlumoB + 3]
                        # print ("LUMO-LUMO+3 coupling: ", j3)
                        # molA HOMO-1
                        j4 = J_eff[nlumoA+1, nlumoB]
                        # print ("LUMO+1-LUMO coupling: ", j4)
                        j5 = J_eff[nlumoA+1, nlumoB + 1]
                        # print ("LUMO+1-LUMO+1 coupling: ", j5)
                        j6 = J_eff[nlumoA+1, nlumoB + 2]
                        # print ("LUMO+1-LUMO+2 coupling: ", j6)
                        j7 = J_eff[nlumoA+1, nlumoB + 3]
                        # print ("LUMO+1-LUMO+3 coupling: ", j7)
                        # molA HOMO-2
                        j8 = J_eff[nlumoA+2, nlumoB]
                        # print ("LUMO+2-LUMO coupling: ", j8)
                        j9 = J_eff[nlumoA+2, nlumoB + 1]
                        # print ("LUMO+2-LUMO+1 coupling: ", j9)
                        j10 = J_eff[nlumoA+2, nlumoB + 2]
                        # print ("LUMO+2-LUMO+2 coupling: ", j10)
                        j11 = J_eff[nlumoA+2, nlumoB + 3]
                        # print ("LUMO+2-LUMO+3 coupling: ", j11)
                        # molA HOMO-3
                        j12 = J_eff[nlumoA+3, nlumoB]
                        # print ("LUMO+3-LUMO coupling: ", j12)
                        j13 = J_eff[nlumoA+3, nlumoB + 1]
                        # print ("LUMO+3-LUMO+1 coupling: ", j13)
                        j14 = J_eff[nlumoA+3, nlumoB + 2]
                        # print ("LUMO+3-LUMO+2 coupling: ", j14)
                        j15 = J_eff[nlumoA+3, nlumoB + 3]
                        # print ("LUMO+3-LUMO+3 coupling: ", j15)
                        # print(j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15)

                        print('------------------------------')
                        # print (' LUMO-LUMO RMS equals', np.sqrt(np.mean(np.square([j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15]))))
                        print (' LUMO-LUMO RMS equals', np.sqrt(np.mean(np.square([j0, j5, j10, j15]))))
                        print('------------------------------')
                        deg_lumo_J = J_eff[int(nlumoA):int(nlumoA + deg_lumo), int(nlumoB):int(nlumoB + deg_lumo)]
                        print ("Degenerate LUMO-LUMO coupling: ", np.sqrt(np.mean(np.square(deg_lumo_J))))
                        print (f"Degenerate LUMO-LUMO coupling: {np.sqrt(np.mean(np.square(deg_lumo_J)))}", file=text_file)
CountPJ()

