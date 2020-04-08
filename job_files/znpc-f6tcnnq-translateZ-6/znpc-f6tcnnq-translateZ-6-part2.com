%nprocshared=16
%mem=14GB
#p SCF(Tight,Conver=8) Integral(Grid=UltraFine) IOp(6/7=3) b3lyp/6-31g* nosymm

CP calculation part 2 - mol B with ghost atoms of mol A

0 1
C-Bq -2.998797 -0.204793  5.99961
 C-Bq -2.020837 -2.224928  5.99960
 C-Bq -4.077313 -1.190641  5.99970
 C-Bq -3.463060 -2.459329  5.99965
 C-Bq -5.465519 -1.064641  5.99981
 C-Bq -4.224856 -3.626502  5.99974
 C-Bq -6.223752 -2.232172  5.99986
 H-Bq -5.931333 -0.086669  5.99984
 C-Bq -5.611046 -3.497480  5.99984
 H-Bq -3.746260 -4.598253  5.99973
 H-Bq -7.306073 -2.168429  5.99992
 H-Bq -6.232200 -4.386106  5.99991
 C-Bq  2.224930 -2.020828  6.00003
 C-Bq  0.204794 -2.998798  5.99985
 C-Bq  2.459331 -3.463051  6.00037
 C-Bq  1.190643 -4.077304  6.00024
 C-Bq  3.626503 -4.224856  6.00075
 C-Bq  1.064644 -5.465510  6.00051
 C-Bq  3.497482 -5.611036  6.00102
 H-Bq  4.598255 -3.746250  6.00085
 C-Bq  2.232174 -6.223743  6.00090
 H-Bq  0.086680 -5.931335  6.00040
 H-Bq  4.386118 -6.232192  6.00131
 H-Bq  2.168441 -7.306065  6.00113
 C-Bq  2.020839  2.224929  5.99976
 C-Bq  2.998799  0.204793  5.99989
 C-Bq  3.463052  2.459330  5.99969
 C-Bq  4.077305  1.190642  5.99980
 C-Bq  4.224868  3.626502  5.99958
 C-Bq  5.465520  1.064632  5.99978
 C-Bq  5.611047  3.497471  5.99956
 H-Bq  3.746261  4.598254  5.99951
 C-Bq  6.223744  2.232174  5.99964
 H-Bq  5.931335  0.086670  5.99986
 H-Bq  6.232202  4.386107  5.99948
 H-Bq  7.306065  2.168430  5.99960
 C-Bq -0.204793  2.998788  5.99982
 C-Bq -2.224927  2.020839  5.99979
 C-Bq -1.190641  4.077304  6.00026
 C-Bq -2.459329  3.463051  6.00027
 C-Bq -1.064642  5.465510  6.00069
 C-Bq -3.626502  4.224857  6.00068
 C-Bq -2.232172  6.223743  6.00112
 H-Bq -0.086669  5.931334  6.00068
 C-Bq -3.497480  5.611037  6.00110
 H-Bq -4.598253  3.746261  6.00069
 H-Bq -2.168429  7.306064  6.00147
 H-Bq -4.386105  6.232202  6.00142
 N-Bq -1.109928 -3.192613  5.99971
 N-Bq  1.109930  3.192614  5.99979
 N-Bq  3.192615 -1.109929  6.00005
 N-Bq -3.192623  1.109930  5.99974
 N-Bq  0.871984 -1.801548  5.99968
 N-Bq  1.801549  0.871983  5.99992
 N-Bq -1.801547 -0.871983  5.99956
 N-Bq -0.871992  1.801549  5.99945
Zn-Bq  0.000000  0.000000  5.99971
C -1.579828 -0.946877 -0.000003
C -0.230859 -0.688683 -0.000006
C  0.230858  0.688684 -0.000007
C -0.777686  1.715832 -0.000007
C -2.100548  1.423385 -0.000003
C  0.777686 -1.715831 -0.000006
C  1.579828  0.946878 -0.000005
C  2.100548 -1.423384 -0.000004
C  2.601583 -0.071915 -0.000002
C  3.961282  0.226940  0.000004
C  4.493899  1.548531 -0.000018
C  4.986970 -0.762818  0.000032
N  5.021346  2.577704 -0.000042
N  5.886576 -1.489434  0.000063
C -2.601583  0.071917  0.000000
C -3.961282 -0.226940  0.000006
C -4.493897 -1.548533 -0.000015
C -4.986972  0.762816  0.000035
N -5.021339 -2.577708 -0.000039
N -5.886581  1.489428  0.000065
F  2.002612  2.206772  0.000000
F  2.971483 -2.430666 -0.000008
F  0.414974 -2.994748 -0.000012
F -0.414974  2.994750 -0.000013
F -2.971483  2.430668 -0.000007
F -2.002613 -2.206770  0.000003
