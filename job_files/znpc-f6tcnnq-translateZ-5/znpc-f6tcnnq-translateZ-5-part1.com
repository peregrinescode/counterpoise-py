%nprocshared=16
%mem=14GB
#p SCF(Tight,Conver=8) Integral(Grid=UltraFine) IOp(6/7=3) b3lyp/6-31g* nosymm

CP calculation part 1 - mol A with ghost atoms of mol B

0 1
C -2.998797 -0.204793  4.99961
 C -2.020837 -2.224928  4.99960
 C -4.077313 -1.190641  4.99970
 C -3.463060 -2.459329  4.99965
 C -5.465519 -1.064641  4.99981
 C -4.224856 -3.626502  4.99974
 C -6.223752 -2.232172  4.99986
 H -5.931333 -0.086669  4.99984
 C -5.611046 -3.497480  4.99984
 H -3.746260 -4.598253  4.99973
 H -7.306073 -2.168429  4.99992
 H -6.232200 -4.386106  4.99991
 C  2.224930 -2.020828  5.00003
 C  0.204794 -2.998798  4.99985
 C  2.459331 -3.463051  5.00037
 C  1.190643 -4.077304  5.00024
 C  3.626503 -4.224856  5.00075
 C  1.064644 -5.465510  5.00051
 C  3.497482 -5.611036  5.00102
 H  4.598255 -3.746250  5.00085
 C  2.232174 -6.223743  5.00090
 H  0.086680 -5.931335  5.00040
 H  4.386118 -6.232192  5.00131
 H  2.168441 -7.306065  5.00113
 C  2.020839  2.224929  4.99976
 C  2.998799  0.204793  4.99989
 C  3.463052  2.459330  4.99969
 C  4.077305  1.190642  4.99980
 C  4.224868  3.626502  4.99958
 C  5.465520  1.064632  4.99978
 C  5.611047  3.497471  4.99956
 H  3.746261  4.598254  4.99951
 C  6.223744  2.232174  4.99964
 H  5.931335  0.086670  4.99986
 H  6.232202  4.386107  4.99948
 H  7.306065  2.168430  4.99960
 C -0.204793  2.998788  4.99982
 C -2.224927  2.020839  4.99979
 C -1.190641  4.077304  5.00026
 C -2.459329  3.463051  5.00027
 C -1.064642  5.465510  5.00069
 C -3.626502  4.224857  5.00068
 C -2.232172  6.223743  5.00112
 H -0.086669  5.931334  5.00068
 C -3.497480  5.611037  5.00110
 H -4.598253  3.746261  5.00069
 H -2.168429  7.306064  5.00147
 H -4.386105  6.232202  5.00142
 N -1.109928 -3.192613  4.99971
 N  1.109930  3.192614  4.99979
 N  3.192615 -1.109929  5.00005
 N -3.192623  1.109930  4.99974
 N  0.871984 -1.801548  4.99968
 N  1.801549  0.871983  4.99992
 N -1.801547 -0.871983  4.99956
 N -0.871992  1.801549  4.99945
Zn  0.000000  0.000000  4.99971
C-Bq -1.579828 -0.946877 -0.000003
C-Bq -0.230859 -0.688683 -0.000006
C-Bq  0.230858  0.688684 -0.000007
C-Bq -0.777686  1.715832 -0.000007
C-Bq -2.100548  1.423385 -0.000003
C-Bq  0.777686 -1.715831 -0.000006
C-Bq  1.579828  0.946878 -0.000005
C-Bq  2.100548 -1.423384 -0.000004
C-Bq  2.601583 -0.071915 -0.000002
C-Bq  3.961282  0.226940  0.000004
C-Bq  4.493899  1.548531 -0.000018
C-Bq  4.986970 -0.762818  0.000032
N-Bq  5.021346  2.577704 -0.000042
N-Bq  5.886576 -1.489434  0.000063
C-Bq -2.601583  0.071917  0.000000
C-Bq -3.961282 -0.226940  0.000006
C-Bq -4.493897 -1.548533 -0.000015
C-Bq -4.986972  0.762816  0.000035
N-Bq -5.021339 -2.577708 -0.000039
N-Bq -5.886581  1.489428  0.000065
F-Bq  2.002612  2.206772  0.000000
F-Bq  2.971483 -2.430666 -0.000008
F-Bq  0.414974 -2.994748 -0.000012
F-Bq -0.414974  2.994750 -0.000013
F-Bq -2.971483  2.430668 -0.000007
F-Bq -2.002613 -2.206770  0.000003
