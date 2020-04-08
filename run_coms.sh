#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:59:59
#SBATCH --job-name=part1
#SBATCH --partition=compute

module load gaussian09
g09 < job_files/znpc-f6tcnnq-translateZ-3/znpc-f6tcnnq-translateZ-3-part1.com > job_files/znpc-f6tcnnq-translateZ-3/znpc-f6tcnnq-translateZ-3-part1.log
