#!/bin/bash
#SBATCH --job-name="fairness"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32          # Number of tasks (MPI processes)
#SBATCH --cpus-per-task=1          # Number of threads per task (OMP threads)
#SBATCH --time=7-00:00:00
#SBATCH --mem=64gb
#SBATCH --output="fairness-%j.o"
#SBATCH --error="fairness-%j.e"
#SBATCH --mail-user=zhaok220@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account="khuriGrp"
#SBATCH --partition=medium

source /deac/csc/khuriGrp/software/zhaok220/bin/activate 

python3 /deac/csc/khuriGrp/zhaok220/ml_fairness/scripts/synthetic_test.py
slurm_mem_report -g

exit 0

