#!/bin/bash -l
#SBATCH -J training
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=4GB
#SBATCH -A plgsanomodeling2-cpu
#SBATCH -p plgrid
#SBATCH --output=hpc_scripts/logs/training_%a.out
#SBATCH --error=hpc_scripts/logs/training_%a.err
#SBATCH --array=1-7

#Load necessary modules (e.g., for GCC, MPI, etc.)
module load julia >/dev/null 2>&1

cd $HOME/CV_0D_models/PINNFuser.jl

#Run your program
stdbuf -o0 julia main/main.jl $SLURM_ARRAY_TASK_ID
