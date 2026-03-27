#!/bin/bash -l
#SBATCH -J test
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:30:00
#SBATCH --mem=1GB
#SBATCH -A plgsanomodeling-cpu
#SBATCH -p plgrid
#SBATCH --output=logs/experiment_all_%a.out
#SBATCH --error=logs/experiment_all_%a.out
#SBATCH --array=1-7

#Load necessary modules (e.g., for GCC, MPI, etc.)
module load julia

cd $HOME/CV_0D_models/PINNFuser.jl

#Run your program
stdbuf -o0 julia main/All_var_additions.jl $SLURM_ARRAY_TASK_ID
