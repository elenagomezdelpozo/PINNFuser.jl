#!/bin/bash -l
#SBATCH -J test
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=1GB
#SBATCH -A plgsanomodeling2-cpu
#SBATCH -p plgrid
#SBATCH --output=logs/experiment_single_data_%a.out
#SBATCH --error=logs/experiment_single_data_%a.err
#SBATCH --array=1-7

#Load necessary modules (e.g., for GCC, MPI, etc.)
module load julia

cd $HOME/CV_0D_models/PINNFuser.jl

#Run your program
stdbuf -o0 julia main/Single_var_additions.jl $SLURM_ARRAY_TASK_ID single_data