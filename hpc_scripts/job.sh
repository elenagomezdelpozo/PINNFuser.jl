#!/bin/bash -l
#SBATCH -J training1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem=4GB
#SBATCH -A plgsanomodeling2-cpu
#SBATCH -p plgrid
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#Load necessary modules (e.g., for GCC, MPI, etc.)
module load julia >/dev/null 2>&1

cd $HOME/CV_0D_models/PINNFuser.jl

#Run your program
stdbuf -o0 julia main/main.jl 
