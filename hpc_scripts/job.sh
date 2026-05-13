#!/bin/bash -l
#SBATCH -J test
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=4GB
#SBATCH -A plgsanomodeling2-cpu
#SBATCH -p plgrid
#SBATCH --output=logs/experiment_%j.out
#SBATCH --error=logs/experiment_%j.err

#Load necessary modules (e.g., for GCC, MPI, etc.)
module load julia

cd $HOME/CV_0D_models/PINNFuser.jl

#Run your program
stdbuf -o0 julia main/main.jl 
