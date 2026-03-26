#!/bin/bash -l
#SBATCH -J test
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=1GB
#SBATCH -A plgsanomodeling-cpu
#SBATCH -p plgrid
#SBATCH --output=./slurm-%j.out
#SBATCH --error=./slurm-%j.err

# Load necessary modules (e.g., for GCC, MPI, etc.)
module load julia

# Run your program
julia Single_var_additions.jl 1