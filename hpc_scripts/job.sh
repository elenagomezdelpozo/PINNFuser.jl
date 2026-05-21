#!/bin/bash -l
#SBATCH -J training
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=10GB
#SBATCH -A plgsanomodeling2-cpu
#SBATCH -p plgrid
#SBATCH --output=hpc_scripts/hpc_logs/training_50_%a.out
#SBATCH --error=hpc_scripts/hpc_logs/training_50_%a.err
#SBATCH --array=1-7

#Load necessary modules (e.g., for GCC, MPI, etc.)
module load julia >/dev/null 2>&1
cd /net/afscra/people/plgelenagdelpozo/PINNFuser.jl

julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()' >/dev/null 2>&1

#Run your program
stdbuf -o0 julia --project=. main/main.jl $SLURM_ARRAY_TASK_ID
