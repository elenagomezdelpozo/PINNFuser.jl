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

module load Julia
cd /net/people/plgrid/plgelenagdelpozo/CV_0D_models/PINNFuser.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
