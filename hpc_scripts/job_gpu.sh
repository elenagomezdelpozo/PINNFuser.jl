#!/bin/bash
#SBATCH --job-name=pinn_infuser
#SBATCH --partition=plgrid-gpu-v100
#SBATCH -A plgsanomodeling2-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=hpc_scripts/hpc_logs/training_gpu_1.out
#SBATCH --error=hpc_scripts/hpc_logs/training_gpu_1.err

module load julia >/dev/null 2>&1
module load CUDA/11.8 >/dev/null 2>&1
cd /net/afscra/people/plgelenagdelpozo/PINNFuser.jl

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_CUDA_MEMORY_POOL=cuda
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi
# Precompile once, then run
julia --threads=auto --project=. -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'  >/dev/null 2>&1
julia --threads=auto --project=. main/main.jl