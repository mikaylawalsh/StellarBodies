#!/bin/bash
#SBATCH --time=010:00:00
# Request use of 1 core and 8GB of memory on 1 node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=60G

# Submit request to gpu partition and request 1 gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=80G

# Job Name
#SBATCH -J STELLAR

# SLURM output (*.out) and error (*.err) file names
# Use '%x' for Job Name,'%A' for array-job ID, '%j' for job ID and '%a' for task ID`
#SBATCH -e %x-%j.err
#SBATCH -o %x-%j.out

# Notify user if job fails or ends (uncomment and add your email address to use)
#SBATCH --mail-user=mikayla_walsh@brown.edu
#SBATCH --mail-type=FAIL,END

#********************
# COMMANDS TO EXECUTE
#********************
# load desired modules (change to suit your particular needs)
# module load python/3.9.0
# module load gcc/10.2 
# activate virtual environment
source /users/mwalsh16/data/mwalsh16/mvts_env/bin/activate

python main.py