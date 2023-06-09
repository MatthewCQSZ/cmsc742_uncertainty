#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=runexp_reacher_0                               
#SBATCH --output=runexp_reacher_0.out
#SBATCH --error=runexp_reacher_0.err 
#SBATCH --time=2-00:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger                                        
#SBATCH --mem=10gb 
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:1

source  /fs/nexus-scratch/chqzhu/pets-test/bin/activate
cd /fs/nexus-scratch/chqzhu/pets-essential
python mbexp.py -env reacher -epinet -epi_coef 0 -logdir log_test_May16_epi_0002
wait                                         # must have at the end
