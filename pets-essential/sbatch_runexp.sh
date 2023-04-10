#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=runexp_halfcheetah                                
#SBATCH --output=runexp_halfcheetah.out
#SBATCH --error=runexp_halfcheetah.err 
#SBATCH --time=2-00:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger                                        
#SBATCH --mem=10gb 
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:1

source  /fs/nexus-scratch/chqzhu/pets-test/bin/activate
cd /fs/nexus-scratch/chqzhu/pets-essential
python mbexp.py -env halfcheetah
wait                                         # must have at the end
