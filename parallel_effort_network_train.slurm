#!/bin/bash
#SBATCH --job-name=array_parallelized_motion
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=p.bendiksen001@umb.edu # Where to send mail
##SBATCH -p Intel6240,Intel6248,DGXA100
#SBATCH -p DGXA100
#SBATCH -A pi_funda.durupinarbabur
#SBATCH --qos=scavenger
##SBATCH -w chimera12,chimera13
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
##SBATCH --gres=gpu:1
#SBATCH -export=NONE # Export no environment variables
#SBATCH --mem=20G
#SBATCH -t 1-23:59:59
#SBATCH --output=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/arrays/%A-%a.out
#SBATCH --error=/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/arrays/%A-%a.err
#SBATCH --array=5

. /etc/profile

# check cpu number per task, should be equal to -n
echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

eval "$(conda shell.bash hook)"
conda activate motion-similarity

python src/main.py $SLURM_ARRAY_TASK_ID
echo "Finish Run"

