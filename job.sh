#!/bin/bash
#SBATCH --partition=long                      # Ask for unkillable job
#SBATCH --cpus-per-task=12                    # Ask for 2 CPUs
#SBATCH --gres=gpu:rtx8000:1                         # Ask for 1 GPU
#SBATCH --mem=32G                             # Ask for 10 GB of RAM
#SBATCH --time=12:00:00                        # The job will run for 3 hours
#SBATCH -o /network/scratch/m/moksh.jain/logs/peano-%j.out  # Write the log on tmp1


module load python/3.9 cuda/11.8
export PYTHONUNBUFFERED=1

source $SCRATCH/minimo_env/bin/activate

cd ~/minimo/learning/

python bootstrap.py theory=groups job.name=repro