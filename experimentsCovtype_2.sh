#!/bin/bash

#SBATCH --account=a_xue_li 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4 
#SBATCH --mem=16G 
#SBATCH --partition=general 
#SBATCH --time=60:00:00

# Set job name with dataset
job_name="ADV_$1"
scontrol update job $SLURM_JOB_ID name=$job_name

# Create output directory if it doesn't exist
outdir="logs"
mkdir -p $outdir

# Redirect output and error to files
exec 1> "${outdir}/out2_${1}.log"
exec 2> "${outdir}/error2_${1}.log"


module load anaconda3
source activate /scratch/user/uqaginan/RQ3/

dataset=$1
echo "Experiemnts on dataset: $dataset"

# python -W ignore adv_train.py -d $dataset -e 50 -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.25 -at scale -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.25 -at model_replacement -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.25 -at direction -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.25 -at gradient_ascent -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.25 -at targeted -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.5 -at scale -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.5 -at model_replacement -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.5 -at direction -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.5 -at gradient_ascent -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.5 -at targeted -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.75 -at scale -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.75 -at model_replacement -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.75 -at direction -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.75 -at gradient_ascent -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 1 -mc 0.75 -at targeted -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.25 -at scale -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.25 -at model_replacement -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.25 -at direction -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.25 -at gradient_ascent -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.25 -at targeted -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.5 -at scale -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.5 -at model_replacement -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.5 -at direction -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.5 -at gradient_ascent -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.5 -at targeted -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.75 -at scale -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.75 -at model_replacement -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.75 -at direction -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.75 -at gradient_ascent -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.25 -mc 0.75 -at targeted -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.25 -at scale -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.25 -at model_replacement -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.25 -at direction -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.25 -at gradient_ascent -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.25 -at targeted -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.5 -at scale -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.5 -at model_replacement -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.5 -at direction -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.5 -at gradient_ascent -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.5 -at targeted -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.75 -at scale -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.75 -at model_replacement -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.75 -at direction -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.75 -at gradient_ascent -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.5 -mc 0.75 -at targeted -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.25 -at scale -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.25 -at model_replacement -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.25 -at direction -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.25 -at gradient_ascent -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.25 -at targeted -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.5 -at scale -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.5 -at model_replacement -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.5 -at direction -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.5 -at gradient_ascent -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.5 -at targeted -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.75 -at scale -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.75 -at model_replacement -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.75 -at direction -c 6
# python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.75 -at gradient_ascent -c 6
python -W ignore adv_train.py -d $dataset -e 50 -rl 0.75 -mc 0.75 -at targeted -c 6
