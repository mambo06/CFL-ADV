#!/bin/bash

#SBATCH --account=a_xue_li 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4 
#SBATCH --mem=16G 
#SBATCH --job-name=ADV$1 
#SBATCH --partition=general 
#SBATCH --time=60:00:00
#SBATCH -o out_$1
#SBATCH -e error_$1

module load anaconda3
source activate /scratch/user/uqaginan/RQ3/

dataset=$1

list="scale" #,model_replacement,direction,gradient_ascent,targeted"
for item in $(echo $list | tr ',' ' ')
do
    echo "dataset: $dataset"
    python -W ignore adv_train.py -d $dataset -e 1

done