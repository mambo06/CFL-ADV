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
exec 1> "${outdir}/out_${1}.log"
exec 2> "${outdir}/error_${1}.log"


module load anaconda3
source activate /scratch/user/uqaginan/RQ3/

dataset=$1
echo "Experiemnts on dataset: $dataset"

attackType="scale,model_replacement,gradient_ascent,targeted"
malClient="0.25,0.5,0.75"
randomLevel="1,0.25,0.5,0.75"

echo "python -W ignore adv_train.py -d $dataset -e 50 -c 6"
python -W ignore adv_train.py -d $dataset -e 50 -c 6

for rl in $(echo $randomLevel | tr ',' ' ')
do
	for mc in $(echo $malClient | tr ',' ' ')
	do
		for at in $(echo $attackType | tr ',' ' ')
		do
			echo "python -W ignore adv_train.py -d $dataset -e 50 -rl $rl -mc $mc -at $at -c 6"
			python -W ignore adv_train.py -d $dataset -e 50 -rl $rl -mc $mc -at $at -c 6
		done

	done

done
