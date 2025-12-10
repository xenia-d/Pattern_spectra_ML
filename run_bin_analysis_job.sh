#!/bin/bash
#SBATCH --job-name=Pattern_Spectra_Bin_Analysis-Mondial-deletion
#SBATCH --time=20:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=8GB

source $HOME/venvs/HTSMvenv_py310/bin/activate

# Create working directory in TMPDIR
cp -r /scratch/$USER/Pattern_spectra_ML $TMPDIR
cd $TMPDIR/Pattern_spectra_ML

# Create results directory in scratch
mkdir -p /scratch/$USER/Pattern_Spectra_ML_Results/job_${SLURM_JOBID}

echo "Starting training..."
python -u Feature_Extraction/run_bin_analysis.py --variant Mondial --combo R_B --eval_type deletion

# Move Saved Files to scratch results folder
echo "Moving results to /scratch/$USER/Pattern_Spectra_ML_Results/job_${SLURM_JOBID}..."
mv "$TMPDIR/Pattern_spectra_ML/Saved_Results" "/scratch/$USER/Pattern_Spectra_ML_Results/job_${SLURM_JOBID}/results"

# Also copy SLURM output log
cp /scratch/$USER/Pattern_spectra_ML/slurm-${SLURM_JOBID}.out /scratch/$USER/Pattern_Spectra_ML_Results/job_${SLURM_JOBID}
echo "Training completed and results moved successfully."
