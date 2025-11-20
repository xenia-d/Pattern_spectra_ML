#!/bin/bash
#SBATCH --job-name=Pattern_Spectra_Classification
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=4GB

# remove all previously loaded modules
source $HOME/venvs/HTSMvenv_py310/bin/activate

# Create working directory in TMPDIR
mkdir $TMPDIR
cp -r /scratch/$USER/Pattern_spectra_ML $TMPDIR
cd $TMPDIR/Pattern_spectra_ML

# Create results directory in scratch
mkdir -p /scratch/$USER/Pattern_Spectra_ML_Results/job_${SLURM_JOBID}

# Print Python version
which python
python -c "import torch; print(torch.__version__)"

echo "Starting training..."
python -u leaf_classification.py 

# Move Saved Files (contains plots and best model) to scratch results folder
echo "Moving results to /scratch/$USER/Pattern_Spectra_ML_Results/job_${SLURM_JOBID}..."
# mv "$TMPDIR/Pattern_spectra_ML/Saved Files" "/scratch/$USER/Pattern_Spectra_ML_Results/job_${SLURM_JOBID}/results"

# Also copy SLURM output log
cp /scratch/$USER/Pattern_spectra_ML/slurm-${SLURM_JOBID}.out /scratch/$USER/Pattern_Spectra_ML_Results/job_${SLURM_JOBID}
echo "Training completed and results moved successfully."
