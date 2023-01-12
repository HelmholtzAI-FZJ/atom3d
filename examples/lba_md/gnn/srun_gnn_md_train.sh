#!/usr/bin/env bash
# `bash -x` for detailed Shell debugging

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --output=/p/project/hai_drug_qm/atom3d/examples/lba_md/gnn/output/%j.out
#SBATCH --error=/p/project/hai_drug_qm/atom3d/examples/lba_md/gnn/output/%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --account=hai_hmgu

source ../../../sc_venv_template/activate.sh
export PYTHONPATH=/p/project/hai_drug_qm/atom3d:$PYTHONPATH
time -p srun python -u train_multigpu.py 
