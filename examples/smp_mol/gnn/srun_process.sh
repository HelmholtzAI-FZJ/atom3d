#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --output=/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/outputs/Ionization_Potential/%j-gnn.out
#SBATCH --error=/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/outputs/Ionization_Potential/%j-gnn.err
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --account=atmlaml

source $HOME/.bashrc

source ../../../sc_juatom3d/activate.sh
export PYTHONPATH=/p/project/hai_drug_qm/atom3d:$PYTHONPATH

srun python -u train_multigpu.py --target_name "Ionization_Potential" 