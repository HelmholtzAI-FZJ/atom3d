#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/outputs/%j-gnn.out
#SBATCH --error=/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/outputs/%j-gnn.err
#SBATCH --time=01:00:00
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:4
#SBATCH --account=hai_hmgu

source $HOME/.bashrc

source ../../../sc_venv_template/activate.sh
CUDA_VISIBLE_DEVICES=0,1,2,3

srun python -u train_multigpu_smp_test.py --target_name "Ionization_Potential"