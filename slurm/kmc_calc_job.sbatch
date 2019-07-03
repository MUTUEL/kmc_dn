#!/bin/bash

#SBATCH -J kmc_calc_job
#SBATCH -c 20
#SBATCH -N 1
#SBATCH -p 50_procent_max_7_days
#SBATCH --time=6-00:00

module add miniconda

source activate kmc
cd /home/geurtsbj/bram/kmc_dn/

python examples/saveResults.py -i 0 -t 50 &
python examples/saveResults.py -i 50 -t 50 &
python examples/saveResults.py -i 100 -t 50 &
python examples/saveResults.py -i 150 -t 50 &
python examples/saveResults.py -i 200 -t 50 &
python examples/saveResults.py -i 250 -t 50 &
python examples/saveResults.py -i 300 -t 50 &
python examples/saveResults.py -i 350 -t 50 &
python examples/saveResults.py -i 400 -t 50 &
python examples/saveResults.py -i 450 -t 50 &
python examples/saveResults.py -i 500 -t 50 &
python examples/saveResults.py -i 550 -t 50 &
python examples/saveResults.py -i 600 -t 50
