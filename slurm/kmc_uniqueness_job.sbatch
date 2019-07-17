#!/bin/bash

#SBATCH -J kmc_vc_job
#SBATCH -c 20
#SBATCH -N 2
#SBATCH -p 50_procent_max_7_days
#SBATCH --time=6-00:00

module add miniconda

source activate kmc
cd /home/geurtsbj/bram/kmc_dn/

python thesis_indrek/generate_samples.py

python thesis_indrek/UniquenessValidation.py -i 0 -t 100 -s 10 -f uniquness_dumps &
python thesis_indrek/UniquenessValidation.py -i 10 -t 100 -s 10 -f uniquness_dumps &
python thesis_indrek/UniquenessValidation.py -i 20 -t 100 -s 10 -f uniquness_dumps &
python thesis_indrek/UniquenessValidation.py -i 30 -t 100 -s 10 -f uniquness_dumps &
python thesis_indrek/UniquenessValidation.py -i 40 -t 100 -s 10 -f uniquness_dumps &
python thesis_indrek/UniquenessValidation.py -i 50 -t 100 -s 10 -f uniquness_dumps &
python thesis_indrek/UniquenessValidation.py -i 60 -t 100 -s 10 -f uniquness_dumps &
python thesis_indrek/UniquenessValidation.py -i 70 -t 100 -s 10 -f uniquness_dumps &
python thesis_indrek/UniquenessValidation.py -i 80 -t 100 -s 10 -f uniquness_dumps &
python thesis_indrek/UniquenessValidation.py -i 90 -t 100 -s 10 -f uniquness_dumps &

python thesis_indrek/count_results.py -s 0 -t 400 -f uniquness_dumps
