#!/bin/bash

#SBATCH -J kmc_job
#SBATCH -c 20
#SBATCH -N 2
#SBATCH -p 50_procent_max_7_days
#SBATCH --time=6-00:00

module add miniconda

source activate kmc
cd /home/geurtsbj/bram/kmc_dn/

python thesis_indrek/voltage_search_tests.py -d 5 -i 0 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 5 -i 25 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 5 -i 50 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 5 -i 75 -t 25 &

python thesis_indrek/voltage_search_tests.py -d 10 -i 100 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 10 -i 125 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 10 -i 150 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 10 -i 175 -t 25 &

python thesis_indrek/voltage_search_tests.py -d 20 -i 200 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 20 -i 225 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 20 -i 250 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 20 -i 275 -t 25 &

python thesis_indrek/voltage_search_tests.py -d 30 -i 300 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 30 -i 325 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 30 -i 350 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 30 -i 375 -t 25 &

python thesis_indrek/voltage_search_tests.py -d 45 -i 400 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 45 -i 425 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 45 -i 450 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 45 -i 475 -t 25 &


python thesis_indrek/voltage_search_tests.py -d 60 -i 500 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 60 -i 525 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 60 -i 550 -t 25 &
python thesis_indrek/voltage_search_tests.py -d 60 -i 575 -t 25

