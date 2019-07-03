#!/bin/bash

#SBATCH -J kmc_vc_job
#SBATCH -c 20
#SBATCH -N 2
#SBATCH -p 50_procent_max_7_days
#SBATCH --time=6-00:00

module add miniconda

source activate kmc
cd /home/geurtsbj/bram/kmc_dn/

python examples/generate_samples.py

python examples/voltage_search_tests.py -d 30 -i 0 -m vc -v 4 -p 30DOPTRY1 -f sample1.kmc &
python examples/voltage_search_tests.py -d 30 -i 15 -m vc -v 5 -p 30DOPTRY1 -f sample1.kmc &

python examples/voltage_search_tests.py -d 30 -i 100 -m vc -v 4 -p 30DOPTRY2 -f sample2.kmc &
python examples/voltage_search_tests.py -d 30 -i 115 -m vc -v 5 -p 30DOPTRY2 -f sample2.kmc &

python examples/voltage_search_tests.py -d 30 -i 200 -m vc -v 4 -p 30DOPTRY3 -f sample3.kmc &
python examples/voltage_search_tests.py -d 30 -i 215 -m vc -v 5 -p 30DOPTRY3 -f sample3.kmc &

python examples/voltage_search_tests.py -d 30 -i 300 -m vc -v 4 -p 30DOPTRY4 -f sample4.kmc &
python examples/voltage_search_tests.py -d 30 -i 315 -m vc -v 5 -p 30DOPTRY4 -f sample4.kmc &

python examples/voltage_search_tests.py -i 400 -m vc -v 4 -p 20DOPTRY1 -f sample5.kmc &
python examples/voltage_search_tests.py -i 415 -m vc -v 5 -p 20DOPTRY1 -f sample5.kmc &

python examples/voltage_search_tests.py -i 500 -m vc -v 4 -p 20DOPTRY2 -f sample6.kmc &
python examples/voltage_search_tests.py -i 515 -m vc -v 5 -p 20DOPTRY2 -f sample6.kmc &

python examples/voltage_search_tests.py -i 600 -m vc -v 4 -p 20DOPTRY3 -f sample7.kmc &
python examples/voltage_search_tests.py -i 615 -m vc -v 5 -p 20DOPTRY3 -f sample7.kmc &

python examples/voltage_search_tests.py -i 700 -m vc -v 4 -p 20DOPTRY4 -f sample8.kmc &
python examples/voltage_search_tests.py -i 715 -m vc -v 5 -p 20DOPTRY4 -f sample8.kmc &

python examples/voltage_search_tests.py -i 800 -m vc -v 4 -p 10DOPTRY1 -f sample9.kmc &
python examples/voltage_search_tests.py -i 815 -m vc -v 5 -p 10DOPTRY1 -f sample9.kmc &

python examples/voltage_search_tests.py -i 900 -m vc -v 4 -p 10DOPTRY2 -f sample10.kmc &
python examples/voltage_search_tests.py -i 915 -m vc -v 5 -p 10DOPTRY2 -f sample10.kmc &

python examples/voltage_search_tests.py -i 1000 -m vc -v 4 -p 10DOPTRY3 -f sample11.kmc &
python examples/voltage_search_tests.py -i 1015 -m vc -v 5 -p 10DOPTRY3 -f sample11.kmc &

python examples/voltage_search_tests.py -i 1100 -m vc -v 4 -p 10DOPTRY4 -f sample12.kmc &
python examples/voltage_search_tests.py -i 1115 -m vc -v 5 -p 10DOPTRY4 -f sample12.kmc 