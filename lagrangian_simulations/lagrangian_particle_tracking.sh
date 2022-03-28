#!/bin/bash
#$ -S /bin/bash
#$ -V
#$ -N lpt_2years
#$ -M federica.guerrini@polimi.it
#$ -m es

echo 'Running two years of Lagrangian particle simulations...'
cd ${HOME}/local_share/Scripts/
python3 lagrangian_particle_tracking.py -ye_st 2015 -mo_st 1 -da_st 1 -ye_end 2016 -mo_end 12 -da_end 31
echo 'Finished computation.'
