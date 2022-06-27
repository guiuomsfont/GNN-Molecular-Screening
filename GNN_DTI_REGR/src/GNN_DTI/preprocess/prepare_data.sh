#!/bin/bash

# In this script we create the result directories, enter the different protein directories and do the reescorings and binding pocket selections for the interactions with each of their corresponding ligands. We do this for the active and for the inactive.

mkdir "DATA"

mkdir "DATA/KEYS"

mkdir "DATA/DATA"

cd "DUD-E"

for rec_dir in */
do
   cd $rec_dir
   
   echo Executing $rec_dir
   
   rec_name=$(basename $rec_dir)
   
   obabel receptor.pdb -O receptor.pdbqt
   
   
   obabel actives_final-redock.mol2 -O actives_final-redock.pdbqt
   ../../rf-score-4/rf-score ../../rf-score-4/pdbbind-2013-refined.rf receptor.pdbqt actives_final-redock.pdbqt > rescoring_list.txt
   rm actives_final-redock.pdbqt
   obabel actives_final-redock.mol2 -O actives_final-redock.pdb
   python3 ../../select_position_get_binding_pocket.py receptor.pdb actives_final-redock.pdb $rec_name "../../DATA/DATA/" "../../DATA/KEYS/" "actives"
   rm actives_final-redock.pdb
   rm rescoring_list.txt

   
   obabel decoys_final-redock.mol2 -O decoys_final-redock.pdbqt
   ../../rf-score-4/rf-score ../../rf-score-4/pdbbind-2013-refined.rf receptor.pdbqt decoys_final-redock.pdbqt > rescoring_list.txt
   rm decoys_final-redock.pdbqt   
   obabel decoys_final-redock.mol2 -O decoys_final-redock.pdb
   python3 ../../select_position_get_binding_pocket.py receptor.pdb decoys_final-redock.pdb  $rec_name "../../DATA/DATA/" "../../DATA/KEYS/" "decoys"
   rm decoys_final-redock.pdb
   rm rescoring_list.txt
   
   
   rm receptor.pdbqt
   
   cd ..

done

   
   
   
   
   
   
   
    


