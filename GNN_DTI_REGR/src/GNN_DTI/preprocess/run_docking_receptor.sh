#!/bin/bash

# This script is used to run the docking with smina for each of the proteins and their corresponding ligands.

#Enter DUD-E dataset
cd "DUD-E"

#List directories
for rec_dir in */
do
   echo Computing "$rec_dir" receptor
   
   #Enter receptor dir
   cd "$rec_dir"

   # Run Smina for Autodocking
   echo Running smina for every ligand

   echo ../../smina.static -r receptor.pdb -l actives_final.mol2.gz --autobox_ligand receptor.pdb -o actives_final-redock.mol2 --cpu 8
   ../../smina.static -r receptor.pdb -l actives_final.mol2.gz --autobox_ligand receptor.pdb -o actives_final-redock.mol2 --cpu 8
   
   echo ../../smina.static -r receptor.pdb -l decoys_final.mol2.gz --autobox_ligand receptor.pdb -o decoys_final-redock.mol2 --cpu 8
   ../../smina.static -r receptor.pdb -l decoys_final.mol2.gz --autobox_ligand receptor.pdb -o decoys_final-redock.mol2 --cpu 8

   cd ..

done


