"""Author: Douwe Orij"""


import numpy as np
import sph_harm_functions as sph_harm
import os
import pyvista as pv
import pandas as pd
import itertools as it

def main(l_max,input,output):

   # Make new directory to save data
   os.makedirs(output,exist_ok=True)

   column_orb = ['orb_{}'.format(i) for i in range(0,(l_max+1)**2)]
   column_names = ['id', 'stl', 'sim', 'bub_num', 'time [s]', 'pos_x', 'pos_y', 'pos_z', 'l_max'] + column_orb

   for folder in os.listdir(input):
      map = os.path.join(input, folder)
      sname_pickle = os.path.join(output, folder + '.pkl')
      print('Currently working in: ', folder)

      # Create new dataframe
      df = pd.DataFrame(columns=column_names)

      for file in os.listdir(map):
         fname = os.path.join(map, file)
         print('Currently working on file ',file)

         df.loc[len(df)] = sph_harm_bubble(fname, l_max)

      # Save dataframe
      df.to_pickle(sname_pickle)

def sph_harm_bubble(fname, l_max):
   # Import stl file
   stl = pv.read(fname)

   # Get spherical harmonics from stl file
   weights, _ , _ = sph_harm.weights_from_stl(stl, rot=[0,0,0], l_max=l_max)   

   # Get relative path
   fdir = os.path.basename(os.path.dirname(fname))
   fname = os.path.basename(fname)
   fsave = os.path.join("data", "bubbles_stl", fdir, fname)
   
   # Get position of bubble
   pos = stl.center
   
   # Get bubble number
   bub_num = int(fname.split('_')[-1].split('.')[0])

   # Set identifier
   id = fdir + '_' + str(bub_num)
   
   # Get time and convert to seconds
   time = float(fname.split('_')[0].lstrip('F'))
   time *= 1e-5 # s
   
   # Save results in dataframe
   data = [id, fsave, fdir, bub_num, time, pos[0], pos[1], pos[2], l_max] + list(weights)

   return data
      
######################## START OF CODE ########################

# Set variables        
l_max = 14

# Set input and output folders
loc = os.path.dirname(os.path.realpath(__file__))
loc = os.path.join(loc, 'data')
input = os.path.join(loc,'bubbles_stl')
output = os.path.join(loc,'pickle_files_FT')

main(l_max, input, output)   
