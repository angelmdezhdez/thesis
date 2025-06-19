import pandas as pd
import numpy as np
import sys

def load_data(name, flag_dir = True):
    ''' 
    Params:
    name: str
        Name of the file to load
    flag_dir: bool, If True, the directory is set as the dir in CIMAT computer
        If False, the directory is set as the dir in Antonio's computer

    Returns:
    data
    '''
    if flag_dir:
        dir = '/home/user/Desktop/Datos/'
        if name[-3:] == 'csv':
            try: 
                data = pd.read_csv(dir + name)
            except:
                print('Error loading csv file')
                sys.exit()
        elif name[-3:] == 'npy':
            try:
                data = np.load(dir + name)
            except:
                print('Error loading numpy file')
                sys.exit()
        else:
            print('Error: file not found')
            sys.exit()
    else:
        dir = '/Users/antoniomendez/Desktop/Tesis/Datos/datos_limpios/'
        if name[-3:] == 'csv':
            try: 
                data = pd.read_csv(dir + name)
            except Exception as e:
                print(f'Error loading csv file: {e}')
                sys.exit()
        elif name[-3:] == 'npy':
            try:
                data = np.load(dir + name)
            except:
                print('Error loading numpy file')
                sys.exit()
        else:
            print('Error: file not found')
            sys.exit()
    return data