import io
#import paramiko
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


#def load_data_cluster(name, cimat=True, user = 'est_posgrado_angel.mendez'):
#    '''
#    Params:
#    name: str
#        Name of the file to load
#    cimat: bool, If True, is with the CIMAT internet connection
#
#    Returns:
#    data
#    '''
#
#    main_path = '/home/est_posgrado_angel.mendez/'
#    key = paramiko.Ed25519Key.from_private_key_file('/Users/antoniomendez/.ssh/id_ed25519')
#
#    if cimat:
#        host = '10.10.25.20'
#        port = 22
#    else:
#        host = '148.207.185.31'
#        port = 2284
#
#    try:
#        transport = paramiko.Transport((host, port))
#        transport.connect(username=user, pkey=key)
#        sftp = paramiko.SFTPClient.from_transport(transport)
#        remote_path = main_path + name
#        print(remote_path)
#        with sftp.open(remote_path, 'rb') as remote_file:
#            memory_file = io.BytesIO(remote_file.read())
#
#        if name[-3:] == 'csv':
#            data = pd.read_csv(memory_file)
#        elif name[-3:] == 'npy':
#            data = np.load(memory_file, allow_pickle=True)
#        else:
#            print('Error: file not found')
#            sys.exit()
#        sftp.close()
#        transport.close()
#    except Exception as e:
#        print(f'Error loading file from cluster: {e}')
#        sys.exit()
#