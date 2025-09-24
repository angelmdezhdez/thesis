import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

def npy_loader_drive(path:str) -> np.ndarray:
    """
    Load a .npy file from a given path in Google Drive.

    Parameters:
    path (str): The file path to the .npy file.

    Returns:
    np.ndarray: The loaded numpy array.
    """
    try:
        data = np.load('/Users/antoniomendez/Library/CloudStorage/GoogleDrive-angel.mendez@cimat.mx/Mi unidad/' + path, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the cosine distance between two vectors.

    Parameters:
    a (np.ndarray): First vector.
    b (np.ndarray): Second vector.

    Returns:
    float: Cosine distance between the two vectors.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Input arrays must be one-dimensional.")
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 1.0  # Maximum distance if one vector is zero
    
    cosine_similarity = dot_product / (norm_a * norm_b)
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance


def which_day_is_it(date:str)->str:
    """This functions receives a date in the format 'YYYY-MM-DD' and returns the day of the week in Spanish."""
    days = {'Monday':'Lunes', 'Tuesday':'Martes', 'Wednesday':'Miércoles', 'Thursday':'Jueves', 'Friday':'Viernes', 'Saturday':'Sábado', 'Sunday':'Domingo'}
    day = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%A')
    return days[day]