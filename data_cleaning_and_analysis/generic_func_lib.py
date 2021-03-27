# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:33:21 2021

@author: bcyk5
"""


def save_pickles(fListIn,NListIn,outLoc):
        ###pickles the outfile and saves it to a dir
    import pickle
    
    #wrkDir = os.getcwd()
    #'relative/path/to/file/you/want'
    #os.chdir('')
    fileNames = NListIn
    vars2Save = fListIn
    for i,fileName in enumerate(fileNames):
        fileName = fileName + '.pckl'
        path = outLoc + '/' + fileName
        with open(path, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(vars2Save[i], f)
        f.close()
        
        
        
def load_one_pickled(fileName, path):
    import pickle
    fileName = path + "\\" +  fileName
    with open(fileName,'rb') as f:
        outName = pickle.load(f)
    
    return outName


def back_up_in_file_tree(path: str, num: int) -> str:
    """ want a generic function which returns the path of the folder 
        above the input path (or num levels above)
    

    Parameters
    ----------
    path : str
        The file path to be modified.
    num : int
        number of times to back up (or levels to go up in the file tree).

    Returns
    -------
    path: str
        the path after the backup operation preformed

    """
    import os
    
    for _ in range(num):
        path = os.path.dirname(path)
    
    
    return path
