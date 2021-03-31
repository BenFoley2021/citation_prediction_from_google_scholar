# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:26:29 2021
script to copy the latest authorList scraper into a selectd dirs

https://datatofish.com/command-prompt-python/

current problems:
    can run the script from command, but it dies when this program closes. also
    doesn't pop up in a seperate window'

@author: bcyk5
"""
import os
from shutil import copyfile
import time
import random

def create_dirs(dir_name_base: str, dir_name_range: list, file_names: list) -> None:
    """ creates dirs with required files to run the author_list_scrapper
    
    Parameters
    ----------
    dir_name_base : Str
        The main part of the dir name, eg. "authorList_V2-7_"
    dir_name_range : List of Ints
        The range of intergers to add to the dir_base_name to create new dirs
    file_names : List of Strs
        Files to be copied from the current dir into the newly made dirs

    Returns
    -------
    None.

    """
    
    cwd = os.getcwd()
    for i in range(dir_name_range[0], dir_name_range[1]):
        if os.path.isdir(cwd + "//" + dir_name_base + "_" + str(i)) == False:
            os.mkdir(cwd + "//" + dir_name_base + "_" + str(i))
            os.mkdir(cwd + "//" + dir_name_base + "_" + str(i) + "//" + "outputs") # creation of this hardcoded for now
            for file_name in file_names:
                copyfile(file_name, cwd + "//" + dir_name_base + "_" + str(i) + "//" +  file_name)
    return None
    
    
def get_dirs(dirc_range: list, file_name: str):
    """ Copies file_name from the cwd into subfolders and runs file_name
        
        The purpose is to spawn a cmd window running file_name(.py) in each selected folder.
        Assumes file_name is a .py script. The file is copied into all subdirectories
        which meet the conditions in is_targeted_dir(), then creates a cmd process
        in that subdir to execute file_name.
    
    Parameters
    ----------
    dirc_range : List of Ints
        Gives the numbering of the folders to select.
    file_name : Str
        A python file which is to be copied into selected folders and executed.

    Returns
    -------
    dircs: list of Str
        List of all directories for which the action has been performed.
    """
    
    def execute_file(file_name: str, dirc: str) -> None:
        """ Executes a python file by starting a new cmd in the dir that the
            file to be executed is located in.
        
        Parameters
        ----------
        file_name : Str
            The name of the file to be executed
        dirc: Str
            Name of the directory where file_name is located.
    
        Returns
        -------
        None.
        """
        
        os.chdir(dirc)
        # os.system('start cmd /k echo Hello, World!') # example of how to open a new cmd from cmd
        command = 'start cmd /k ' + "python " + file_name          #'"Your Command Prompt Command"'
        #os.system('cmd /k "Your Command Prompt Command"') general syntax for running cmd from python script
        os.system(command)
        os.chdir(base_dir)
        return None
    
    def is_targeted_dir():
        """ Decides whether the directory is to be selected for later use
        
            Operates on dirc from the loop structure, depending on its name returns
            true or false. The main purpose of this func is too provide a seperate
            container for housing the logic which decides if the dir is to be used.

        Returns
        -------
        bool
            Whether or not dirc will be used for subsequent functions in the loop.
        """
        
        ### logic to decide if this is a dir we want
        # some if statements...
        if "authorList_V2-7_" not in dirc:
            return False
        
        if int(dirc.split("_")[-1]) >= dirc_range[0] and \
            int(dirc.split("_")[-1]) <= dirc_range[1]:
            return True
        
    """ The main loop of the get_dirs function places the file in any dir
        which meets the criteria in is_targeted_dir, then spawns
         a new cmd window running that file in the dir. It then waits an
         amount of time before moving on
    """
    base_dir = os.getcwd()
    dircs = []
    for dirc in os.listdir(os.getcwd()):
        if os.path.isfile(dirc) == False:
            if is_targeted_dir() == True:
                copyfile(file_name, dirc)
                execute_file(file_name, dirc)
                dircs.append(dirc)
                time.sleep(random.randint(100, 900)) # spacing out start times so everything doesn't start at once. This is also in the hopes that it makes the scrapper hards to detect
                
    return dircs


if __name__ == "__main__":
    """ User selects dirs to be created or used to execute the scrapper script
    """
    
    dirc_range = [13,20]
    file_name = "author_list_scrapper.py"
    dirs_done = get_dirs(dirc_range, file_name)



