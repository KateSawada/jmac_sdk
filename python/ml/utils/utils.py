def read_txt(file_list):
    """Read .txt file list

    Arg:
        file_list (str): txt file filename

    Return:
        (list): list of read lines

    """
    with open(file_list, "r") as f:
        filenames = f.readlines()
    return [filename.replace("\n", "") for filename in filenames]
