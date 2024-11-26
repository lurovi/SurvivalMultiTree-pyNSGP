import bz2
import pickle
import _pickle as cPickle


# Saves the "data" with the "title" and adds the .pkl
# title is the path of the filename to create without extension, data is what you want to make persistent
# note that this method creates a new file from scratch, but it does not create new folders
# if you want to save this file into a directory that is different from the current directory, please make sure that
# all directories in the path have already been created in the correct order given by the hierarchical structure
# of the input path.
def full_pickle(title, data):
    pikd = open(title + '.pkl', 'wb')
    pickle.dump(data, pikd)
    pikd.close()

# loads and returns a pickled objects
# file is the path of the file to load, extension included
def loosen(file):
    pikd = open(file, 'rb')
    data = pickle.load(pikd)
    pikd.close()
    return data

# Pickle a file and then compress it into a file with extension
# title is the path of the filename to create without extension, data is what you want to make persistent
# note that this method creates a new file from scratch, but it does not create new folders
# if you want to save this file into a directory that is different from the current directory, please make sure that
# all directories in the path have already been created in the correct order given by the hierarchical structure
# of the input path.
def compress_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)

# Load any compressed pickle file
# file is the path of the file to load, extension included
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    return cPickle.load(data)

