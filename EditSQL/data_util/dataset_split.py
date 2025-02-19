""" Utility functions for loading and processing ATIS data.
"""
import os
import pickle
import json

class DatasetSplit:
    """Stores a split of the ATIS dataset.

    Attributes:
        examples (list of Interaction): Stores the examples in the split.
    """
    def __init__(self, processed_filename, raw_filename, load_function):
        if os.path.exists(processed_filename):
            print("Loading preprocessed data from " + processed_filename)
            with open(processed_filename, 'rb') as infile:
                self.examples = pickle.load(infile)
        else:
            print("Loading raw data from " +
                raw_filename +
                " and NOT writing to " +
                processed_filename)

            infile = open(raw_filename, 'rb')

            if raw_filename[-3:]=='pkl':
                examples_from_file = pickle.load(infile)
            else:
                examples_from_file = json.load(infile)
                
            #print(examples_from_file)
            assert isinstance(examples_from_file, list), raw_filename + \
                " does not contain a list of examples"
            infile.close()

            self.examples = []
            for example in examples_from_file:
                obj, keep = load_function(example)

                if keep:
                    self.examples.append(obj)

            print("Loaded " + str(len(self.examples)) + " examples")
            #outfile = open(processed_filename, 'wb')
            #pickle.dump(self.examples, outfile)
            #outfile.close()

    def get_ex_properties(self, function):
        """ Applies some function to the examples in the dataset.

        Inputs:
            function: (lambda Interaction -> T): Function to apply to all
                examples.

        Returns
            list of the return value of the function
        """
        elems = []
        for example in self.examples:
            elems.append(function(example))
        return elems
