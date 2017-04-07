#########################################################################################################################

# THIS CODE (parsers.py) IS TAKEN NEARLY IN ENTIRETY FROM THE SOURCE CODE FROM THE PROBABILISTIC GRAPH MODELS FOR PYTHON 
# SOURCE CODE. SEE: https://github.com/pgmpy/pgmpy for the PGMPY package. I'm very grateful to them for their parsers! 

#########################################################################################################################




import re
from string import Template

import numpy
from pyparsing import Word, alphanums, Suppress, Optional, CharsNotIn, Group, nums, ZeroOrMore, OneOrMore,\
    cppStyleComment, printables
try:
    from lxml import etree
except ImportError:
    try:
        import xml.etree.ElementTree as etree
    except ImportError:
        # try:
        #    import xml.etree.cElementTree as etree
        #    commented out because xml.etree.cElementTree is giving errors with dictionary attributes
        print("Failed to import ElementTree from any known place")
        


import numpy as np
class XMLBIFReader(object):
    """
    Base class for reading network file in XMLBIF format.
    """

    def __init__(self, path=None, string=None):
        """
        Initialisation of XMLBIFReader object.

        Parameters
        ----------
        path : file or str
            File of XMLBIF data
        string : str
            String of XMLBIF data

        Examples
        --------
        # xmlbif_test.xml is the file present in
        # http://www.cs.cmu.edu/~fgcozman/Research/InterchangeFormat/
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        """
        if path:
            self.network = etree.ElementTree(file=path).getroot().find('NETWORK')
        elif string:
            self.network = etree.fromstring(string).find('NETWORK')
        else:
            raise ValueError("Must specify either path or string")
        self.network_name = self.network.find('NAME').text
        self.variables = self.get_variables()
        self.variable_parents = self.get_parents()
        self.edge_list = self.get_edges()
        self.variable_states = self.get_states()
        self.variable_CPD = self.get_cpd()
        self.variable_property = self.get_property()

    def get_variables(self):
        """
        Returns list of variables of the network

        Examples
        --------
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_variables()
        ['light-on', 'bowel-problem', 'dog-out', 'hear-bark', 'family-out']
        """
        variables = [variable.find('NAME').text for variable in self.network.findall('VARIABLE')]
        return variables


    def get_edges(self):
        """
        Returns the edges of the network

        Examples
        --------
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_edges()
        [['family-out', 'light-on'],
         ['family-out', 'dog-out'],
         ['bowel-problem', 'dog-out'],
         ['dog-out', 'hear-bark']]
        """
        edge_list = [[value, key] for key in self.variable_parents
                     for value in self.variable_parents[key]]
        return edge_list


    def get_states(self):
        """
        Returns the states of variables present in the network

        Examples
        --------
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_states()
        {'bowel-problem': ['true', 'false'],
         'dog-out': ['true', 'false'],
         'family-out': ['true', 'false'],
         'hear-bark': ['true', 'false'],
         'light-on': ['true', 'false']}
        """
        variable_states = {variable.find('NAME').text: [outcome.text for outcome in variable.findall('OUTCOME')]
                           for variable in self.network.findall('VARIABLE')}
        return variable_states


    def get_parents(self):
        """
        Returns the parents of the variables present in the network

        Examples
        --------
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_parents()
        {'bowel-problem': [],
         'dog-out': ['family-out', 'bowel-problem'],
         'family-out': [],
         'hear-bark': ['dog-out'],
         'light-on': ['family-out']}
        """
        variable_parents = {definition.find('FOR').text: [edge.text for edge in definition.findall('GIVEN')]
                            for definition in self.network.findall('DEFINITION')}
        return variable_parents


    def get_cpd(self):
        """
        Returns the CPD of the variables present in the network

        Examples
        --------
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_cpd()
        {'bowel-problem': array([[ 0.01],
                                 [ 0.99]]),
         'dog-out': array([[ 0.99,  0.01,  0.97,  0.03],
                           [ 0.9 ,  0.1 ,  0.3 ,  0.7 ]]),
         'family-out': array([[ 0.15],
                              [ 0.85]]),
         'hear-bark': array([[ 0.7 ,  0.3 ],
                             [ 0.01,  0.99]]),
         'light-on': array([[ 0.6 ,  0.4 ],
                            [ 0.05,  0.95]])}
        """
        variable_CPD = {definition.find('FOR').text: list(map(float, table.text.split()))
                        for definition in self.network.findall('DEFINITION')
                        for table in definition.findall('TABLE')}
        for variable in variable_CPD:
            arr = np.array(variable_CPD[variable])
            arr = arr.reshape((len(self.variable_states[variable]),
                               arr.size // len(self.variable_states[variable])), order='F')
            variable_CPD[variable] = arr
        return variable_CPD


    def get_property(self):
        """
        Returns the property of the variable

        Examples
        --------
        >>> reader = XMLBIF.XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_property()
        {'bowel-problem': ['position = (190, 69)'],
         'dog-out': ['position = (155, 165)'],
         'family-out': ['position = (112, 69)'],
         'hear-bark': ['position = (154, 241)'],
         'light-on': ['position = (73, 165)']}
        """
        variable_property = {variable.find('NAME').text: [property.text for property in variable.findall('PROPERTY')]
                             for variable in self.network.findall('VARIABLE')}
        return variable_property













class BIFReader(object):

    """
    Base class for reading network file in bif format
    """

    def __init__(self, path=None, string=None):
        """
        Initialisation of BifReader object

        Parameters
        ----------------
        path : file or str
                File of bif data
        string : str
                String of bif data
        Examples
        -----------------
        # dog-problem.bif file is present at
        # http://www.cs.cmu.edu/~javabayes/Examples/DogProblem/dog-problem.bif
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader = BIFReader("bif_test.bif")
        <pgmpy.readwrite.BIF.BIFReader object at 0x7f2375621cf8>
        """
        if path:
            with open(path, 'r') as network:
                self.network = network.read()

        elif string:
            self.network = string

        else:
            raise ValueError("Must specify either path or string")

        if '"' in self.network:
            # Replacing quotes by spaces to remove case sensitivity like:
            # "Dog-Problem" and Dog-problem
            # or "true""false" and "true" "false" and true false
            self.network = self.network.replace('"', ' ')

        if '/*' in self.network or '//' in self.network:
            self.network = cppStyleComment.suppress().transformString(self.network)  # removing comments from the file

        self.name_expr, self.state_expr, self.property_expr = self.get_variable_grammar()
        self.probability_expr, self.cpd_expr = self.get_probability_grammar()
        self.network_name = self.get_network_name()
        self.variable_names = self.get_variables()
        self.variable_states = self.get_states()
        self.variable_properties = self.get_property()
        self.variable_parents = self.get_parents()
        self.variable_cpds = self.get_cpd()
        self.variable_edges = self.get_edges()

    def get_variable_grammar(self):
        """
         A method that returns variable grammar
        """
        # Defining a expression for valid word
        word_expr = Word(alphanums + '_' + '-')
        word_expr2 = Word(initChars=printables, excludeChars=['{', '}', ',', ' '])
        name_expr = Suppress('variable') + word_expr + Suppress('{')
        state_expr = ZeroOrMore(word_expr2 + Optional(Suppress(",")))
        # Defining a variable state expression
        variable_state_expr = Suppress('type') + Suppress(word_expr) + Suppress('[') + Suppress(Word(nums)) + \
            Suppress(']') + Suppress('{') + Group(state_expr) + Suppress('}') + Suppress(';')
        # variable states is of the form type description [args] { val1, val2 }; (comma may or may not be present)

        property_expr = Suppress('property') + CharsNotIn(';') + Suppress(';')  # Creating a expr to find property

        return name_expr, variable_state_expr, property_expr


    def get_probability_grammar(self):
        """
        A method that returns probability grammar
        """
        # Creating valid word expression for probability, it is of the format
        # wor1 | var2 , var3 or var1 var2 var3 or simply var
        word_expr = Word(alphanums + '-' + '_') + Suppress(Optional("|")) + Suppress(Optional(","))
        word_expr2 = Word(initChars=printables, excludeChars=[',', ')', ' ', '(']) + Suppress(Optional(","))
        # creating an expression for valid numbers, of the format
        # 1.00 or 1 or 1.00. 0.00 or 9.8e-5 etc
        num_expr = Word(nums + '-' + '+' + 'e' + 'E' + '.') + Suppress(Optional(","))
        probability_expr = Suppress('probability') + Suppress('(') + OneOrMore(word_expr) + Suppress(')')
        optional_expr = Suppress('(') + Suppress(OneOrMore(word_expr2)) + Suppress(')')
        probab_attributes = optional_expr | Suppress('table')
        cpd_expr = probab_attributes + OneOrMore(num_expr)

        return probability_expr, cpd_expr


    def variable_block(self):
        start = re.finditer('variable', self.network)
        for index in start:
            end = self.network.find('}\n', index.start())
            yield self.network[index.start():end]

    def probability_block(self):
        start = re.finditer('probability', self.network)
        for index in start:
            end = self.network.find('}\n', index.start())
            yield self.network[index.start():end]

    def get_network_name(self):
        """
        Retruns the name of the network

        Example
        ---------------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIF.BifReader("bif_test.bif")
        >>> reader.network_name()
        'Dog-Problem'
        """
        start = self.network.find('network')
        end = self.network.find('}\n', start)
        # Creating a network attribute
        network_attribute = Suppress('network') + Word(alphanums + '_' + '-') + '{'
        network_name = network_attribute.searchString(self.network[start:end])[0][0]

        return network_name


    def get_variables(self):
        """
        Returns list of variables of the network

        Example
        -------------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_variables()
        ['light-on','bowel_problem','dog-out','hear-bark','family-out']
        """
        variable_names = []
        for block in self.variable_block():
            name = self.name_expr.searchString(block)[0][0]
            variable_names.append(name)

        return variable_names


    def get_states(self):
        """
        Returns the states of variables present in the network

        Example
        -----------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_states()
        {'bowel-problem': ['true','false'],
        'dog-out': ['true','false'],
        'family-out': ['true','false'],
        'hear-bark': ['true','false'],
        'light-on': ['true','false']}
        """
        variable_states = {}
        for block in self.variable_block():
            name = self.name_expr.searchString(block)[0][0]
            variable_states[name] = list(self.state_expr.searchString(block)[0][0])

        return variable_states


    def get_property(self):
        """
        Returns the property of the variable

        Example
        -------------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_property()
        {'bowel-problem': ['position = (335, 99)'],
        'dog-out': ['position = (300, 195)'],
        'family-out': ['position = (257, 99)'],
        'hear-bark': ['position = (296, 268)'],
        'light-on': ['position = (218, 195)']}
        """
        variable_properties = {}
        for block in self.variable_block():
            name = self.name_expr.searchString(block)[0][0]
            properties = self.property_expr.searchString(block)
            variable_properties[name] = [y.strip() for x in properties for y in x]
        return variable_properties


    def get_parents(self):
        """
        Returns the parents of the variables present in the network

        Example
        --------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_parents()
        {'bowel-problem': [],
        'dog-out': ['family-out', 'bowel-problem'],
        'family-out': [],
        'hear-bark': ['dog-out'],
        'light-on': ['family-out']}
        """
        variable_parents = {}
        for block in self.probability_block():
            names = self.probability_expr.searchString(block.split('\n')[0])[0]
            variable_parents[names[0]] = names[1:]
        return variable_parents


    def get_cpd(self):
        """
        Returns the CPD of the variables present in the network

        Example
        --------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_cpd()
        {'bowel-problem': np.array([[0.01],
                                    [0.99]]),
        'dog-out': np.array([[0.99, 0.97, 0.9, 0.3],
                            [0.01, 0.03, 0.1, 0.7]]),
        'family-out': np.array([[0.15],
                                [0.85]]),
        'hear-bark': np.array([[0.7, 0.01],
                                [0.3, 0.99]]),
        'light-on': np.array([[0.6, 0.05],
                            [0.4, 0.95]])}
         """
        variable_cpds = {}
        for block in self.probability_block():
            name = self.probability_expr.searchString(block)[0][0]
            cpds = self.cpd_expr.searchString(block)
            arr = [float(j) for i in cpds for j in i]
            if 'table' in block:
                arr = numpy.array(arr)
                arr = arr.reshape((len(self.variable_states[name]),
                                   arr.size // len(self.variable_states[name])))
            else:
                length = len(self.variable_states[name])
                reshape_arr = [[] for i in range(length)]
                for i, val in enumerate(arr):
                    reshape_arr[i % length].append(val)
                arr = reshape_arr
                arr = numpy.array(arr)
            variable_cpds[name] = arr

        return variable_cpds


    def get_edges(self):
        """
        Returns the edges of the network

        Example
        --------
        >>> from pgmpy.readwrite import BIFReader
        >>> reader = BIFReader("bif_test.bif")
        >>> reader.get_edges()
        [['family-out', 'light-on'],
         ['family-out', 'dog-out'],
         ['bowel-problem', 'dog-out'],
         ['dog-out', 'hear-bark']]
        """
        edges = [[value, key] for key in self.variable_parents.keys()
                 for value in self.variable_parents[key]]
        return edges
