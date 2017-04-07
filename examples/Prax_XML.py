
# coding: utf-8

# In[341]:

import re
from string import Template
import random
import numpy
from pyparsing import Word, alphanums, Suppress, Optional, CharsNotIn, Group, nums, ZeroOrMore, OneOrMore,    cppStyleComment, printables
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
        
import copy

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


# In[84]:




# In[18]:

reader = XMLBIFReader('aima-alarm.xml')
parents = reader.get_parents()
len(parents['B']) is 0 


# In[3]:

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
        variable_state_expr = Suppress('type') + Suppress(word_expr) + Suppress('[') + Suppress(Word(nums)) +             Suppress(']') + Suppress('{') + Group(state_expr) + Suppress('}') + Suppress(';')
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


# In[7]:



# In[397]:

class Bayesian_Network():
    def __init__(self):
        self.nodes = {}
        self.vars = []
        self.parents = {}
        self.probabilities = {}
        self.distributions = {}
        self.remaining_hidden_vars = []
    
    def __init__(self,variables,parents,probabilities,states):
        self.nodes = {}
        for var in variables:
            new_node = Node(var,parents[var],probabilities[var],states[var])
            self.nodes[var] = new_node
        self.num_vars = len(variables)
        self.vars = variables
        self.ordered_vars = copy.copy(self.vars)
        self.ordered_vars = order_topologically(self,self.ordered_vars)
        self.parents = parents
        self.completed_nodes = []
        self.prev_node = None
        
    
    def update(self,var,val):
        self.nodes[var].value = val
        
    def clear(self,label):
        self.nodes[label].value = None
        
    def add_to_completed_nodes(self,Y):
        if not contains(Y,self.completed_nodes):
            self.completed_nodes.append(Y)
        self.prev_node = Y
    
    def reset_completed_nodes(self):
        self.completed_nodes = []
        
    def remove_from_completed(self,Y):
        if contains(Y,self.completed_nodes):
            self.completed_nodes.remove(Y)
        
    def get_condit_prob(self,Y):
        node = self.nodes[Y]
        if node.is_top_level():
            return node.condit_probs[node.states.index(node.value)][0]
        else: #calculating index in condtional_prob table
            num_states = []
            indices = []
            
            for parent_name in node.parents:
                parent = self.nodes[parent_name]
                num_states.append(len(parent.states))
                indices.append(parent.states.index(parent.value))
            index = 0
            for i in range(len(node.parents)):
                index += indices[i] * multiply_all(num_states,i)
            return node.condit_probs[node.states.index(node.value)][index]
def contains(var,node_names):
    for name in node_names:
        if name == var:
            return True
    return False 


def order_topologically(bn,array,currently_ordered=[]):
    #orders an array such that parents are always before their children
    if len(array) == 0: #if we are finished!
        return currently_ordered
    no_parents_yet = []
    ordered_array = copy.copy(currently_ordered) #to avoid passing by reference
    for var in array: 
        if bn.nodes[var].is_top_level(): #get the top nodes out of the way first
            ordered_array.append(var)
        else:
            contains_all_parents = True 
            for parent in bn.nodes[var].parents: #only add nodes if all parents already in array
                if not contains(parent,ordered_array):
                    contains_all_parents = False
                    break
            if contains_all_parents:
                ordered_array.append(var)
            else:
                no_parents_yet.append(var) #re-do process till complete
    return order_topologically(bn,no_parents_yet,currently_ordered=ordered_array)



def multiply_all(num_states,i):
    #multiplies all values in num_states from index i+1 to the end
    if i == 0:
        return 1
    else:
        product = 1
        adjusted_num_states = num_states[i+1:]
        for element in adjusted_num_states:
            product *= element
        return product
            

        
            
            
    

        
        
    

class Node(Bayesian_Network):
    
    def __init__(self, var_name,parents,probability_table,states):
        self.var = var_name
        self.condit_probs = probability_table
        self.parents = parents
        self.states = states
        self.value = None
    
    def is_top_level(self): #returns whether or not a given node has no parents
        if len(self.parents) == 0:
            return True
        else: 
            return False
            


            
            
net = Bayesian_Network(reader.get_variables(),reader.get_parents(),reader.get_cpd(),reader.get_states())


# In[402]:

def likelihood_weighting(X,E,e,bn,N):
    # returns the normalized probabilities of the different states X can take on given evidence E/e.
    for var, val in zip(E,e):
        bn.update(var,val) #adding evidence to Bayesian network
    weighted_counts = [0 for i in bn.nodes[X].states]
    index = bn.ordered_vars.index(X) 
    for j in range(N):
        x,w = weighted_sample(bn)
        index_of_state = bn.nodes[X].states.index(x[index]) 
        # ^ the index of the state in x[index]
        #where x[index] is the value corresponding to variable X
        weighted_counts[index_of_state] += w
    normalized_probabilities = [1/sum(weighted_counts) * i for i in weighted_counts]
    return { key:val for key,val in zip(bn.nodes[X].states,normalized_probabilities)} #returns as dictionary
    #got the idea for this nifty dictionary creation from: http://stackoverflow.com/questions/18634650/set-to-dict-python    
        
    


# In[399]:

a = ['a','b','c','d','e','f']
b=[1,2,3,4,5,6]
c = {i:j for i,j in zip(a,b)}
c


# In[400]:

def weighted_sample(bn):
    #generates one weighted sample
    w = 1
    x = []
    affected_vars = []
    for var in bn.ordered_vars: #for every variable in bn
        if bn.nodes[var].value is not None: #if it is an evidence var
            x.append(bn.nodes[var].value)
            w *= bn.get_condit_prob(var)
        else: #if it's not an evidence var
            probs = []
            for state in bn.nodes[var].states: #get conditional prob for each state of var
                bn.update(var,state)
                probs.append(bn.get_condit_prob(var))
            bn.clear(var) #tidying up
            normalized_probs = [1/sum(probs) * i for i in probs]
            state = select_element(normalized_probs,bn.nodes[var].states)
            x.append(state)
            affected_vars.append(var)
            bn.update(var,state)
    for var in affected_vars:
        bn.clear(var)
    return x,w

        
            
def select_element(probs,states):
    #randomly selects a state from states weighted by probabilities in probs
    #probs must sum to 1
    probs = copy.copy(probs)
    states = copy.copy(states)
    rand_num = random.random()
    for probability, state in zip(probs,states):
        rand_num -= probability #in this way each state has a certain interval of probability
        if rand_num < 0:
            return state
    raise ValueError ('rand_num larger than the sum of probabilities.')
            


# In[404]:

reader = XMLBIFReader('aima-alarm.xml')
net = Bayesian_Network(reader.get_variables(),reader.get_parents(),reader.get_cpd(),reader.get_states())
net.update('B','true')
estimated = likelihood_weighting('A',['J'],['true'],net,5000)
actual = enumeration_ask('A',['J'],['true'],net)
print('Estimate: ',estimated)
print('Actual: ',actual)


# In[353]:

multiply_all([0,1,2,3,4,5,6],2)


# In[354]:

a=[0,1,2,3,4,5]
a[:]


# In[355]:

def get_next_var(bn):
    for var in bn.vars:
        if bn.nodes[var].is_top_level() and not contains(var,bn.completed_nodes):
            return var
        else:
            if not contains(var,bn.completed_nodes): #if this node hasn't already been selected
                parents_have_values = True #interesting statement
                for parent in bn.nodes[var].parents: #iterate over parents
                    if bn.nodes[parent].value is None:
                        parents_have_values = False #if any parent does not have a value yet don't choose var
                        break 
                if parents_have_values == True:
                    return var
    raise ValueError('No variables left with defined parents.')



# In[375]:

def enumeration_ask(X,E,e,bn):
    '''Returns the disitrbution for the values of X (dictionary of states and values)
    X: Query Variable Name (String)
    E: Evidence Variable Names (list)
    e: values for the evidence variables in the same order (list)
    bn: Bayesian network (Bayesian_Network)
    '''
    Q = {}
    for var,val in zip(E,e):
        bn.update(var,val)
    for val in bn.nodes[X].states:
        bn.update(X,val)
        bn.reset_completed_nodes()
        Q[val] = enumerate_all(bn)
    return normalize(Q,bn.nodes[X].states)

def normalize(Q,states):
    Q = dict(Q)
    correction_factor = sum(list(Q.values()))
    for val in states:
        Q[val] = Q[val] / correction_factor
    return Q  
        
def enumerate_all(bn):
    if len(bn.completed_nodes) == len(bn.vars):
        return 1.0
    else:
        Y = get_next_var(bn)
        if bn.nodes[Y].value is not None:
            bn.add_to_completed_nodes(Y)           
            val = bn.get_condit_prob(Y)*enumerate_all(copy.deepcopy(bn))
            return val
        else:
            summation = 0
            for y in bn.nodes[Y].states:
                bn.update(Y,y)
                bn.add_to_completed_nodes(Y)
                summation += bn.get_condit_prob(Y) * enumerate_all(copy.deepcopy(bn))
                bn.clear(Y)
            return summation
                
        
        


# In[376]:

reader = XMLBIFReader('aima-alarm.xml')
net = Bayesian_Network(reader.get_variables(),reader.get_parents(),reader.get_cpd(),reader.get_states())
#enumeration_ask('B',['J','M'],['true','false'],net)
enumeration_ask('B',['J','M'],['true','false'],net)


# In[403]:

reader = XMLBIFReader('dog-problem.xml')
net = Bayesian_Network(reader.get_variables(),reader.get_parents(),reader.get_cpd(),reader.get_states())
enumeration_ask('hear-bark',['bowel-problem','family-out'],['true','true'],net)


# In[160]:

reader.get_states()


# In[161]:

reader.get_cpd()


# In[166]:




# In[61]:

reader_grass = XMLBIFReader('aima-wet-grass.xml')
help(reader_grass.get_property)
reader.get_property()

