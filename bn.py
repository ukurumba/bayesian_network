import random
import copy

################################################################# CLASSES #############################################################

class Bayesian_Network():
    def __init__(self):
        self.nodes = {}
        self.vars = []
        self.parents = {}
        self.probabilities = {}
        self.distributions = {}
        self.ordered_vars = []
        self.completed_nodes = []
        self.prev_node = None
    
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
    	print(Y,'OKKKKK GURL')
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
            print(index)
            return node.condit_probs[node.states.index(node.value)][index]

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

############################################## INFERENCE IMPLEMENTATION #########################################################


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

def enumerate_all(bn):
    if len(bn.completed_nodes) == len(bn.vars):
        return 1.0
    else:
        Y = get_next_var(bn)
        if bn.nodes[Y].value is not None:
            bn.add_to_completed_nodes(Y)           
            val = bn.get_condit_prob(Y)*enumerate_all(copy.deepcopy(bn)) #don't want actual bn to be affected by future state changes
            return val
        else:
            summation = 0
            for y in bn.nodes[Y].states:
                bn.update(Y,y)
                bn.add_to_completed_nodes(Y)
                summation += bn.get_condit_prob(Y) * enumerate_all(copy.deepcopy(bn))
                bn.clear(Y)
            return summation

def get_next_var(bn):
	#helper function for enumeration method. returns the next variable to select. 
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

            
##################################################### UTILITY FUNCTIONS #########################################################


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

def contains(var,node_names):
	#returns whether or not the array (node_names) contains the var
    for name in node_names:
        if name == var:
            return True
    return False 

def normalize(Q,states):
	#normalizes a dictionary of probabilities with keys = states
    Q = dict(Q)
    correction_factor = sum(list(Q.values()))
    for val in states:
        Q[val] = Q[val] / correction_factor
    return Q  
            