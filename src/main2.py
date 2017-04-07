from src.parsers import XMLBIFReader, BIFReader
import src.bn as bn

reader = XMLBIFReader('../examples/aima-alarm.xml')
net = bn.Bayesian_Network(reader.get_variables(),reader.get_parents(),reader.get_cpd(),reader.get_states())
for var in net.vars:
	print(net.nodes[var].states)
estimated = bn.likelihood_weighting('A',['J'],['true'],net,5000)
actual = bn.enumeration_ask('A',['J'],['true'],net)
print('Estimate: ',estimated)
print('Actual: ',actual)