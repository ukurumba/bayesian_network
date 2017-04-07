import src.parsers as p
import src.bn as bn
import xml.etree.ElementTree as ET
import sys
import os

if sys.argv[1][-4:] == '.xml':
	tree = ET.parse(sys.argv[1])
	tree.write('temp.xml')
	reader = p.XMLBIFReader('temp.xml')
	os.remove('temp.xml')

elif sys.argv[1][-4] == '.bif':
	reader = p.BIFReader(sys.argv[1])

else:
	raise NameError("File I/O Failed. Please ensure file format is .xml or .bif and correct order of arguments. See README for format/examples")

E = []
e = []
for i in range(len(sys.argv)):
	if i > 2:
		if i/2 == float(i//2): #even arguments
			e.append(sys.argv[i])
		else:
			E.append(sys.argv[i])

net = bn.Bayesian_Network(reader.get_variables(),reader.get_parents(),reader.get_cpd(),reader.get_states())
estimated = bn.likelihood_weighting(sys.argv[2],E,e,net,1000)
actual = bn.enumeration_ask(sys.argv[2],E,e,net)
print('Estimate: ',estimated)
print('Actual: ',actual)

