from lxml import etree
import xml.etree.ElementTree as ET
import sys

# XML = str(open(sys.argv[1]).read()
# parser =  etree.XMLParser(remove_comments=True)
# tree= etree.fromstring(XML, parser = parser)
# print (etree.tostring(tree))

tree = ET.parse(sys.argv[1])
tree.write('testing_faithfulness_alarm.xml')

	