import libsbml
import sys

reader = libsbml.SBMLReader()
doc = reader.readSBML("../output/Elowitz2000 - Repressilator-BIOMD0000000012/BIOMD0000000012.xml")
model = doc.getModel()

print("--- Assignment Rules ---")
for i in range(model.getNumRules()):
    rule = model.getRule(i)
    print(f"Variable: {rule.getVariable()}, Formula: {rule.getFormula()}")

print("\n--- Initial Assignments ---")
for i in range(model.getNumInitialAssignments()):
    ia = model.getInitialAssignment(i)
    print(f"Symbol: {ia.getSymbol()}, Formula: {ia.getMath().toFormula()}")
