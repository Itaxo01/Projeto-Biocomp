import sys
import os
import json
import libsbml
from typing import Dict, List, Any, Optional

class SBMLParser:
    """
    Parses an SBML file to extract information relevant for Gillespie simulations
    and Petri Net visualization.
    """

    def __init__(self, filepath: str) -> None:
        """
        Initialize the parser with the path to the SBML file.

        Args:
            filepath (str): Path to the .xml SBML file.
        """
        self.filepath = filepath
        self.reader = libsbml.SBMLReader()
        self.document = None
        self.model = None

    def parse(self) -> Optional[Dict[str, Any]]:
        """
        Parses the SBML file and returns a dictionary with the extracted data.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing model details, species,
                                      parameters, and reactions, or None if parsing fails.
        """
        if not os.path.exists(self.filepath):
            print(f"Error: File '{self.filepath}' not found.")
            return None

        self.document = self.reader.readSBML(self.filepath)
        
        if self.document.getNumErrors() > 0:
            print(f"Error reading SBML file: {self.document.getError(0).getMessage()}")
            return None

        self.model = self.document.getModel()
        if not self.model:
            print("Error: Could not retrieve model from SBML document.")
            return None

        # Extract function definitions first
        function_defs = self._extract_function_definitions()
        
        # Extract compartments
        compartments = self._extract_compartments()
        
        # Extract parameters
        parameters = self._extract_parameters()
        
        # Extract assignment rules (for dynamic evaluation during simulation)
        assignment_rules = self._extract_assignment_rules()
        
        # Extract time-related information (units, symbol)
        time_info = self._extract_time_info()
        
        # Extract SBML events (triggers, delays, assignments)
        events = self._extract_events()
        
        # Process Assignment Rules to update initial parameter values
        parameters = self._process_assignment_rules(parameters)

        # Extract species and reactions
        species = self._extract_species()
        reactions = self._extract_reactions()
        
        # Try to infer better species names from reaction names
        # (useful for models with generic variable names like x1, x2, etc.)
        species = self._infer_species_names_from_reactions(species, reactions)

        data = {
            "model_id": self.model.getId(),
            "compartments": compartments,
            "species": species,
            "parameters": parameters,
            "reactions": reactions,
            "function_definitions": function_defs,
            "assignment_rules": assignment_rules,
            "time_info": time_info,
            "events": events
        }
        
        return data

    def _extract_assignment_rules(self) -> List[Dict[str, str]]:
        """Extracts assignment rules for dynamic evaluation during simulation."""
        rules = []
        for i in range(self.model.getNumRules()):
            rule = self.model.getRule(i)
            if rule.isAssignment():
                rules.append({
                    "variable": rule.getVariable(),
                    "formula": rule.getFormula()
                })
        return rules

    def _extract_time_info(self) -> Dict[str, Any]:
        """
        Extracts time-related information from the SBML model.
        Returns a dict with:
          - time_symbol: The symbol used for time in formulas (e.g., 'time', 'T')
          - time_unit: The unit definition for time (e.g., 'second', 'hour')
          - time_scale: Conversion factor to seconds (1.0 if already seconds)
        """
        time_info = {
            "time_symbol": "time",  # Default SBML time symbol
            "time_unit": "dimensionless",
            "time_scale": 1.0  # seconds per model time unit
        }
        
        # Check for time unit definition
        for i in range(self.model.getNumUnitDefinitions()):
            unit_def = self.model.getUnitDefinition(i)
            unit_id = unit_def.getId()
            
            if unit_id.lower() == 'time':
                # Parse the unit definition
                for j in range(unit_def.getNumUnits()):
                    unit = unit_def.getUnit(j)
                    kind = libsbml.UnitKind_toString(unit.getKind())
                    multiplier = unit.getMultiplier()
                    scale = unit.getScale()  # 10^scale
                    exponent = unit.getExponent()
                    
                    # Calculate conversion factor
                    factor = multiplier * (10 ** scale) ** exponent
                    
                    if kind == 'second':
                        time_info["time_unit"] = "second"
                        time_info["time_scale"] = factor
                    elif kind == 'hour':
                        time_info["time_unit"] = "hour"
                        time_info["time_scale"] = factor * 3600  # convert to seconds
                    elif kind == 'minute':
                        time_info["time_unit"] = "minute"
                        time_info["time_scale"] = factor * 60
                    elif kind == 'day':
                        time_info["time_unit"] = "day"
                        time_info["time_scale"] = factor * 86400
                    else:
                        time_info["time_unit"] = kind
                        time_info["time_scale"] = factor
        
        # Also check if model uses 'T' as time symbol (common in some models)
        # by scanning kinetic laws and event triggers for csymbol time references
        # Note: libSBML doesn't directly expose this, but we can check reactions
        for i in range(self.model.getNumReactions()):
            rxn = self.model.getReaction(i)
            kl = rxn.getKineticLaw()
            if kl:
                formula = kl.getFormula()
                # If 'T' appears as a standalone token (not part of another word)
                # and 'time' doesn't, then 'T' is likely the time symbol
                import re
                if re.search(r'\bT\b', formula) and not re.search(r'\btime\b', formula):
                    time_info["time_symbol"] = "T"
                    break
        
        # Check events for time symbol usage
        for i in range(self.model.getNumEvents()):
            event = self.model.getEvent(i)
            trigger = event.getTrigger()
            if trigger:
                trigger_math = trigger.getMath()
                if trigger_math:
                    formula_str = libsbml.formulaToString(trigger_math)
                    import re
                    if re.search(r'\bT\b', formula_str):
                        time_info["time_symbol"] = "T"
                        break
        
        return time_info

    def _extract_events(self) -> List[Dict[str, Any]]:
        """
        Extracts SBML events with triggers, delays, and event assignments.
        
        Events in SBML consist of:
        - trigger: A boolean condition that, when it becomes true, fires the event
        - delay: Optional time delay between trigger firing and assignment execution
        - listOfEventAssignments: Variables to modify when the event fires
        
        Returns:
            List of event dictionaries with structure:
            {
                "id": str,
                "name": str,
                "trigger": str (formula),
                "trigger_initial_value": bool,
                "trigger_persistent": bool,
                "delay": str (formula, usually "0"),
                "use_values_from_trigger_time": bool,
                "assignments": [{"variable": str, "formula": str}, ...]
            }
        """
        events = []
        
        for i in range(self.model.getNumEvents()):
            event = self.model.getEvent(i)
            event_data = {
                "id": event.getId(),
                "name": event.getName() if event.getName() else event.getId(),
                "trigger": "",
                "trigger_initial_value": True,  # Default: trigger starts as "can fire"
                "trigger_persistent": True,     # Default: trigger must stay true
                "delay": "0",
                "use_values_from_trigger_time": True,
                "assignments": []
            }
            
            # Extract trigger
            trigger = event.getTrigger()
            if trigger:
                trigger_math = trigger.getMath()
                if trigger_math:
                    # Convert MathML to infix string
                    trigger_formula = libsbml.formulaToString(trigger_math)
                    # Replace SBML time csymbol with our time variable
                    # The csymbol for time in SBML is typically represented as 'time' in formula
                    event_data["trigger"] = trigger_formula
                
                # Get trigger properties (SBML Level 3)
                if trigger.isSetInitialValue():
                    event_data["trigger_initial_value"] = trigger.getInitialValue()
                if trigger.isSetPersistent():
                    event_data["trigger_persistent"] = trigger.getPersistent()
            
            # Extract delay
            delay = event.getDelay()
            if delay:
                delay_math = delay.getMath()
                if delay_math:
                    event_data["delay"] = libsbml.formulaToString(delay_math)
            
            # Extract use values from trigger time (SBML Level 3)
            if event.isSetUseValuesFromTriggerTime():
                event_data["use_values_from_trigger_time"] = event.getUseValuesFromTriggerTime()
            
            # Extract event assignments
            for j in range(event.getNumEventAssignments()):
                ea = event.getEventAssignment(j)
                assignment_math = ea.getMath()
                if assignment_math:
                    event_data["assignments"].append({
                        "variable": ea.getVariable(),
                        "formula": libsbml.formulaToString(assignment_math)
                    })
            
            events.append(event_data)
        
        return events

    def _extract_compartments(self) -> List[Dict[str, Any]]:
        """Extracts compartments information."""
        compartment_list = []
        for i in range(self.model.getNumCompartments()):
            comp = self.model.getCompartment(i)
            compartment_list.append({
                "id": comp.getId(),
                "name": comp.getName() if comp.getName() else comp.getId(),
                "size": comp.getSize() if comp.isSetSize() else 1.0
            })
        return compartment_list

    def _process_assignment_rules(self, parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluates Assignment Rules to update parameter values.
        """
        # Convert list to dict for easy access
        param_dict = {p['id']: p['value'] for p in parameters}
        
        # Add math constants/functions for eval
        import math
        context = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
        context.update(param_dict)

        # Get rules
        rules = []
        for i in range(self.model.getNumRules()):
            rule = self.model.getRule(i)
            if rule.isAssignment():
                rules.append((rule.getVariable(), rule.getFormula()))

        # Iteratively evaluate rules to resolve dependencies
        # (Simple fixed-point iteration)
        max_iterations = len(rules) + 2
        for _ in range(max_iterations):
            changed = False
            for variable, formula in rules:
                try:
                    # Update context with current values
                    context.update(param_dict)
                    new_value = eval(formula, {}, context)
                    
                    if variable in param_dict:
                        if abs(param_dict[variable] - new_value) > 1e-9:
                            param_dict[variable] = new_value
                            changed = True
                    else:
                        # Variable might be a parameter not yet in the list (though usually it is)
                        param_dict[variable] = new_value
                        changed = True
                except Exception as e:
                    # print(f"Warning: Could not evaluate rule for {variable}: {e}")
                    pass
            if not changed:
                break
        
        # Convert back to list
        updated_params = [{"id": k, "value": v} for k, v in param_dict.items()]
        return updated_params

    def _extract_function_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Extracts SBML function definitions."""
        func_defs = {}
        for i in range(self.model.getNumFunctionDefinitions()):
            func_def = self.model.getFunctionDefinition(i)
            func_id = func_def.getId()
            
            # Get the math (lambda expression)
            math = func_def.getMath()
            if math:
                # Get argument names
                args = []
                for j in range(func_def.getNumArguments()):
                    args.append(func_def.getArgument(j).getName())
                
                # Get the body of the lambda (the actual formula)
                body = math.getChild(math.getNumChildren() - 1)  # Last child is the body
                formula = libsbml.formulaToString(body)
                
                func_defs[func_id] = {
                    "id": func_id,
                    "name": func_def.getName() if func_def.getName() else func_id,
                    "arguments": args,
                    "formula": formula
                }
        return func_defs

    def _extract_species(self) -> List[Dict[str, Any]]:
        """Extracts species (Petri Net Places) information."""
        species_list = []
        for i in range(self.model.getNumSpecies()):
            sp = self.model.getSpecies(i)
            species_list.append({
                "id": sp.getId(),
                "name": sp.getName() if sp.getName() else sp.getId(),
                "initial_amount": sp.getInitialAmount() if sp.isSetInitialAmount() else sp.getInitialConcentration(),
                "boundary_condition": sp.getBoundaryCondition()
            })
        return species_list

    def _infer_species_names_from_reactions(self, species_list: List[Dict[str, Any]], 
                                            reactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Infers better species names from reaction names when species have generic names (x1, x2, etc.).
        This is useful for models that use generic variable names but descriptive reaction names.
        """
        import re
        
        # Build a mapping of species_id -> inferred name
        inferred_names = {}
        
        for rxn in reactions:
            rxn_name = rxn.get('name', rxn.get('id', ''))
            
            # Parse reaction name patterns like "r001 - mA_transcription", "r002 - A_translation"
            # Extract the biological process and target
            match = re.search(r'[-_]\s*(\w+?)_(transcription|translation|degradation|transport|dim)', rxn_name, re.IGNORECASE)
            if match:
                target = match.group(1)  # e.g., "mA", "A", "I", "Ie"
                process = match.group(2).lower()
                
                # Map to species based on products/reactants
                if process == 'transcription':
                    # Product is mRNA
                    for prod in rxn.get('products', []):
                        sp_id = prod['species']
                        if sp_id not in inferred_names or inferred_names[sp_id].startswith('x'):
                            inferred_names[sp_id] = f"mRNA_{target.replace('m', '')}" if target.startswith('m') else f"mRNA_{target}"
                elif process == 'translation':
                    # Product (excluding mRNA) is protein
                    for prod in rxn.get('products', []):
                        sp_id = prod['species']
                        # Skip if it's already marked as mRNA
                        if sp_id in inferred_names and 'mRNA' in inferred_names[sp_id]:
                            continue
                        if sp_id not in inferred_names or inferred_names[sp_id].startswith('x'):
                            inferred_names[sp_id] = f"Protein_{target}"
                elif process == 'degradation':
                    # Reactant is the species being degraded
                    for react in rxn.get('reactants', []):
                        sp_id = react['species']
                        if sp_id not in inferred_names or inferred_names[sp_id].startswith('x'):
                            if target.startswith('m'):
                                inferred_names[sp_id] = f"mRNA_{target[1:]}"
                            elif target == 'I':
                                inferred_names[sp_id] = "Inducer"
                            elif target == 'Ie':
                                inferred_names[sp_id] = "Inducer_ext"
                            elif target.startswith('AI'):
                                inferred_names[sp_id] = f"Complex_{target}"
                            else:
                                inferred_names[sp_id] = f"Protein_{target}"
                elif process == 'transport':
                    for react in rxn.get('reactants', []):
                        sp_id = react['species']
                        if sp_id not in inferred_names or inferred_names[sp_id].startswith('x'):
                            if 'I' in target:
                                inferred_names[sp_id] = "Inducer"
                elif process == 'dim':
                    # Dimerization - products are complexes
                    for prod in rxn.get('products', []):
                        sp_id = prod['species']
                        if sp_id not in inferred_names or inferred_names[sp_id].startswith('x'):
                            inferred_names[sp_id] = f"Complex_{target}"
        
        # Also check for patterns like "Ie_degradation" for external inducer
        for rxn in reactions:
            rxn_name = rxn.get('name', rxn.get('id', ''))
            if 'Ie' in rxn_name:
                for react in rxn.get('reactants', []):
                    sp_id = react['species']
                    if sp_id not in inferred_names or inferred_names[sp_id].startswith('x'):
                        inferred_names[sp_id] = "Inducer_ext"
        
        # Update species names
        for sp in species_list:
            sp_id = sp['id']
            # Only update if current name is generic (same as id or matches x\d+ pattern)
            if sp['name'] == sp['id'] or re.match(r'^x\d+$', sp['name']):
                if sp_id in inferred_names:
                    sp['name'] = inferred_names[sp_id]
        
        return species_list

    def _extract_parameters(self) -> List[Dict[str, Any]]:
        """Extracts global parameters."""
        param_list = []
        for i in range(self.model.getNumParameters()):
            param = self.model.getParameter(i)
            param_list.append({
                "id": param.getId(),
                "value": param.getValue()
            })
        return param_list

    def _extract_reactions(self) -> List[Dict[str, Any]]:
        """Extracts reactions (Petri Net Transitions) information."""
        reaction_list = []
        for i in range(self.model.getNumReactions()):
            rxn = self.model.getReaction(i)
            
            reactants = []
            for j in range(rxn.getNumReactants()):
                ref = rxn.getReactant(j)
                reactants.append({
                    "species": ref.getSpecies(),
                    "stoichiometry": ref.getStoichiometry()
                })

            products = []
            for j in range(rxn.getNumProducts()):
                ref = rxn.getProduct(j)
                products.append({
                    "species": ref.getSpecies(),
                    "stoichiometry": ref.getStoichiometry()
                })
            
            modifiers = []
            seen_modifiers = set()  # Track unique modifiers
            for j in range(rxn.getNumModifiers()):
                ref = rxn.getModifier(j)
                species_id = ref.getSpecies()
                # Avoid duplicate modifiers (some SBML files have duplicates)
                if species_id not in seen_modifiers:
                    seen_modifiers.add(species_id)
                    modifiers.append({
                        "species": species_id
                    })

            kinetic_law = rxn.getKineticLaw()
            formula = kinetic_law.getFormula() if kinetic_law else ""
            
            # Extract local parameters if any
            local_params = []
            if kinetic_law:
                for k in range(kinetic_law.getNumParameters()):
                    lp = kinetic_law.getParameter(k)
                    local_params.append({
                        "id": lp.getId(),
                        "value": lp.getValue()
                    })

            reaction_list.append({
                "id": rxn.getId(),
                "name": rxn.getName() if rxn.getName() else rxn.getId(),
                "reversible": rxn.getReversible(),
                "reactants": reactants,
                "products": products,
                "modifiers": modifiers,
                "kinetic_law": formula,
                "local_parameters": local_params
            })
        return reaction_list

    def save_to_json(self, data: Dict[str, Any], output_file: str) -> None:
        """
        Saves the extracted data to a JSON file.

        Args:
            data (Dict[str, Any]): The data to save.
            output_file (str): The path to the output JSON file.
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Successfully saved parsed model to '{os.path.relpath(output_file)}'.")
        except Exception as e:
            print(f"Error saving JSON: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_sbml.py <path_to_sbml_xml>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = os.path.splitext(input_file)[0] + "_parsed.json"

    parser = SBMLParser(input_file)
    parsed_data = parser.parse()

    if parsed_data:
        parser.save_to_json(parsed_data, output_file)
