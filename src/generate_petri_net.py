import json
import sys
import os
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom
import graphviz
import networkx as nx
import numpy as np
from typing import Dict, Any, List, Tuple

class PetriNetGenerator:
    """
    Generates a Stochastic Petri Net (SPN) visualization and PNML file 
    from a parsed SBML JSON structure.
    """

    def __init__(self, json_filepath: str) -> None:
        """
        Initialize with the path to the parsed JSON file.
        """
        self.json_filepath = json_filepath
        self.data = self._load_json()
        self.model_id = self.data.get("model_id", "model")
        # Determine output directory based on input file location
        self.output_dir = os.path.dirname(os.path.abspath(json_filepath))
        # Pre-calculate layout positions using NetworkX
        self.positions = self._calculate_layout()

    def _load_json(self) -> Dict[str, Any]:
        """Loads the JSON data."""
        with open(self.json_filepath, 'r') as f:
            return json.load(f)

    def _calculate_layout(self) -> Dict[str, Tuple[float, float]]:
        """
        Calculates optimized node positions using the Sugiyama hierarchical layout algorithm.
        This is ideal for directed graphs like Petri Nets, providing clear flow visualization.
        Returns a dictionary mapping node IDs to (x, y) coordinates.
        """
        species_list = self.data.get("species", [])
        reactions_list = self.data.get("reactions", [])
        
        if not species_list and not reactions_list:
            return {}
        
        # Build the graph
        G = nx.DiGraph()
        
        place_ids = [s['id'] for s in species_list]
        transition_ids = [r['id'] for r in reactions_list]
        
        for s in species_list:
            G.add_node(s['id'], node_type='place')
        for r in reactions_list:
            G.add_node(r['id'], node_type='transition')
        
        # Add edges based on reactions
        for reaction in reactions_list:
            for reactant in reaction.get("reactants", []):
                G.add_edge(reactant['species'], reaction['id'])
            for product in reaction.get("products", []):
                G.add_edge(reaction['id'], product['species'])
            for modifier in reaction.get("modifiers", []):
                G.add_edge(modifier['species'], reaction['id'])
        
        if len(G.nodes()) == 0:
            return {}
        
        # Sugiyama Algorithm Implementation
        # Step 1: Cycle removal (reverse edges to make DAG)
        # Step 2: Layer assignment (longest path layering)
        # Step 3: Crossing minimization (barycenter method)
        # Step 4: Coordinate assignment
        
        pos = self._sugiyama_layout(G, place_ids, transition_ids)
        
        # Scale and translate positions to fit canvas
        canvas_width = 1000
        canvas_height = 800
        margin = 80
        
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Avoid division by zero
            range_x = max_x - min_x if max_x != min_x else 1.0
            range_y = max_y - min_y if max_y != min_y else 1.0
            
            scaled_pos = {}
            for node_id, (x, y) in pos.items():
                norm_x = (x - min_x) / range_x
                norm_y = (y - min_y) / range_y
                
                scaled_x = margin + norm_x * (canvas_width - 2 * margin)
                scaled_y = margin + norm_y * (canvas_height - 2 * margin)
                
                scaled_pos[node_id] = (scaled_x, scaled_y)
            
            return scaled_pos
        
        return {}

    def _sugiyama_layout(self, G: nx.DiGraph, place_ids: List[str], transition_ids: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Implements the Sugiyama hierarchical layout algorithm.
        """
        # Step 1: Handle cycles by creating a DAG
        # We'll use a simple approach: find back edges and temporarily ignore them
        dag = self._make_acyclic(G.copy())
        
        # Step 2: Assign layers using longest path from sources
        layers = self._assign_layers(dag)
        
        # Step 3: Order nodes within layers to minimize crossings (barycenter method)
        layers = self._minimize_crossings(dag, layers)
        
        # Step 4: Assign coordinates
        pos = self._assign_coordinates(layers)
        
        return pos

    def _make_acyclic(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Removes cycles by reversing back edges using DFS.
        """
        try:
            # Find all cycles
            cycles = list(nx.simple_cycles(G))
            if not cycles:
                return G
            
            # Find edges that participate in cycles and reverse some of them
            edges_to_reverse = set()
            for cycle in cycles:
                # Reverse the edge from last to first node in cycle
                if len(cycle) >= 2:
                    edges_to_reverse.add((cycle[-1], cycle[0]))
            
            for u, v in edges_to_reverse:
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
                    # Don't add reverse - just remove to break cycle
        except:
            pass
        
        return G

    def _assign_layers(self, G: nx.DiGraph) -> List[List[str]]:
        """
        Assigns nodes to layers using longest path layering.
        Sources go to layer 0, and each node is placed at max(layer of predecessors) + 1.
        """
        if len(G.nodes()) == 0:
            return []
        
        # Find all nodes with no incoming edges (sources)
        in_degrees = dict(G.in_degree())
        
        # Calculate longest path to each node
        node_layer = {}
        
        # Topological sort (if possible) or just iterate
        try:
            topo_order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            # Graph has cycles, use all nodes in arbitrary order
            topo_order = list(G.nodes())
        
        # Assign layers based on longest path
        for node in topo_order:
            predecessors = list(G.predecessors(node))
            if not predecessors:
                node_layer[node] = 0
            else:
                max_pred_layer = max(node_layer.get(p, 0) for p in predecessors)
                node_layer[node] = max_pred_layer + 1
        
        # Handle any remaining nodes (disconnected or in cycles)
        for node in G.nodes():
            if node not in node_layer:
                node_layer[node] = 0
        
        # Group nodes by layer
        max_layer = max(node_layer.values()) if node_layer else 0
        layers = [[] for _ in range(max_layer + 1)]
        for node, layer in node_layer.items():
            layers[layer].append(node)
        
        return layers

    def _minimize_crossings(self, G: nx.DiGraph, layers: List[List[str]]) -> List[List[str]]:
        """
        Minimizes edge crossings using the barycenter method.
        Iteratively reorders nodes in each layer based on average position of neighbors.
        """
        if len(layers) <= 1:
            return layers
        
        # Multiple passes (typically 2-4 is sufficient)
        for _ in range(4):
            # Forward pass (top to bottom)
            for i in range(1, len(layers)):
                self._reorder_layer_barycenter(G, layers, i, direction='down')
            
            # Backward pass (bottom to top)
            for i in range(len(layers) - 2, -1, -1):
                self._reorder_layer_barycenter(G, layers, i, direction='up')
        
        return layers

    def _reorder_layer_barycenter(self, G: nx.DiGraph, layers: List[List[str]], layer_idx: int, direction: str):
        """
        Reorders nodes in a layer based on barycenter of connected nodes in adjacent layer.
        """
        if layer_idx < 0 or layer_idx >= len(layers):
            return
        
        current_layer = layers[layer_idx]
        
        if direction == 'down' and layer_idx > 0:
            adj_layer = layers[layer_idx - 1]
        elif direction == 'up' and layer_idx < len(layers) - 1:
            adj_layer = layers[layer_idx + 1]
        else:
            return
        
        # Create position map for adjacent layer
        adj_positions = {node: idx for idx, node in enumerate(adj_layer)}
        
        # Calculate barycenter for each node in current layer
        barycenters = []
        for node in current_layer:
            if direction == 'down':
                neighbors = list(G.predecessors(node))
            else:
                neighbors = list(G.successors(node))
            
            # Filter to neighbors in adjacent layer
            neighbor_positions = [adj_positions[n] for n in neighbors if n in adj_positions]
            
            if neighbor_positions:
                barycenter = sum(neighbor_positions) / len(neighbor_positions)
            else:
                # Keep original position if no neighbors
                barycenter = current_layer.index(node)
            
            barycenters.append((node, barycenter))
        
        # Sort by barycenter
        barycenters.sort(key=lambda x: x[1])
        layers[layer_idx] = [node for node, _ in barycenters]

    def _assign_coordinates(self, layers: List[List[str]]) -> Dict[str, Tuple[float, float]]:
        """
        Assigns x, y coordinates to nodes based on their layer and position within layer.
        """
        pos = {}
        
        if not layers:
            return pos
        
        # Horizontal spacing between layers
        layer_spacing = 150
        # Vertical spacing between nodes in same layer
        node_spacing = 100
        
        # Find max nodes in any layer for centering
        max_nodes_in_layer = max(len(layer) for layer in layers) if layers else 1
        
        for layer_idx, layer in enumerate(layers):
            x = layer_idx * layer_spacing
            
            # Center nodes vertically
            layer_height = (len(layer) - 1) * node_spacing
            start_y = (max_nodes_in_layer - 1) * node_spacing / 2 - layer_height / 2
            
            for node_idx, node in enumerate(layer):
                y = start_y + node_idx * node_spacing
                pos[node] = (x, y)
        
        return pos

    def generate_image(self, output_filename: str = None) -> str:
        """
        Generates a visual representation of the Petri Net using Graphviz.
        Returns the path to the generated image file.
        """
        if output_filename is None:
            output_filename = f"{self.model_id}_petri_net"
        
        # Ensure output path is in the same directory as input
        output_path_prefix = os.path.join(self.output_dir, output_filename)

        dot = graphviz.Digraph(comment=f'Petri Net for {self.model_id}')
        dot.attr(rankdir='LR')  # Left to Right layout
        dot.attr('node', fontname='Helvetica', fontsize='10')
        dot.attr('edge', fontsize='9')
        dot.attr('graph', nodesep='0.5', ranksep='1.0', splines='ortho')

        # Add Places (Species) - improved visualization
        for species in self.data.get("species", []):
            sp_name = species['name']
            sp_id = species['id']
            initial = species['initial_amount']
            
            # Use shorter label: prefer name over id, show tokens
            if sp_name != sp_id:
                label = f"{sp_name}\n[{int(initial) if initial == int(initial) else initial:.2f}]"
            else:
                label = f"{sp_id}\n[{int(initial) if initial == int(initial) else initial:.2f}]"
            
            # Use double circle for boundary species, single for regular
            if species.get('boundary_condition', False):
                dot.node(sp_id, label=label, shape='doublecircle', style='filled', 
                        fillcolor='lightyellow', width='0.8', height='0.8')
            else:
                dot.node(sp_id, label=label, shape='circle', style='filled', 
                        fillcolor='lightblue', width='0.8', height='0.8')

        # Add Transitions (Reactions) - improved visualization
        for reaction in self.data.get("reactions", []):
            t_id = reaction['id']
            rxn_name = reaction.get('name', t_id)
            
            # Extract a short, readable name from the reaction
            # Remove prefixes like "r001 - " or "reaction_"
            import re
            short_name = re.sub(r'^r?\d+\s*[-_]\s*', '', rxn_name)
            short_name = re.sub(r'^reaction_?', '', short_name, flags=re.IGNORECASE)
            
            # Truncate kinetic law for display (show only key part)
            kinetic_law = reaction.get('kinetic_law', '')
            if len(kinetic_law) > 30:
                # Try to extract the essential rate expression
                kinetic_short = kinetic_law[:27] + '...'
            else:
                kinetic_short = kinetic_law
            
            label = f"{short_name}"
            dot.node(t_id, label=label, shape='box', style='filled', 
                    fillcolor='lightgrey', width='0.3', height='0.6')

            # Arcs from Reactants to Transition
            for reactant in reaction.get("reactants", []):
                species_id = reactant['species']
                stoichiometry = reactant.get('stoichiometry', 1.0)
                edge_label = str(int(stoichiometry)) if stoichiometry != 1.0 else ""
                dot.edge(species_id, t_id, label=edge_label, arrowhead='normal')

            # Arcs from Transition to Products
            for product in reaction.get("products", []):
                species_id = product['species']
                stoichiometry = product.get('stoichiometry', 1.0)
                edge_label = str(int(stoichiometry)) if stoichiometry != 1.0 else ""
                dot.edge(t_id, species_id, label=edge_label, arrowhead='normal')

            # Modifier Arcs (e.g., enzymes, inhibitors) - dashed lines with different arrowhead
            for modifier in reaction.get("modifiers", []):
                species_id = modifier['species']
                dot.edge(species_id, t_id, style='dashed', arrowhead='odot', color='gray')

        # Render
        output_path = dot.render(output_path_prefix, format='png', cleanup=True)
        print(f"Petri Net image generated at: {os.path.relpath(output_path)}")
        return output_path

    def generate_pnml(self, output_filename: str = None) -> str:
        """
        Generates a PNML (Petri Net Markup Language) file compatible with tools like PIPE2, Snoopy, etc.
        Returns the path to the generated PNML file.
        """
        if output_filename is None:
            output_filename = f"{self.model_id}.pnml"
            
        full_output_path = os.path.join(self.output_dir, output_filename)

        # PNML Namespace
        pnml = ET.Element('pnml')
        net = ET.SubElement(pnml, 'net', id=self.model_id, type="http://www.pnml.org/version-2009/grammar/ptnet")
        page = ET.SubElement(net, 'page', id="page0")

        # Helper to generate unique IDs for arcs
        self.arc_counter = 0
        
        # Get pre-calculated layout positions
        species_list = self.data.get("species", [])
        reactions_list = self.data.get("reactions", [])

        # Places
        for i, species in enumerate(species_list):
            place = ET.SubElement(page, 'place', id=species['id'])
            
            # Graphics / Position from pre-calculated layout
            x, y = self.positions.get(species['id'], (100 + i * 80, 100))
            
            graphics = ET.SubElement(place, 'graphics')
            position = ET.SubElement(graphics, 'position', x=str(int(x)), y=str(int(y)))
            
            name = ET.SubElement(place, 'name')
            text = ET.SubElement(name, 'text')
            text.text = species['name']
            # Name graphics offset
            name_graphics = ET.SubElement(name, 'graphics')
            ET.SubElement(name_graphics, 'offset', x="0", y="-25")
            
            initial_marking = ET.SubElement(place, 'initialMarking')
            text_marking = ET.SubElement(initial_marking, 'text')
            # PNML expects integer tokens usually, but we'll put the float/int value
            text_marking.text = str(int(float(species['initial_amount'])))

        # Transitions
        for i, reaction in enumerate(reactions_list):
            transition = ET.SubElement(page, 'transition', id=reaction['id'])
            
            # Graphics / Position from pre-calculated layout
            x, y = self.positions.get(reaction['id'], (300, 100 + i * 80))
            
            graphics = ET.SubElement(transition, 'graphics')
            position = ET.SubElement(graphics, 'position', x=str(int(x)), y=str(int(y)))

            name = ET.SubElement(transition, 'name')
            text = ET.SubElement(name, 'text')
            text.text = reaction['name']
            # Name graphics offset
            name_graphics = ET.SubElement(name, 'graphics')
            ET.SubElement(name_graphics, 'offset', x="0", y="-20")
            
            # Add rate information (Tool specific, but we can add it as a tool specific tag or just name)
            # For standard PNML, rate is not always strictly defined in the core, but often in extensions.
            # We will stick to basic PT-Net structure.

            # Arcs (Reactants -> Transition)
            for reactant in reaction.get("reactants", []):
                self._add_arc(page, source=reactant['species'], target=reaction['id'], 
                              weight=reactant.get('stoichiometry', 1.0))

            # Arcs (Transition -> Products)
            for product in reaction.get("products", []):
                self._add_arc(page, source=reaction['id'], target=product['species'], 
                              weight=product.get('stoichiometry', 1.0))
            
            # Modifiers (Read Arcs)
            # Standard PNML doesn't strictly support read-arcs in the basic PT-Net grammar without extensions.
            # We will model them as a loop (Consume -> Produce back) to preserve the token count,
            # which is the standard way to model a catalyst in a basic Petri Net.
            for modifier in reaction.get("modifiers", []):
                species_id = modifier['species']
                # Arc In
                self._add_arc(page, source=species_id, target=reaction['id'], weight=1)
                # Arc Out (Restore token)
                self._add_arc(page, source=reaction['id'], target=species_id, weight=1)

        # Pretty print XML
        xml_str = minidom.parseString(ET.tostring(pnml)).toprettyxml(indent="   ")
        
        with open(full_output_path, "w") as f:
            f.write(xml_str)
            
        print(f"PNML file generated at: {os.path.relpath(full_output_path)}")
        return full_output_path

    def _add_arc(self, page_element, source, target, weight):
        """Helper to add an arc to the PNML."""
        self.arc_counter += 1
        arc_id = f"arc{self.arc_counter}"
        arc = ET.SubElement(page_element, 'arc', id=arc_id, source=source, target=target)
        
        inscription = ET.SubElement(arc, 'inscription')
        text = ET.SubElement(inscription, 'text')
        text.text = str(int(float(weight)))

    def generate_greatspn(self, output_filename: str = None) -> Tuple[str, str]:
        """
        Generates GreatSPN native format files (.net and .def).
        This format fully supports stochastic rates and is the recommended
        format for simulation in GreatSPN.
        
        For repression/activation modifiers, we use marking-dependent rates
        defined in the .def file.
        
        Returns tuple of (net_path, def_path).
        """
        if output_filename is None:
            output_filename = self.model_id
        
        net_path = os.path.join(self.output_dir, f"{output_filename}.net")
        def_path = os.path.join(self.output_dir, f"{output_filename}.def")
        
        species_list = self.data.get("species", [])
        reactions_list = self.data.get("reactions", [])
        parameters = {p['id']: p['value'] for p in self.data.get("parameters", [])}
        
        # GreatSPN reserved keywords to avoid
        RESERVED_KEYWORDS = {'X', 'Y', 'Z', 'E', 'PI', 'T', 'S', 'I', 'O', 'N', 'P', 'R', 
                            'x', 'y', 'z', 'e', 'pi', 't', 's', 'i', 'o', 'n', 'p', 'r',
                            'if', 'else', 'when', 'and', 'or', 'not', 'mod', 'div',
                            'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'pow', 'abs',
                            'min', 'max', 'floor', 'ceil', 'frac', 'int'}
        
        def safe_name(name: str) -> str:
            """Convert name to safe identifier for GreatSPN"""
            if name in RESERVED_KEYWORDS or len(name) == 1:
                return f"p_{name}"
            return name
        
        def descriptive_reaction_name(reaction: dict, index: int) -> str:
            """Generate a descriptive name for a reaction based on its reactants/products"""
            reactants = [r['species'] for r in reaction.get('reactants', [])]
            products = [p['species'] for p in reaction.get('products', [])]
            modifiers = [m['species'] for m in reaction.get('modifiers', [])]
            
            # Determine reaction type and create descriptive name
            if not products and reactants:
                # Degradation: X -> ∅
                return f"deg_{safe_name(reactants[0])}"
            elif not reactants and products:
                prod = products[0]
                # Check if product is protein (typically starts with P or has P prefix)
                if prod.startswith('P') and len(prod) > 1:
                    # Translation: ∅ -> Protein (e.g., PX)
                    return f"transl_{safe_name(prod)}"
                else:
                    # Transcription: ∅ -> mRNA (e.g., X)
                    return f"transc_{safe_name(prod)}"
            elif reactants and products:
                # Conversion: A -> B
                return f"conv_{safe_name(reactants[0])}to{safe_name(products[0])}"
            else:
                # Fallback
                return f"T{index+1}"
        
        # Create safe name mappings
        safe_species_names = {s['id']: safe_name(s['id']) for s in species_list}
        safe_reaction_names = {r['id']: descriptive_reaction_name(r, i) for i, r in enumerate(reactions_list)}
        
        # Build mappings (1-indexed for GreatSPN)
        place_map = {s['id']: i + 1 for i, s in enumerate(species_list)}
        
        # Analyze reactions to categorize modifiers
        reaction_info = self._analyze_reactions_for_greatspn(reactions_list, parameters)
        
        # Count elements
        num_places = len(species_list)
        num_trans = len(reactions_list)
        num_rate_pars = 0  # Using numeric values directly in transitions
        num_mark_pars = 0
        num_groups = 1  # All timed transitions in one group
        
        # GreatSPN uses inches for coordinates. Typical visible range is 0-15 inches.
        # We need to normalize all positions to fit in a small grid
        SCALE = 0.012  # Scale down coordinates to fit in ~10 inch range
        OFFSET_X = 1.0  # Base offset in inches
        OFFSET_Y = 1.0
        
        # ===== Write .net file =====
        with open(net_path, 'w') as f:
            # Header
            f.write("|0|\n")
            f.write("|\n")
            
            # Format line: f <mark_pars> <places> <rate_pars> <trans> <groups> 0 0
            f.write(f"f {num_mark_pars} {num_places} {num_rate_pars} {num_trans} {num_groups} 0 0\n")
            
            # Places: name initial_mark x y label_x label_y layer
            for i, species in enumerate(species_list):
                orig_x, orig_y = self.positions.get(species['id'], (100 + i * 100, 100))
                x = OFFSET_X + orig_x * SCALE
                y = OFFSET_Y + orig_y * SCALE
                init_mark = int(float(species['initial_amount']))
                safe_id = safe_species_names[species['id']]
                f.write(f"{safe_id} {init_mark} {x:.2f} {y:.2f} {x+0.5:.2f} {y-0.3:.2f} 0\n")
            
            # No rate parameters section (using numeric values directly in transitions)
            
            # Groups: one group for all exponential transitions
            f.write(f"G1 0.0 0.0 0\n")
            
            # Transitions
            for i, reaction in enumerate(reactions_list):
                t_id = reaction['id']
                safe_t_id = safe_reaction_names[t_id]
                orig_x, orig_y = self.positions.get(t_id, (200, 100 + i * 100))
                x = OFFSET_X + orig_x * SCALECompare
                y = OFFSET_Y + orig_y * SCALE
                
                # Get rate value directly (numeric, not parameter reference)
                info = reaction_info[reaction['id']]
                rate_value = info['base_rate']
                
                # Build arcs - only reactants as input, products as output
                # Catalytic modifiers get read arcs, regulatory modifiers just affect rate
                input_arcs = {}  # Use dict to merge arcs to same place: p_idx -> multiplicity
                output_arcs = {}
                inhibitor_arcs = {}
                
                # Reactants are consumed
                for reactant in reaction.get("reactants", []):
                    p_idx = place_map[reactant['species']]
                    mult = int(reactant.get('stoichiometry', 1))
                    input_arcs[p_idx] = input_arcs.get(p_idx, 0) + mult
                
                # Catalytic modifiers (like mRNA in translation) - read arcs
                for mod_species in info.get('catalytic_modifiers', []):
                    p_idx = place_map[mod_species]
                    # Only add if not already in input_arcs (avoid duplicate)
                    if p_idx not in input_arcs:
                        input_arcs[p_idx] = 1
                        output_arcs[p_idx] = output_arcs.get(p_idx, 0) + 1
                
                # Products are produced
                for product in reaction.get("products", []):
                    p_idx = place_map[product['species']]
                    mult = int(product.get('stoichiometry', 1))
                    output_arcs[p_idx] = output_arcs.get(p_idx, 0) + mult
                
                # Regulatory modifiers (repressors/activators) - no arcs, just rate dependency
                # They're handled through marking-dependent rates in .def
                
                # Convert dicts to lists for output
                input_arc_list = [(mult, p_idx) for p_idx, mult in input_arcs.items()]
                output_arc_list = [(mult, p_idx) for p_idx, mult in output_arcs.items()]
                inhibitor_arc_list = [(mult, p_idx) for p_idx, mult in inhibitor_arcs.items()]
                
                num_input_arcs = len(input_arc_list)
                
                # Transition line format:
                # name rate enabling_degree priority num_input_arcs orientation x y tag_x tag_y rate_x rate_y layer
                # Rate must be a numeric value (not a parameter name)
                f.write(f"{safe_t_id} {rate_value:.6e} 1 0 {num_input_arcs} 0 {x:.2f} {y:.2f} {x+0.5:.2f} {y-0.3:.2f} {x+0.5:.2f} {y+0.3:.2f} 0\n")
                
                # Input arcs
                for mult, p_idx in input_arc_list:
                    f.write(f"   {mult} {p_idx} 0 0\n")
                
                # Output arcs
                f.write(f"   {len(output_arc_list)}\n")
                for mult, p_idx in output_arc_list:
                    f.write(f"   {mult} {p_idx} 0 0\n")
                
                # Inhibitor arcs
                f.write(f"   {len(inhibitor_arc_list)}\n")
                for mult, p_idx in inhibitor_arc_list:
                    f.write(f"   {mult} {p_idx} 0 0\n")
        
        # ===== Write .def file =====
        with open(def_path, 'w') as f:
            # Header
            f.write("|256\n")
            f.write("%\n")
            f.write("% Definition file for " + self.model_id + "\n")
            f.write("% Generated from SBML model\n")
            f.write("%\n")
            
            # Generate marking-dependent rate definitions
            f.write("% Marking-dependent rate definitions:\n")
            f.write("%\n")
            
            for i, reaction in enumerate(reactions_list):
                info = reaction_info[reaction['id']]
                rate_name = f"r{i+1}"
                
                if info['marking_dependent_expr']:
                    # Complex rate that depends on place markings
                    f.write(f"% {reaction['id']}: {info['kinetic_law']}\n")
                    f.write(f"{rate_name} = {info['marking_dependent_expr']};\n")
                else:
                    # Simple constant rate
                    f.write(f"% {reaction['id']}: {info['kinetic_law']}\n")
                    f.write(f"{rate_name} = {info['base_rate']:.6e};\n")
                f.write("%\n")
            
            f.write("|\n")
        
        print(f"GreatSPN .net file generated at: {os.path.relpath(net_path)}")
        print(f"GreatSPN .def file generated at: {os.path.relpath(def_path)}")
        
        return net_path, def_path

    def _analyze_reactions_for_greatspn(self, reactions_list: List[Dict], parameters: Dict[str, float]) -> Dict:
        """
        Analyzes reactions to determine:
        - Base firing rate
        - Whether rate is marking-dependent
        - Which modifiers are catalytic vs regulatory
        
        Returns dict mapping reaction_id to analysis info.
        """
        import re
        
        result = {}
        
        for reaction in reactions_list:
            r_id = reaction['id']
            kinetic_law = reaction.get('kinetic_law', '')
            modifiers = [m['species'] for m in reaction.get('modifiers', [])]
            reactants = [r['species'] for r in reaction.get('reactants', [])]
            products = [p['species'] for p in reaction.get('products', [])]
            
            info = {
                'kinetic_law': kinetic_law,
                'base_rate': 1.0,
                'marking_dependent_expr': None,
                'catalytic_modifiers': [],
                'regulatory_modifiers': []
            }
            
            # Analyze the kinetic law to categorize
            if kinetic_law:
                # Check for Hill functions (repression/activation)
                has_hill = 'pow' in kinetic_law and ('/' in kinetic_law or '+' in kinetic_law)
                
                # Check which modifiers appear in denominator (regulatory)
                # vs those that are just multiplied (catalytic)
                
                for mod in modifiers:
                    if mod in kinetic_law:
                        # Check if it's in a Hill function denominator (regulatory)
                        # Pattern: pow(KM, n) / (pow(KM, n) + pow(MOD, n)) for repression
                        # or: pow(MOD, n) / (pow(KM, n) + pow(MOD, n)) for activation
                        hill_pattern = rf'pow\s*\(\s*{mod}\s*,\s*\w+\s*\)'
                        if re.search(hill_pattern, kinetic_law):
                            info['regulatory_modifiers'].append(mod)
                        elif f'* {mod}' in kinetic_law or f'*{mod}' in kinetic_law or kinetic_law.startswith(mod):
                            # Catalytic: appears as a simple multiplier
                            info['catalytic_modifiers'].append(mod)
                        else:
                            # Default: treat as catalytic
                            info['catalytic_modifiers'].append(mod)
                
                # Calculate base rate from kinetic law
                info['base_rate'] = self._extract_base_rate(kinetic_law, parameters)
                
                # Generate marking-dependent expression for GreatSPN
                if info['regulatory_modifiers'] or has_hill:
                    info['marking_dependent_expr'] = self._convert_kinetic_to_greatspn(
                        kinetic_law, parameters, reactants, info['catalytic_modifiers']
                    )
            
            result[r_id] = info
        
        return result
    
    def _extract_base_rate(self, kinetic_law: str, parameters: Dict[str, float]) -> float:
        """Extract a reasonable base rate from a kinetic law expression."""
        import re
        
        # Try to find rate constants (k_*, kd_*, a_*, etc.)
        rate_patterns = [
            r'\b(k_?\w+)\b',
            r'\b(kd_?\w+)\b', 
            r'\b(a_?\w+)\b',
            r'\b(v_?\w+)\b'
        ]
        
        for pattern in rate_patterns:
            matches = re.findall(pattern, kinetic_law)
            for match in matches:
                if match in parameters:
                    return parameters[match]
        
        # If no parameter found, try to evaluate the expression at nominal values
        try:
            eval_env = parameters.copy()
            # Add common defaults
            for species in self.data.get('species', []):
                if species['id'] not in eval_env:
                    eval_env[species['id']] = max(1.0, float(species.get('initial_amount', 1)))
            
            # Safe math functions
            import math
            eval_env['pow'] = pow
            eval_env['exp'] = math.exp
            eval_env['log'] = math.log
            eval_env['sqrt'] = math.sqrt
            
            result = eval(kinetic_law, {"__builtins__": {}}, eval_env)
            return max(0.001, float(result))
        except:
            return 1.0
    
    def _convert_kinetic_to_greatspn(self, kinetic_law: str, parameters: Dict[str, float],
                                      reactants: List[str], catalytic_mods: List[str]) -> str:
        """
        Convert SBML kinetic law to GreatSPN marking-dependent expression.
        GreatSPN uses #place_name to refer to current marking of a place.
        """
        import re
        
        expr = kinetic_law
        
        # Replace species names with GreatSPN place marking references
        species_list = self.data.get('species', [])
        species_names = [s['id'] for s in species_list]
        
        # Sort by length (longest first) to avoid partial replacements
        for species in sorted(species_names, key=len, reverse=True):
            # Replace species name with #species (GreatSPN marking reference)
            # But be careful not to replace inside parameter names
            pattern = rf'\b{re.escape(species)}\b'
            expr = re.sub(pattern, f'#{species}', expr)
        
        # Replace parameters with their values
        for param, value in parameters.items():
            pattern = rf'\b{re.escape(param)}\b'
            expr = re.sub(pattern, f'{value:.6e}', expr)
        
        # Convert pow(a, b) to power notation
        expr = re.sub(r'pow\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', r'pow(\1,\2)', expr)
        
        return expr

    def _calculate_transition_rates(self, reactions_list: List[Dict], parameters: Dict[str, float]) -> Dict[str, float]:
        """
        Calculates base transition rates from kinetic laws.
        For marking-dependent rates, we extract the base rate constant.
        
        Returns a dictionary mapping reaction_id to rate value.
        """
        import re
        
        rates = {}
        
        for reaction in reactions_list:
            kinetic_law = reaction.get('kinetic_law', '')
            reaction_id = reaction['id']
            
            if not kinetic_law:
                # Default rate if no kinetic law
                rates[reaction_id] = 1.0
                continue
            
            # Try to extract the rate constant from the kinetic law
            # Common patterns:
            # - "k * X" -> rate = k
            # - "kd_mRNA * X" -> rate = kd_mRNA
            # - "k_tl * X" -> rate = k_tl  
            # - "a0_tr + a_tr * ..." -> more complex, use a0_tr + a_tr as base
            
            rate = self._extract_rate_from_kinetic_law(kinetic_law, parameters)
            rates[reaction_id] = rate
        
        return rates

    def _extract_rate_from_kinetic_law(self, kinetic_law: str, parameters: Dict[str, float]) -> float:
        """
        Attempts to extract a base rate constant from a kinetic law expression.
        For simple mass-action kinetics (k * S), returns k.
        For more complex expressions, tries to find the dominant rate.
        """
        import re
        
        # Clean up the expression
        expr = kinetic_law.strip()
        
        # Pattern 1: Simple mass action "param * species" or "param * species1 * species2"
        # Match parameter at the start
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\*', expr)
        if match:
            param_name = match.group(1)
            if param_name in parameters:
                return parameters[param_name]
        
        # Pattern 2: Hill function "a0 + a * K^n / (K^n + S^n)"
        # Try to find parameters in the expression
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)
        
        # Find the first token that is a known parameter
        for token in tokens:
            if token in parameters and token not in ['pow']:  # Exclude function names
                return parameters[token]
        
        # Pattern 3: Try to evaluate numerically if it's a constant expression
        try:
            # Replace known parameters
            eval_expr = expr
            for param_name, param_value in parameters.items():
                eval_expr = re.sub(r'\b' + param_name + r'\b', str(param_value), eval_expr)
            
            # Replace pow() with ** for Python eval
            eval_expr = re.sub(r'pow\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', r'(\1)**(\2)', eval_expr)
            
            # If no species variables remain, try to evaluate
            remaining_vars = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', eval_expr)
            remaining_vars = [v for v in remaining_vars if v not in ['pow', 'exp', 'log', 'sqrt']]
            
            if not remaining_vars:
                result = eval(eval_expr)
                if isinstance(result, (int, float)) and result > 0:
                    return float(result)
        except:
            pass
        
        # Default: return 1.0 as a fallback
        # This will require manual adjustment in GreatSPN
        return 1.0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_petri_net.py <path_to_parsed_json>")
        sys.exit(1)

    json_file = sys.argv[1]
    generator = PetriNetGenerator(json_file)
    
    # Generate Image
    generator.generate_image()
    
    # Generate PNML
    generator.generate_pnml()
    
    # Generate GreatSPN native format
    generator.generate_greatspn()
