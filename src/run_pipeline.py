import sys
import os
from fetch_biomodel import BioModelAgent
from parse_sbml import SBMLParser
from generate_petri_net import PetriNetGenerator
from simulate_circuit import GillespieSimulator, ODESimulator, choose_simulator

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py [duration] [num_runs] [method] <circuit_name_or_biomd_id>")
        print("  duration: simulation duration (default: 500)")
        print("  num_runs: number of independent runs for Gillespie (default: 1)")
        print("  method: 'gillespie', 'ode', or 'auto' (default: auto)")
        print("Examples:")
        print("  python run_pipeline.py 'toggle switch'")
        print("  python run_pipeline.py 50 'repressilator'")
        print("  python run_pipeline.py 100 20 'toggle switch'")
        print("  python run_pipeline.py 3000 1 ode 'circadian'")
        sys.exit(1)

    # Parse arguments: [duration] [num_runs] [method] <model_name>
    duration = 500.0
    num_runs = 1
    method = 'auto'
    
    args = sys.argv[1:]
    
    # Check if first arg is a number (duration)
    try:
        duration = float(args[0])
        args = args[1:]
        
        # Check if second arg is also a number (num_runs)
        try:
            num_runs = int(args[0])
            args = args[1:]
            
            # Check if third arg is a method
            if args and args[0].lower() in ['gillespie', 'ode', 'auto']:
                method = args[0].lower()
                args = args[1:]
        except (ValueError, IndexError):
            # Maybe it's the method?
            if args and args[0].lower() in ['gillespie', 'ode', 'auto']:
                method = args[0].lower()
                args = args[1:]
    except ValueError:
        # Maybe first arg is the method?
        if args[0].lower() in ['gillespie', 'ode', 'auto']:
            method = args[0].lower()
            args = args[1:]
    
    user_input = " ".join(args)
    if not user_input:
        print("Error: No model name or ID provided.")
        sys.exit(1)
        
    print(f"--- Starting Pipeline for: {user_input} ---")
    print(f"    Duration: {duration}, Number of runs: {num_runs}, Method: {method}")

    # Step 1: Fetch Model
    print("\n[Step 1] Fetching Model...")
    agent = BioModelAgent()
    biomd_id, common_name = agent.resolve_input(user_input)
    
    if not biomd_id or not common_name:
        print("Failed to resolve model. Exiting.")
        sys.exit(1)

    xml_path = agent.download_model(biomd_id, common_name)
    if not xml_path:
        print("Failed to download model. Exiting.")
        sys.exit(1)

    # Step 2: Parse SBML
    print("\n[Step 2] Parsing SBML...")
    parser = SBMLParser(xml_path)
    parsed_data = parser.parse()
    
    if not parsed_data:
        print("Failed to parse SBML. Exiting.")
        sys.exit(1)

    # Construct JSON path in the same directory
    json_path = os.path.splitext(xml_path)[0] + "_parsed.json"
    parser.save_to_json(parsed_data, json_path)

    # Step 3: Generate Petri Net
    print("\n[Step 3] Generating Petri Net...")
    generator = PetriNetGenerator(json_path)
    
    image_path = generator.generate_image()
    pnml_path = generator.generate_pnml()
    net_path, def_path = generator.generate_greatspn()

    # Step 4: Run Simulation (Gillespie or ODE)
    print("\n[Step 4] Running Simulation...")
    
    # Auto-select method if needed
    if method == 'auto':
        method, reason = choose_simulator(json_path)
        print(f"    Auto-selected method: {method.upper()} - {reason}")
    
    if method == 'ode':
        print("    Using ODE (deterministic) simulation...")
        simulator = ODESimulator(json_path)
        simulator.run_and_plot(duration=duration)
    else:
        print("    Using Gillespie (stochastic) simulation...")
        simulator = GillespieSimulator(json_path)
        simulator.run_comparative_analysis(duration=duration, num_runs=num_runs)

    print("\n--- Pipeline Completed Successfully ---")
    print(f"XML File: {os.path.relpath(xml_path)}")
    print(f"JSON File: {os.path.relpath(json_path)}")
    print(f"Petri Net Image: {os.path.relpath(image_path)}")
    print(f"PNML File: {os.path.relpath(pnml_path)}")
    print(f"GreatSPN Files: {os.path.relpath(net_path)}, {os.path.relpath(def_path)}")
    # Note: The simulation plot path is printed by the simulator itself

if __name__ == "__main__":
    main()
