import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import re
import random
import warnings
import math
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Suppress numpy warnings globally for this module (division by zero is handled in code)
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')  # Suppress all numpy floating-point errors


# ============================================================================
# MODULE-LEVEL WORKER FUNCTION FOR PARALLEL PROCESSING
# Must be at module level for pickle serialization (ProcessPoolExecutor)
# ============================================================================

def _gillespie_worker(args: Tuple) -> Dict[str, Any]:
    """
    Standalone Gillespie simulation worker for ProcessPoolExecutor.
    
    This function is at module level to allow pickle serialization.
    It receives all necessary data as arguments (no class instance).
    """
    (duration, label, initial_state, parameters, reactions, 
     function_defs, assignment_rules, max_steps) = args
    
    # Suppress warnings in worker process
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    np.seterr(all='ignore')
    
    time = 0.0
    state = initial_state.copy()
    
    times = [0.0]
    history = {sp: [count] for sp, count in state.items()}
    
    step_count = 0
    use_tau_leaping = False
    
    # Helper: evaluate assignment rules
    def evaluate_rules(ctx):
        for rule in assignment_rules:
            var = rule.get('variable')
            formula = rule.get('formula', '')
            if var and formula:
                try:
                    ctx[var] = float(eval(formula, {"__builtins__": {}, "math": math}, ctx))
                except:
                    pass
        return ctx
    
    # Helper: create callable from function definition
    def make_func(arg_names, formula, base_ctx):
        def fn(*args):
            if formula == 'NaN' or formula is None:
                return 0.0
            local = base_ctx.copy()
            for name, val in zip(arg_names, args):
                local[name] = val
            try:
                result = eval(formula, {"__builtins__": {}, "math": math}, local)
                if np.isnan(result) or np.isinf(result):
                    return 0.0
                return float(result)
            except:
                return 0.0
        return fn
    
    # Build function objects
    func_objects = {}
    for fname, fdef in function_defs.items():
        body = fdef.get('formula', fdef.get('body', '0'))
        func_objects[fname] = make_func(fdef.get('arguments', []), body, {})
    
    while time < duration and step_count < max_steps:
        # Build context
        context = {**parameters, **state}
        context.update({k: getattr(np, k) for k in ['exp', 'log', 'sqrt', 'sin', 'cos', 'pow', 'floor', 'ceil', 'abs']})
        context['math'] = math
        context = evaluate_rules(context)
        context.update(func_objects)
        
        # Calculate propensities
        propensities = []
        for rxn in reactions:
            ctx = {**context, **rxn.get('local_params', {})}
            try:
                rate = eval(rxn['formula'], {"__builtins__": {}, "math": math}, ctx)
                if np.isnan(rate) or np.isinf(rate) or rate < 0:
                    rate = 0.0
                rate = min(float(rate), 1e10)
                propensities.append(rate)
            except:
                propensities.append(0.0)
        
        propensities = np.array(propensities)
        a_total = propensities.sum()
        
        if a_total <= 0 or np.isnan(a_total) or np.isinf(a_total):
            break
        
        # Use tau-leaping for very fast reactions
        if a_total > 50000:
            use_tau_leaping = True
        
        if use_tau_leaping:
            tau = min(0.01, duration - time)
            for i, rxn in enumerate(reactions):
                if propensities[i] > 0:
                    expected = propensities[i] * tau
                    if expected > 0:
                        num_fire = np.random.poisson(expected)
                        for _ in range(num_fire):
                            for r in rxn.get('reactants', []):
                                stoich = r.get('stoichiometry', 1.0)
                                state[r['species']] = max(0, state[r['species']] - stoich)
                            for p in rxn.get('products', []):
                                stoich = p.get('stoichiometry', 1.0)
                                state[p['species']] += stoich
            time += tau
            step_count += 1
        else:
            # Standard Gillespie SSA
            tau = (1.0 / a_total) * np.log(1.0 / random.random())
            if np.isnan(tau) or np.isinf(tau) or tau <= 0:
                tau = 1e-9
            
            time += tau
            
            threshold = random.random() * a_total
            cumsum = 0.0
            rxn_idx = -1
            for i, p in enumerate(propensities):
                cumsum += p
                if cumsum >= threshold:
                    rxn_idx = i
                    break
            
            if rxn_idx == -1:
                break
            
            rxn = reactions[rxn_idx]
            for r in rxn.get('reactants', []):
                stoich = r.get('stoichiometry', 1.0)
                state[r['species']] = max(0, state[r['species']] - stoich)
            for p in rxn.get('products', []):
                stoich = p.get('stoichiometry', 1.0)
                state[p['species']] += stoich
            
            step_count += 1
        
        # Record history (subsample to save memory)
        if step_count <= 10000 or step_count % 10 == 0:
            times.append(time)
            for sp in history:
                history[sp].append(state[sp])
    
    return {"time": times, "history": history, "label": label}

class GillespieSimulator:
    """
    Performs Gillespie Stochastic Simulations based on parsed SBML data.
    Includes heuristics for missing parameters and generates comparative plots.
    """

    def __init__(self, json_filepath: str):
        self.json_filepath = json_filepath
        self.output_dir = os.path.dirname(os.path.abspath(json_filepath))
        self.data = self._load_json()
        self.model_id = self.data.get("model_id", "model")
        
        # Simulation State
        self.species_state = {}
        self.parameters = {}
        self.reactions = []
        
        self._initialize_model()

    def _load_json(self) -> Dict[str, Any]:
        with open(self.json_filepath, 'r') as f:
            return json.load(f)

    def _initialize_model(self):
        """Initializes state and parameters, applying heuristics for missing values."""
        
        # 1. Load Compartments
        for comp in self.data.get("compartments", []):
            self.parameters[comp['id']] = float(comp['size'])
        
        # 2. Load Species (State) and keep reference to species data
        self.species = self.data.get("species", [])
        for sp in self.species:
            self.species_state[sp['id']] = float(sp['initial_amount'])

        # 3. Load Parameters
        for param in self.data.get("parameters", []):
            self.parameters[param['id']] = float(param['value'])
        
        # 4. Load Function Definitions (SBML user-defined functions)
        self.function_defs = self.data.get("function_definitions", {})
        
        # 5. Load Assignment Rules (for dynamic parameter updates during simulation)
        self.assignment_rules = self.data.get("assignment_rules", [])

        # 6. Load Reactions and Check for Missing Parameters
        for rxn in self.data.get("reactions", []):
            # Extract local parameters if any
            local_params = {p['id']: float(p['value']) for p in rxn.get("local_parameters", [])}
            
            # Combine global and local for this reaction context
            context = {**self.parameters, **local_params, **self.species_state}
            
            # Analyze kinetic law for missing variables
            formula = rxn.get('kinetic_law', '')
            if not formula:
                print(f"Warning: Reaction {rxn['id']} has no kinetic law. Skipping.")
                continue
                
            # Heuristic: Identify tokens in formula
            # This regex finds words that are valid identifiers
            tokens = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', formula))
            
            # Standard math functions and SBML function definitions to ignore
            math_funcs = {'exp', 'log', 'sin', 'cos', 'pow', 'sqrt', 'floor', 'ceil', 'abs', 'min', 'max'}
            sbml_funcs = set(self.function_defs.keys())
            known_funcs = math_funcs | sbml_funcs
            
            missing = []
            for token in tokens:
                if token not in context and token not in known_funcs:
                    missing.append(token)
            
            # Apply AI Heuristic for missing parameters
            for m in missing:
                guessed_value = self._heuristic_guess(m)
                print(f"(!) Missing parameter '{m}' in reaction '{rxn['id']}'. AI Heuristic assigned: {guessed_value}")
                self.parameters[m] = guessed_value # Add to global parameters
            
            self.reactions.append({
                "id": rxn['id'],
                "formula": formula,
                "products": rxn.get("products", []),
                "reactants": rxn.get("reactants", []),
                "local_params": local_params
            })

    def _heuristic_guess(self, name: str) -> float:
        """
        Uses simple heuristics to guess parameter values based on naming conventions.
        In a real scenario, this could call an LLM.
        """
        name_lower = name.lower()
        if "deg" in name_lower or "decay" in name_lower:
            return 0.1  # Typical degradation rate
        if "prod" in name_lower or "syn" in name_lower or "transcri" in name_lower:
            return 10.0 # Typical production rate
        if "bind" in name_lower:
            return 1.0
        if "dissoc" in name_lower:
            return 0.5
        return 1.0 # Default fallback

    def _evaluate_assignment_rules(self, context: Dict[str, float]) -> Dict[str, float]:
        """Evaluates SBML assignment rules to update dynamic parameters."""
        # Assignment rules define parameters that depend on species concentrations
        # They must be re-evaluated at each simulation step
        for rule in self.assignment_rules:
            variable = rule['variable']
            formula = rule['formula']
            try:
                value = eval(formula, {}, context)
                context[variable] = float(value)
            except Exception:
                pass  # Keep previous value if evaluation fails
        return context

    def _calculate_propensities(self, current_state: Dict[str, float]) -> Tuple[np.ndarray, float]:
        """Calculates the propensity (rate) for each reaction."""
        propensities = []
        
        # Context for eval()
        # Note: eval is used for flexibility with arbitrary SBML formulas. 
        # In production, use a safer parser.
        context = {**self.parameters, **current_state}
        # Add math functions
        context.update({k: getattr(np, k) for k in dir(np) if not k.startswith('_')})
        
        # IMPORTANT: Evaluate assignment rules BEFORE calculating propensities
        # This updates dynamic parameters that depend on current species concentrations
        context = self._evaluate_assignment_rules(context)
        
        # Add user-defined functions from SBML
        for func_id, func_info in self.function_defs.items():
            # Create a callable function from the definition
            args = func_info.get('arguments', [])
            formula = func_info.get('formula', '0')
            # Skip invalid function definitions
            if formula == 'NaN' or formula is None:
                context[func_id] = lambda *args: 0.0
            else:
                # Create a lambda that evaluates the formula with given arguments
                context[func_id] = self._create_function(args, formula, context)
        
        for rxn in self.reactions:
            # Update context with local params if needed (though we added them to global or handled them)
            # Ideally local params should override global, but for simplicity we assume unique names or global precedence
            # Re-inject local params to ensure they are available
            context.update(rxn['local_params'])
            
            try:
                # SBML formulas often use 'time', we'll set it to 0 or current time if needed, 
                # but standard Gillespie is time-homogeneous usually.
                with np.errstate(divide='ignore', invalid='ignore'):
                    rate = eval(rxn['formula'], {}, context)
                # Propensity must be non-negative and finite
                if np.isnan(rate) or np.isinf(rate) or rate < 0:
                    rate = 0.0
                # Cap extremely large propensities to prevent numerical issues
                rate = min(float(rate), 1e10)
                propensities.append(rate)
            except Exception as e:
                # print(f"Error evaluating formula for {rxn['id']}: {e}")
                propensities.append(0.0)
                
        return np.array(propensities), sum(propensities)
    
    def _create_function(self, arg_names: List[str], formula: str, base_context: Dict) -> callable:
        """Creates a callable function from SBML function definition."""
        def func(*args):
            # Handle NaN formula (some SBML models have this for rateOf)
            if formula == 'NaN' or formula is None:
                return 0.0
            # Build local context with argument values
            local_ctx = base_context.copy()
            for name, value in zip(arg_names, args):
                local_ctx[name] = value
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    result = eval(formula, {}, local_ctx)
                if np.isnan(result) or np.isinf(result):
                    return 0.0
                return float(result)
            except Exception:
                return 0.0
        return func

    def run_simulation(self, duration: float, label: str = "Simulation", max_steps: int = 500000) -> Dict[str, Any]:
        """Runs a single stochastic simulation using Gillespie SSA with tau-leaping for fast systems."""
        time = 0.0
        # Deep copy initial state
        state = self.species_state.copy()
        
        times = [0.0]
        # Track history for all species
        history = {sp: [count] for sp, count in state.items()}
        
        step_count = 0
        last_progress = 0
        warned_slow = False
        use_tau_leaping = False
        
        while time < duration and step_count < max_steps:
            propensities, a_total = self._calculate_propensities(state)
            
            if a_total <= 0 or np.isnan(a_total) or np.isinf(a_total):
                break
            
            # Warn if propensities are very high (simulation will be slow)
            # Switch to tau-leaping for very fast reactions
            if a_total > 50000:
                if not warned_slow:
                    print(f"    Note: High reaction rates (a_total={a_total:.1f}). Using tau-leaping approximation...")
                    warned_slow = True
                use_tau_leaping = True
            
            if use_tau_leaping:
                # Tau-leaping: use fixed time step and approximate number of reactions
                tau = min(0.01, duration / 1000)  # Small fixed step
                
                # For each reaction, sample from Poisson distribution
                for i, rxn in enumerate(self.reactions):
                    if propensities[i] > 0:
                        # Expected number of reactions in tau time
                        expected = propensities[i] * tau
                        # Sample actual number from Poisson
                        num_reactions = np.random.poisson(expected)
                        
                        if num_reactions > 0:
                            # Apply the reaction num_reactions times
                            for r in rxn['reactants']:
                                stoich = r.get('stoichiometry', 1.0)
                                state[r['species']] = max(0, state[r['species']] - stoich * num_reactions)
                            for p in rxn['products']:
                                stoich = p.get('stoichiometry', 1.0)
                                state[p['species']] += stoich * num_reactions
                
                time += tau
                step_count += 1
            else:
                # Standard Gillespie SSA
                # 1. Determine time to next reaction (tau)
                r1 = np.random.random()
                if r1 <= 0:
                    r1 = 1e-10  # Avoid log(0)
                tau = (1.0 / a_total) * np.log(1.0 / r1)
            
                # Safety: skip if tau is invalid or too small (prevents infinite loops)
                if np.isnan(tau) or np.isinf(tau) or tau <= 0:
                    break
                
                # If tau is extremely small, skip ahead to avoid getting stuck
                if tau < 1e-12:
                    time += 1e-9
                    continue
                
                # 2. Determine which reaction occurs
                r2 = np.random.random()
                threshold = r2 * a_total
                current_sum = 0.0
                reaction_idx = -1
                
                for i, p in enumerate(propensities):
                    current_sum += p
                    if current_sum >= threshold:
                        reaction_idx = i
                        break
                
                if reaction_idx == -1:
                    break
                    
                # 3. Update State
                rxn = self.reactions[reaction_idx]
                
                # Consume reactants
                for r in rxn['reactants']:
                    stoich = r.get('stoichiometry', 1.0)
                    state[r['species']] = max(0, state[r['species']] - stoich)
                    
                # Produce products
                for p in rxn['products']:
                    stoich = p.get('stoichiometry', 1.0)
                    state[p['species']] += stoich
                    
                # Update time
                time += tau
                step_count += 1
            
            # Progress reporting (every 10% of duration)
            progress = int((time / duration) * 10)
            if progress > last_progress and progress <= 10:
                last_progress = progress
            
            # Record history (sample every N steps if too many)
            # To prevent memory issues, we can subsample
            if step_count <= 10000 or step_count % 10 == 0:
                times.append(time)
                for sp in history:
                    history[sp].append(state[sp])
        
        if step_count >= max_steps:
            print(f"    Warning: Reached max steps ({max_steps}). Simulation may be incomplete.")
                
        return {"time": times, "history": history, "label": label}

    def _run_single_simulation(self, duration: float, label: str, initial_state: Dict[str, float]) -> Dict[str, Any]:
        """Thread-safe simulation runner for parallel execution.
        
        Creates isolated state for the simulation to avoid race conditions.
        
        Args:
            duration: Simulation duration
            label: Label for the simulation result
            initial_state: Dictionary of species initial amounts
        
        Returns:
            Simulation result dictionary with time, history, and label
        """
        # Create a local copy of state for thread safety
        time = 0.0
        state = initial_state.copy()
        
        times = [0.0]
        history = {sp: [count] for sp, count in state.items()}
        
        step_count = 0
        max_steps = 500000
        use_tau_leaping = False
        
        while time < duration and step_count < max_steps:
            propensities, a_total = self._calculate_propensities(state)
            
            if a_total <= 0 or np.isnan(a_total) or np.isinf(a_total):
                break
            
            # Switch to tau-leaping for very fast reactions
            if a_total > 50000:
                use_tau_leaping = True
            
            if use_tau_leaping:
                # Tau-leaping: use a fixed time step
                tau = min(0.01, duration - time)
                
                # Fire reactions according to expected number
                for i, rxn in enumerate(self.reactions):
                    if propensities[i] > 0:
                        expected_firings = propensities[i] * tau
                        # Sample from Poisson distribution
                        if expected_firings > 0:
                            num_firings = np.random.poisson(expected_firings)
                            for _ in range(num_firings):
                                for r in rxn['reactants']:
                                    stoich = r.get('stoichiometry', 1.0)
                                    state[r['species']] = max(0, state[r['species']] - stoich)
                                for p in rxn['products']:
                                    stoich = p.get('stoichiometry', 1.0)
                                    state[p['species']] += stoich
                
                time += tau
                step_count += 1
            else:
                # Standard Gillespie SSA
                tau = (1.0 / a_total) * np.log(1.0 / random.random())
                
                if np.isnan(tau) or np.isinf(tau) or tau <= 0:
                    tau = 1e-9
                
                time += tau
                
                threshold = random.random() * a_total
                current_sum = 0.0
                reaction_idx = -1
                
                for i, p in enumerate(propensities):
                    current_sum += p
                    if current_sum >= threshold:
                        reaction_idx = i
                        break
                
                if reaction_idx == -1:
                    break
                
                rxn = self.reactions[reaction_idx]
                
                for r in rxn['reactants']:
                    stoich = r.get('stoichiometry', 1.0)
                    state[r['species']] = max(0, state[r['species']] - stoich)
                
                for p in rxn['products']:
                    stoich = p.get('stoichiometry', 1.0)
                    state[p['species']] += stoich
                
                step_count += 1
            
            # Record history (subsample if too many steps)
            if step_count <= 10000 or step_count % 10 == 0:
                times.append(time)
                for sp in history:
                    history[sp].append(state[sp])
        
        return {"time": times, "history": history, "label": label}

    def _identify_input_species(self) -> List[str]:
        """
        Identifies 'input' species - those that can be set by the user to control circuit behavior.
        These are species that:
        1. Are not produced or consumed by any reaction (boundary conditions)
        2. Appear in assignment rules (affecting dynamic parameters)
        3. Have boundaryCondition=true in SBML
        
        Common examples: IPTG, aTc (anhydrotetracycline), arabinose, etc.
        """
        # Find all species that participate in reactions as reactants or products
        reacting_species = set()
        for rxn in self.reactions:
            for r in rxn.get('reactants', []):
                reacting_species.add(r['species'])
            for p in rxn.get('products', []):
                reacting_species.add(p['species'])
        
        # Input species are those NOT in reacting_species
        all_species = set(sp['id'] for sp in self.species)
        input_species = all_species - reacting_species
        
        # Also check for species mentioned in assignment rules
        for rule in self.assignment_rules:
            formula = rule['formula']
            for sp in self.species:
                if sp['id'] in formula and sp['id'] not in reacting_species:
                    input_species.add(sp['id'])
        
        return list(input_species)

    def run_comparative_analysis(self, duration: float = 100.0, num_runs: int = 1):
        """
        Runs simulations. If num_runs > 1, demonstrates stochastic variability.
        Automatically detects input species and tests different input levels.
        """
        # Store original state
        original_state = self.species_state.copy()
        
        # Check if concentrations are too small for Gillespie (needs discrete counts)
        max_initial = max(original_state.values()) if original_state else 0
        if max_initial < 10:
            print(f"Warning: Initial species amounts are very small (max={max_initial:.4f}).")
            print("         This model may use concentrations instead of molecule counts.")
            print("         Gillespie SSA works best with discrete molecule numbers (10s-1000s).")
            print("         Consider scaling up initial values or using ODE simulation instead.")
        
        # Get species names
        sp_names = {sp['id']: sp.get('name', sp['id']) for sp in self.species}
        
        # Identify input species (like IPTG)
        input_species = self._identify_input_species()
        
        # Find reacting species (not boundary conditions)
        reacting_species = []
        for rxn in self.reactions:
            for r in rxn.get('reactants', []):
                if r['species'] not in reacting_species:
                    reacting_species.append(r['species'])
            for p in rxn.get('products', []):
                if p['species'] not in reacting_species:
                    reacting_species.append(p['species'])
        
        # If we found input species, run comparative analysis with different input levels
        if input_species:
            input_sp = input_species[0]  # Use first input species
            input_name = sp_names.get(input_sp, input_sp)
            print(f"Detected input species: {input_name} ({input_sp})")
            print(f"Running comparative analysis with different {input_name} levels...")
            
            # Split runs between OFF and ON states (at least 1 each)
            runs_per_state = max(1, num_runs // 2)
            high_value = self._estimate_input_high_value(input_sp)
            
            # Run simulations (parallel if multiple runs)
            if runs_per_state > 1:
                n_workers = min(runs_per_state, os.cpu_count() or 4)
                print(f"\n--- Running {runs_per_state} OFF + {runs_per_state} ON simulations in parallel ({n_workers} processes) ---")
                
                # Prepare worker arguments
                off_args = [self._prepare_worker_args(duration, f"{input_name}=0, Run {i+1}", 
                           {**original_state, input_sp: 0.0}) for i in range(runs_per_state)]
                on_args = [self._prepare_worker_args(duration, f"{input_name}={high_value}, Run {i+1}",
                          {**original_state, input_sp: high_value}) for i in range(runs_per_state)]
                
                # OFF simulations with ProcessPoolExecutor
                runs_off = []
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [executor.submit(_gillespie_worker, args) for args in off_args]
                    for future in as_completed(futures):
                        try:
                            runs_off.append(future.result())
                        except Exception as e:
                            print(f"  Warning: OFF run failed: {e}")
                print(f"  OFF runs completed ({len(runs_off)}/{runs_per_state})")
                
                # ON simulations with ProcessPoolExecutor
                runs_on = []
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [executor.submit(_gillespie_worker, args) for args in on_args]
                    for future in as_completed(futures):
                        try:
                            runs_on.append(future.result())
                        except Exception as e:
                            print(f"  Warning: ON run failed: {e}")
                print(f"  ON runs completed ({len(runs_on)}/{runs_per_state})")
            else:
                # Sequential for single run
                print(f"\n--- {input_name} = 0 (OFF) ---")
                self.species_state = {**original_state, input_sp: 0.0}
                runs_off = [self.run_simulation(duration, label=f"{input_name}=0, Run 1")]
                print(f"  Run 1/1 completed")
                
                print(f"\n--- {input_name} = {high_value} (ON) ---")
                self.species_state = {**original_state, input_sp: high_value}
                runs_on = [self.run_simulation(duration, label=f"{input_name}={high_value}, Run 1")]
                print(f"  Run 1/1 completed")
            
            # Restore original state
            self.species_state = original_state
            
            # Plot comparison
            self._plot_input_comparison(runs_off, runs_on, input_name, reacting_species, sp_names)
        else:
            # No input species found - run standard simulations
            print(f"No input species detected. Running {num_runs} simulation(s)...")
            
            if num_runs == 1:
                # Single run - no parallelization needed
                self.species_state = original_state.copy()
                result = self.run_simulation(duration, label="Run 1")
                all_runs = [result]
                print(f"  Run 1/1 completed")
            else:
                # Multiple runs - use parallel execution with ProcessPoolExecutor
                n_workers = min(num_runs, os.cpu_count() or 4)
                print(f"  Using {n_workers} parallel processes")
                
                # Prepare worker arguments
                all_args = [self._prepare_worker_args(duration, f"Run {i+1}", original_state.copy()) 
                           for i in range(num_runs)]
                
                all_runs = []
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [executor.submit(_gillespie_worker, args) for args in all_args]
                    for future in as_completed(futures):
                        try:
                            all_runs.append(future.result())
                        except Exception as e:
                            print(f"  Warning: Run failed: {e}")
                
                print(f"  All {len(all_runs)}/{num_runs} runs completed")
            
            # Restore original state
            self.species_state = original_state
            
            # Plot results - use single run plot if only 1 run
            if num_runs == 1:
                self._plot_single_run(all_runs[0], reacting_species, sp_names)
            else:
                self._plot_multiple_runs(all_runs, reacting_species, sp_names)

    def _prepare_worker_args(self, duration: float, label: str, initial_state: Dict[str, float]) -> Tuple:
        """Prepares arguments tuple for the module-level worker function."""
        return (
            duration,
            label,
            initial_state,
            self.parameters.copy(),
            self.reactions,  # List of dicts - serializable
            self.function_defs,  # Dict - serializable
            self.assignment_rules,  # List - serializable
            500000  # max_steps
        )

    def _estimate_input_high_value(self, input_sp: str) -> float:
        """
        Estimates an appropriate 'high' value for an input species.
        Looks at assignment rules to find relevant parameters (like K, dissociation constant).
        """
        # Look for the input species in assignment rules
        for rule in self.assignment_rules:
            if input_sp in rule['formula']:
                # Try to find a K or dissociation constant
                # Common pattern: species_X / K or species_X / parameter_Y
                for param_id, param_val in self.parameters.items():
                    if param_id in rule['formula'] and param_val > 0 and param_val < 1:
                        # This is likely a K value - return 10x K for "high" induction
                        return param_val * 100
        
        # Default: return 0.001 (works for most molecular concentrations)
        return 0.001

    def _plot_input_comparison(self, runs_off: List[Dict], runs_on: List[Dict], 
                                input_name: str, reacting_species: List[str], sp_names: Dict[str, str]):
        """Generates plots comparing simulations with input OFF vs ON - shows ALL reacting species."""
        import numpy as np
        
        # Use ALL reacting species, not just first 2
        plot_species = reacting_species if reacting_species else list(runs_off[0]['history'].keys())
        
        # Remove any species that don't change (boundary conditions)
        plot_species = [sp for sp in plot_species 
                       if len(set(runs_off[0]['history'].get(sp, [0]))) > 1]
        
        if not plot_species:
            plot_species = reacting_species[:3] if reacting_species else list(runs_off[0]['history'].keys())[:3]
        
        num_species = len(plot_species)
        
        # Create distinct colors for each species
        species_colors = plt.cm.tab10(np.linspace(0, 1, max(num_species, 2)))
        
        # --- Figure 1: Time series comparison (OFF vs ON) ---
        # Calculate rows needed for species (2 species per row: OFF and ON)
        species_rows = (num_species + 1) // 2  # Round up
        fig1 = plt.figure(figsize=(14, 4 * species_rows + 1))
        
        gs = fig1.add_gridspec(species_rows + 1, 4, height_ratios=[1] * species_rows + [0.5],
                               hspace=0.4, wspace=0.3)
        
        for idx, sp in enumerate(plot_species):
            row = idx // 2
            col_offset = (idx % 2) * 2  # 0 or 2
            sp_name = sp_names.get(sp, sp)
            color = species_colors[idx]
            
            # OFF plot
            ax_off = fig1.add_subplot(gs[row, col_offset])
            for run in runs_off:
                ax_off.step(run['time'], run['history'][sp], where='post', 
                           color=color, alpha=0.3, linewidth=1)
            ax_off.set_xlabel("Time", fontsize=9)
            ax_off.set_ylabel("Count", fontsize=9)
            ax_off.set_title(f"{sp_name} ({input_name}=OFF)", fontsize=10, fontweight='bold')
            ax_off.grid(True, alpha=0.3)
            ax_off.set_xlim(left=0)
            ax_off.set_ylim(bottom=0)
            
            # ON plot
            ax_on = fig1.add_subplot(gs[row, col_offset + 1])
            for run in runs_on:
                ax_on.step(run['time'], run['history'][sp], where='post', 
                          color=color, alpha=0.3, linewidth=1)
            ax_on.set_xlabel("Time", fontsize=9)
            ax_on.set_ylabel("Count", fontsize=9)
            ax_on.set_title(f"{sp_name} ({input_name}=ON)", fontsize=10, fontweight='bold')
            ax_on.grid(True, alpha=0.3)
            ax_on.set_xlim(left=0)
            ax_on.set_ylim(bottom=0)
        
        # Main title
        fig1.suptitle(f"Gillespie Simulation - {self.model_id}\n"
                      f"Input Signal Analysis: {input_name} (All {num_species} Species)",
                      fontsize=13, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        output_file1 = os.path.join(self.output_dir, f"{self.model_id}_input_comparison_timeseries.png")
        plt.savefig(output_file1, dpi=150)
        print(f"\nTime series comparison saved to: {os.path.relpath(output_file1)}")
        plt.close()
        
        # --- Figure 2: Summary statistics ---
        fig2 = plt.figure(figsize=(14, 8))
        
        # Bar chart comparing final values
        ax_bar = fig2.add_subplot(2, 2, 1)
        
        x = np.arange(len(plot_species))
        width = 0.35
        
        means_off = []
        stds_off = []
        means_on = []
        stds_on = []
        
        for sp in plot_species:
            finals_off = [run['history'][sp][-1] for run in runs_off]
            finals_on = [run['history'][sp][-1] for run in runs_on]
            means_off.append(np.mean(finals_off))
            stds_off.append(np.std(finals_off))
            means_on.append(np.mean(finals_on))
            stds_on.append(np.std(finals_on))
        
        ax_bar.bar(x - width/2, means_off, width, yerr=stds_off, label=f'{input_name}=OFF', 
                   color='#3498db', capsize=3, alpha=0.8)
        ax_bar.bar(x + width/2, means_on, width, yerr=stds_on, label=f'{input_name}=ON', 
                   color='#e74c3c', capsize=3, alpha=0.8)
        
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([sp_names.get(sp, sp) for sp in plot_species], rotation=45, ha='right', fontsize=8)
        ax_bar.set_ylabel("Final Molecule Count", fontsize=10)
        ax_bar.set_title("Final Values: OFF vs ON", fontsize=11, fontweight='bold')
        ax_bar.legend()
        ax_bar.grid(True, alpha=0.3, axis='y')
        
        # Combined time series (first run of each)
        ax_combined_off = fig2.add_subplot(2, 2, 2)
        for i, sp in enumerate(plot_species):
            sp_name = sp_names.get(sp, sp)
            run = runs_off[0]
            ax_combined_off.step(run['time'], run['history'][sp], where='post', 
                                color=species_colors[i], alpha=0.8, linewidth=1.5, label=sp_name)
        ax_combined_off.set_xlabel("Time", fontsize=10)
        ax_combined_off.set_ylabel("Molecule Count", fontsize=10)
        ax_combined_off.set_title(f"All Species ({input_name}=OFF)", fontsize=11, fontweight='bold')
        ax_combined_off.legend(loc='upper right', fontsize=7)
        ax_combined_off.grid(True, alpha=0.3)
        
        ax_combined_on = fig2.add_subplot(2, 2, 3)
        for i, sp in enumerate(plot_species):
            sp_name = sp_names.get(sp, sp)
            run = runs_on[0]
            ax_combined_on.step(run['time'], run['history'][sp], where='post', 
                               color=species_colors[i], alpha=0.8, linewidth=1.5, label=sp_name)
        ax_combined_on.set_xlabel("Time", fontsize=10)
        ax_combined_on.set_ylabel("Molecule Count", fontsize=10)
        ax_combined_on.set_title(f"All Species ({input_name}=ON)", fontsize=11, fontweight='bold')
        ax_combined_on.legend(loc='upper right', fontsize=7)
        ax_combined_on.grid(True, alpha=0.3)
        
        # Phase space or scatter
        ax_phase = fig2.add_subplot(2, 2, 4)
        
        if len(plot_species) >= 2:
            sp1, sp2 = plot_species[0], plot_species[1]
            
            # OFF points
            x_off = [run['history'][sp1][-1] for run in runs_off]
            y_off = [run['history'][sp2][-1] for run in runs_off]
            ax_phase.scatter(x_off, y_off, c='#3498db', s=100, alpha=0.7, 
                            label=f'{input_name}=OFF', edgecolors='black', linewidth=0.5)
            
            # ON points
            x_on = [run['history'][sp1][-1] for run in runs_on]
            y_on = [run['history'][sp2][-1] for run in runs_on]
            ax_phase.scatter(x_on, y_on, c='#e74c3c', s=100, alpha=0.7, 
                            label=f'{input_name}=ON', edgecolors='black', linewidth=0.5)
            
            ax_phase.set_xlabel(f"Final {sp_names.get(sp1, sp1)}", fontsize=10)
            ax_phase.set_ylabel(f"Final {sp_names.get(sp2, sp2)}", fontsize=10)
            ax_phase.set_title("Phase Space: Final States", fontsize=11, fontweight='bold')
            ax_phase.legend()
        ax_phase.grid(True, alpha=0.3)
        
        # Main title
        fig2.suptitle(f"Gillespie Simulation Summary - {self.model_id}\n"
                      f"Input: {input_name} | Species: {num_species}",
                      fontsize=13, fontweight='bold')
        
        # Explanation
        explanation = (
            f"This circuit has {num_species} dynamic species and responds to input '{input_name}'.\n"
            f"OFF: {input_name}=0 | ON: {input_name}=HIGH. Phase space shows final state clustering."
        )
        fig2.text(0.5, 0.01, explanation, ha='center', fontsize=9, style='italic',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        
        output_file2 = os.path.join(self.output_dir, f"{self.model_id}_simulation_comparison.png")
        plt.savefig(output_file2, dpi=150)
        print(f"Simulation summary saved to: {os.path.relpath(output_file2)}")
        plt.close()

    def _plot_single_run(self, run: Dict, reacting_species: List[str], sp_names: Dict[str, str]):
        """Generates plots for a single simulation run - shows ALL reacting species."""
        import numpy as np
        
        # Use ALL reacting species
        plot_species = reacting_species if reacting_species else list(run['history'].keys())
        
        # Remove any species that don't change (boundary conditions)
        plot_species = [sp for sp in plot_species 
                       if len(set(run['history'].get(sp, [0]))) > 1]
        
        if not plot_species:
            plot_species = reacting_species[:3] if reacting_species else list(run['history'].keys())[:3]
        
        num_species = len(plot_species)
        
        # Determine grid layout based on number of species
        if num_species <= 2:
            rows, cols = 1, 2
        elif num_species <= 4:
            rows, cols = 2, 2
        elif num_species <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        # Create distinct colors for each species
        species_colors = plt.cm.tab10(np.linspace(0, 1, num_species))
        
        fig = plt.figure(figsize=(5 * cols, 4 * rows))
        
        # --- First plot: All species overlaid ---
        ax_combined = fig.add_subplot(rows, cols, 1)
        
        for i, sp in enumerate(plot_species):
            sp_name = sp_names.get(sp, sp)
            ax_combined.step(run['time'], run['history'][sp], where='post', 
                           color=species_colors[i], alpha=0.8, linewidth=1.5, label=sp_name)
        
        ax_combined.set_xlabel("Time", fontsize=10)
        ax_combined.set_ylabel("Molecule Count", fontsize=10)
        ax_combined.set_title("All Species", fontsize=11, fontweight='bold')
        ax_combined.legend(loc='upper right', fontsize=8)
        ax_combined.grid(True, alpha=0.3)
        ax_combined.set_xlim(left=0)
        ax_combined.set_ylim(bottom=0)
        
        # --- Individual plots for each species ---
        for idx, sp in enumerate(plot_species):
            if idx >= rows * cols - 1:  # Leave room for combined plot
                break
                
            ax = fig.add_subplot(rows, cols, idx + 2)
            sp_name = sp_names.get(sp, sp)
            
            ax.step(run['time'], run['history'][sp], where='post', 
                   color=species_colors[idx], alpha=0.8, linewidth=1.5)
            
            ax.set_xlabel("Time", fontsize=10)
            ax.set_ylabel("Molecule Count", fontsize=10)
            ax.set_title(f"{sp_name}", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            
            # Add final value annotation
            final_val = run['history'][sp][-1]
            ax.axhline(y=final_val, color='red', linestyle='--', alpha=0.3)
            ax.text(0.98, 0.95, f"Final: {final_val:.1f}", transform=ax.transAxes, 
                   fontsize=8, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Main title
        fig.suptitle(f"Gillespie Stochastic Simulation - {self.model_id}\n"
                     f"Single Run, {num_species} Species",
                     fontsize=13, fontweight='bold')
        
        # Explanation
        explanation = (
            f"Single stochastic simulation showing {num_species} reacting species.\n"
            f"For ensemble analysis (variability), increase num_runs parameter."
        )
        fig.text(0.5, 0.01, explanation, ha='center', fontsize=9, style='italic',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        
        output_file = os.path.join(self.output_dir, f"{self.model_id}_simulation.png")
        plt.savefig(output_file, dpi=150)
        print(f"Simulation plot saved to: {os.path.relpath(output_file)}")
        plt.close()

    def _plot_multiple_runs(self, all_runs: List[Dict], reacting_species: List[str], sp_names: Dict[str, str]):
        """Generates plots comparing multiple simulation runs - shows ALL reacting species."""
        import numpy as np
        
        # Use ALL reacting species, not just first 2
        plot_species = reacting_species if reacting_species else list(all_runs[0]['history'].keys())
        
        # Remove any species that don't change (boundary conditions)
        plot_species = [sp for sp in plot_species 
                       if len(set(all_runs[0]['history'].get(sp, [0]))) > 1]
        
        if not plot_species:
            plot_species = reacting_species[:3] if reacting_species else list(all_runs[0]['history'].keys())[:3]
        
        num_species = len(plot_species)
        num_runs = len(all_runs)
        
        # Determine grid layout based on number of species
        if num_species <= 2:
            rows, cols = 2, 2  # 2x2 grid
        elif num_species <= 4:
            rows, cols = 2, 2  # 2x2 grid, one combined + bar chart
        elif num_species <= 6:
            rows, cols = 2, 3  # 2x3 grid
        else:
            rows, cols = 3, 3  # 3x3 grid for many species
        
        # Create distinct colors for each species
        species_colors = plt.cm.tab10(np.linspace(0, 1, num_species))
        run_colors = plt.cm.viridis(np.linspace(0, 0.8, num_runs))
        
        fig = plt.figure(figsize=(5 * cols, 4 * rows))
        
        # --- First plot: All species overlaid (single representative run) ---
        ax_combined = fig.add_subplot(rows, cols, 1)
        
        # Use first run for combined view
        run = all_runs[0]
        for i, sp in enumerate(plot_species):
            sp_name = sp_names.get(sp, sp)
            ax_combined.step(run['time'], run['history'][sp], where='post', 
                           color=species_colors[i], alpha=0.8, linewidth=1.5, label=sp_name)
        
        ax_combined.set_xlabel("Time", fontsize=10)
        ax_combined.set_ylabel("Molecule Count", fontsize=10)
        ax_combined.set_title("All Species (Single Run)", fontsize=11, fontweight='bold')
        ax_combined.legend(loc='upper right', fontsize=8)
        ax_combined.grid(True, alpha=0.3)
        ax_combined.set_xlim(left=0)
        ax_combined.set_ylim(bottom=0)
        
        # --- Individual plots for each species (all runs overlaid) ---
        # Reserve last 2 positions for bar chart and phase space
        max_individual_plots = rows * cols - 3  # Position 1 is combined, last 2 are bar/phase
        for idx, sp in enumerate(plot_species):
            if idx >= max_individual_plots:  # Leave room for combined, bar chart, and phase space
                break
                
            ax = fig.add_subplot(rows, cols, idx + 2)
            sp_name = sp_names.get(sp, sp)
            
            for i, run in enumerate(all_runs):
                ax.step(run['time'], run['history'][sp], where='post', 
                       color=run_colors[i], alpha=0.5, linewidth=1)
            
            ax.set_xlabel("Time", fontsize=10)
            ax.set_ylabel("Molecule Count", fontsize=10)
            ax.set_title(f"{sp_name} ({num_runs} runs)", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
        
        # --- Bar chart of final values ---
        ax_bar = fig.add_subplot(rows, cols, rows * cols - 1)
        
        final_values = {sp: [] for sp in plot_species}
        for run in all_runs:
            for sp in plot_species:
                final_values[sp].append(run['history'][sp][-1])
        
        x_pos = np.arange(len(plot_species))
        means = [np.mean(final_values[sp]) for sp in plot_species]
        stds = [np.std(final_values[sp]) for sp in plot_species]
        labels = [sp_names.get(sp, sp) for sp in plot_species]
        
        bars = ax_bar.bar(x_pos, means, yerr=stds, capsize=5, 
                         color=[species_colors[i] for i in range(len(plot_species))], alpha=0.7)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax_bar.set_ylabel("Final Molecule Count", fontsize=10)
        ax_bar.set_title(f"Final Values (n={num_runs})", fontsize=11, fontweight='bold')
        ax_bar.grid(True, alpha=0.3, axis='y')
        
        # --- Phase space or time series comparison ---
        ax_phase = fig.add_subplot(rows, cols, rows * cols)
        
        if num_species >= 2:
            # For oscillators: plot time series overlay of first run (different lengths prevent averaging)
            run = all_runs[0]
            time_points = run['time']
            
            for i, sp in enumerate(plot_species[:min(6, num_species)]):  # Max 6 species
                sp_name = sp_names.get(sp, sp)
                ax_phase.step(time_points, run['history'][sp], where='post',
                             color=species_colors[i], linewidth=1.5, label=sp_name, alpha=0.8)
            
            ax_phase.set_xlabel("Time", fontsize=10)
            ax_phase.set_ylabel("Molecule Count", fontsize=10)
            ax_phase.set_title("Representative Run (All Species)", fontsize=11, fontweight='bold')
            ax_phase.legend(loc='upper right', fontsize=8)
            ax_phase.grid(True, alpha=0.3)
        
        # Main title
        fig.suptitle(f"Gillespie Stochastic Simulation - {self.model_id}\n"
                     f"Ensemble Analysis: {num_runs} Runs, {num_species} Species",
                     fontsize=13, fontweight='bold')
        
        # Explanation
        explanation = (
            f"Showing all {num_species} reacting species. Each colored line in individual plots represents one simulation run.\n"
            f"Bar chart shows mean ± std of final values. Mean trajectories show average behavior across runs."
        )
        fig.text(0.5, 0.01, explanation, ha='center', fontsize=9, style='italic',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        
        output_file = os.path.join(self.output_dir, f"{self.model_id}_simulation_comparison.png")
        plt.savefig(output_file, dpi=150)
        print(f"Simulation comparison plot saved to: {os.path.relpath(output_file)}")
        plt.close()


class ODESimulator:
    """
    Performs deterministic ODE simulations based on parsed SBML data.
    Uses scipy's odeint for numerical integration.
    Works well for models with continuous concentrations (e.g., circadian clock).
    """

    def __init__(self, json_filepath: str):
        self.json_filepath = json_filepath
        self.output_dir = os.path.dirname(os.path.abspath(json_filepath))
        self.data = self._load_json()
        self.model_id = self.data.get("model_id", "model")
        
        # Model components
        self.species_state = {}
        self.parameters = {}
        self.reactions = []
        self.species = []
        self.species_ids = []
        
        self._initialize_model()

    def _load_json(self) -> Dict[str, Any]:
        with open(self.json_filepath, 'r') as f:
            return json.load(f)

    def _initialize_model(self):
        """Initializes state and parameters."""
        
        # 1. Load Compartments
        for comp in self.data.get("compartments", []):
            self.parameters[comp['id']] = float(comp['size'])
        
        # 2. Load Species
        self.species = self.data.get("species", [])
        for sp in self.species:
            self.species_state[sp['id']] = float(sp['initial_amount'])
            self.species_ids.append(sp['id'])

        # 3. Load Parameters
        for param in self.data.get("parameters", []):
            self.parameters[param['id']] = float(param['value'])
        
        # 4. Load Function Definitions
        self.function_defs = self.data.get("function_definitions", {})
        
        # 5. Load Assignment Rules
        self.assignment_rules = self.data.get("assignment_rules", [])

        # 6. Load Reactions
        for rxn in self.data.get("reactions", []):
            local_params = {p['id']: float(p['value']) for p in rxn.get("local_parameters", [])}
            formula = rxn.get('kinetic_law', '')
            if formula:
                self.reactions.append({
                    "id": rxn['id'],
                    "formula": formula,
                    "products": rxn.get("products", []),
                    "reactants": rxn.get("reactants", []),
                    "local_params": local_params
                })

    def _create_function(self, arg_names: List[str], formula: str, base_context: Dict) -> callable:
        """Creates a callable function from SBML function definition."""
        def func(*args):
            if formula == 'NaN' or formula is None:
                return 0.0
            local_ctx = base_context.copy()
            for name, value in zip(arg_names, args):
                local_ctx[name] = value
            try:
                result = eval(formula, {}, local_ctx)
                if np.isnan(result) or np.isinf(result):
                    return 0.0
                return float(result)
            except Exception:
                return 0.0
        return func

    def _evaluate_assignment_rules(self, context: Dict[str, float]) -> Dict[str, float]:
        """Evaluates SBML assignment rules."""
        for rule in self.assignment_rules:
            variable = rule['variable']
            formula = rule['formula']
            try:
                value = eval(formula, {}, context)
                context[variable] = float(value)
            except Exception:
                pass
        return context

    def _derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Computes the derivatives (dX/dt) for all species.
        This is the core of ODE simulation.
        """
        # Build context with current state
        context = {**self.parameters}
        for i, sp_id in enumerate(self.species_ids):
            context[sp_id] = state[i]
        
        # Add numpy functions
        context.update({k: getattr(np, k) for k in dir(np) if not k.startswith('_')})
        
        # Evaluate assignment rules
        context = self._evaluate_assignment_rules(context)
        
        # Add user-defined functions
        for func_id, func_info in self.function_defs.items():
            args = func_info.get('arguments', [])
            formula = func_info.get('formula', '0')
            if formula != 'NaN' and formula is not None:
                context[func_id] = self._create_function(args, formula, context)
            else:
                context[func_id] = lambda *args: 0.0
        
        # Initialize derivatives to zero
        dxdt = np.zeros(len(self.species_ids))
        
        # For each reaction, compute rate and update derivatives
        for rxn in self.reactions:
            context.update(rxn['local_params'])
            
            try:
                rate = eval(rxn['formula'], {}, context)
                if np.isnan(rate) or np.isinf(rate):
                    rate = 0.0
                rate = float(rate)
            except Exception:
                rate = 0.0
            
            # Subtract rate from reactants
            for r in rxn['reactants']:
                sp_idx = self.species_ids.index(r['species'])
                stoich = r.get('stoichiometry', 1.0)
                dxdt[sp_idx] -= rate * stoich
            
            # Add rate to products
            for p in rxn['products']:
                sp_idx = self.species_ids.index(p['species'])
                stoich = p.get('stoichiometry', 1.0)
                dxdt[sp_idx] += rate * stoich
        
        return dxdt

    def run_simulation(self, duration: float, num_points: int = 1000) -> Dict[str, Any]:
        """
        Runs ODE simulation using scipy's odeint.
        
        Args:
            duration: Total simulation time
            num_points: Number of time points to sample
            
        Returns:
            Dictionary with 'time' and 'history' (species trajectories)
        """
        from scipy.integrate import odeint
        
        # Initial state vector
        y0 = np.array([self.species_state[sp_id] for sp_id in self.species_ids])
        
        # Time points
        t = np.linspace(0, duration, num_points)
        
        # Solve ODE
        print(f"    Running ODE integration (duration={duration}, points={num_points})...")
        solution = odeint(self._derivatives, y0, t)
        
        # Convert to history dict
        history = {}
        for i, sp_id in enumerate(self.species_ids):
            history[sp_id] = solution[:, i].tolist()
        
        return {"time": t.tolist(), "history": history}

    def run_and_plot(self, duration: float = 100.0, num_points: int = 1000):
        """
        Runs ODE simulation and generates plots showing all species dynamics.
        """
        result = self.run_simulation(duration, num_points)
        
        # Get species names
        sp_names = {sp['id']: sp.get('name', sp['id']) for sp in self.species}
        
        # Find species that actually change
        dynamic_species = []
        for sp_id in self.species_ids:
            values = result['history'][sp_id]
            if max(values) - min(values) > 1e-10:  # Has variation
                dynamic_species.append(sp_id)
        
        if not dynamic_species:
            dynamic_species = self.species_ids[:6]  # Show first 6 if nothing changes
        
        num_species = len(dynamic_species)
        
        # Determine subplot layout
        if num_species <= 4:
            rows, cols = 2, 2
        elif num_species <= 6:
            rows, cols = 2, 3
        elif num_species <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 4, 4
        
        # Create color palette
        colors = plt.cm.tab10(np.linspace(0, 1, min(num_species, 10)))
        
        fig = plt.figure(figsize=(5 * cols, 4 * rows))
        
        # Plot 1: All species combined
        ax_combined = fig.add_subplot(rows, cols, 1)
        for i, sp_id in enumerate(dynamic_species[:10]):  # Max 10 in combined
            sp_name = sp_names.get(sp_id, sp_id)
            ax_combined.plot(result['time'], result['history'][sp_id], 
                           color=colors[i % len(colors)], linewidth=1.5, 
                           label=sp_name[:20], alpha=0.8)
        ax_combined.set_xlabel("Time", fontsize=10)
        ax_combined.set_ylabel("Concentration", fontsize=10)
        ax_combined.set_title("All Dynamic Species", fontsize=11, fontweight='bold')
        ax_combined.legend(loc='upper right', fontsize=7, ncol=2)
        ax_combined.grid(True, alpha=0.3)
        
        # Individual species plots
        for idx, sp_id in enumerate(dynamic_species[:rows*cols-1]):
            ax = fig.add_subplot(rows, cols, idx + 2)
            sp_name = sp_names.get(sp_id, sp_id)
            
            ax.plot(result['time'], result['history'][sp_id], 
                   color=colors[idx % len(colors)], linewidth=2)
            ax.set_xlabel("Time", fontsize=10)
            ax.set_ylabel("Concentration", fontsize=10)
            ax.set_title(f"{sp_name[:25]}", fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add min/max annotations
            values = result['history'][sp_id]
            ax.axhline(y=max(values), color='red', linestyle='--', alpha=0.3)
            ax.axhline(y=min(values), color='blue', linestyle='--', alpha=0.3)
        
        # Main title
        fig.suptitle(f"ODE Deterministic Simulation - {self.model_id}\n"
                     f"Duration: {duration}, Species: {len(self.species_ids)} total, {num_species} dynamic",
                     fontsize=13, fontweight='bold')
        
        # Explanation
        explanation = (
            f"ODE simulation shows deterministic (average) behavior.\n"
            f"Unlike Gillespie, results are identical every run. Best for continuous concentrations."
        )
        fig.text(0.5, 0.01, explanation, ha='center', fontsize=9, style='italic',
                 bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        
        output_file = os.path.join(self.output_dir, f"{self.model_id}_ode_simulation.png")
        plt.savefig(output_file, dpi=150)
        print(f"ODE simulation plot saved to: {os.path.relpath(output_file)}")
        plt.close()
        
        return result


def choose_simulator(json_filepath: str) -> str:
    """
    Automatically chooses the best simulation method based on model characteristics.
    Returns 'gillespie' or 'ode'.
    
    Rationale (based on Gillespie 1977, Higham 2008):
    - Gillespie SSA: Best for small molecule counts where stochastic noise is significant
    - ODE: Best for large molecule counts (thermodynamic limit) or continuous concentrations
    
    Heuristics:
    1. Very small fractional values (< 1) suggest concentrations → ODE
    2. Small integer-like values (1-1000) suggest discrete counts → Gillespie  
    3. Very large values (> 10000) approach thermodynamic limit → ODE (faster & equivalent)
    4. High reaction rates make Gillespie computationally expensive → ODE
    """
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    
    species = data.get("species", [])
    if not species:
        return 'ode', "no species found"
    
    # Get initial amounts
    amounts = [sp.get('initial_amount', 0) for sp in species]
    max_amount = max(amounts)
    avg_amount = sum(amounts) / len(amounts)
    
    # Check if values look like concentrations (very small, fractional)
    # Concentrations are typically << 1 (e.g., 0.001 mM, 1e-6 M)
    non_zero_amounts = [a for a in amounts if a > 0]
    if non_zero_amounts:
        min_nonzero = min(non_zero_amounts)
        # If smallest non-zero value is very small (< 0.1), likely concentrations
        if min_nonzero < 0.1 and max_amount < 100:
            return 'ode', "small fractional values suggest concentrations"
    
    # Check if values are very large (thermodynamic limit)
    # With > 10000 molecules, stochastic effects are negligible (law of large numbers)
    # ODE is also much faster for these cases
    if max_amount > 10000 or avg_amount > 5000:
        return 'ode', "very large values suggest thermodynamic limit"
    
    # Intermediate range (1-10000): Gillespie is appropriate
    # This is where stochastic effects matter biologically
    # (gene expression noise, small regulatory circuits, etc.)
    if max_amount >= 1 and max_amount <= 10000:
        return 'gillespie', "intermediate values suggest discrete counts"
    
    # Default fallback: ODE (safer, always works)
    return 'ode', "defaulting to ODE simulation"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simulate_circuit.py <path_to_parsed_json> [method] [duration] [num_runs]")
        print("       method: 'gillespie', 'ode', or 'auto' (default)")
        print("       duration: simulation duration (default: 500)")
        print("       num_runs: number of runs for Gillespie (default: 1)")
        sys.exit(1)

    json_file = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'auto'
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0
    num_runs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    
    # Some methods take a really long time if the initial amounts are large
    if method == 'auto':
        method, reason = choose_simulator(json_file)
        print(f"Auto-selected simulation method: {method.upper()} - {reason}")
    
    if method == 'ode':
        simulator = ODESimulator(json_file)
        simulator.run_and_plot(duration=duration)
    else:
        simulator = GillespieSimulator(json_file)
        simulator.run_comparative_analysis(duration=duration, num_runs=num_runs)

