import requests
import re
import sys
import os
from typing import Optional, Tuple

class BioModelAgent:
    """
    An agent responsible for searching, retrieving, and verifying biological models 
    from the BioModels database.
    """

    def __init__(self) -> None:
        """
        Initialize the BioModelAgent with API endpoints.
        """
        self.search_url: str = "https://www.ebi.ac.uk/biomodels/search"
        self.download_base_url: str = "https://www.ebi.ac.uk/biomodels/model/download"
        # Define output directory relative to this script (assuming script is in src/)
        self.output_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

    def resolve_input(self, user_input: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Rationalizes the input. If it's a BIOMD ID, returns it.
        If it's a name, uses 'intelligent' search to find the ID.

        Args:
            user_input (str): The input string provided by the user. 
                              Can be a common name (e.g., "Repressilator") or a BIOMD ID.

        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple containing (BIOMD ID, Common Name).
        """
        # Check if it matches BIOMD format (case insensitive)
        if re.match(r'^BIOMD\d+$', user_input, re.IGNORECASE):
            print(f"Input identified as BIOMD ID: {user_input.upper()}")
            # If input is ID, we need to fetch the name to create the folder
            name = self.get_name_from_id(user_input.upper())
            return user_input.upper(), name
        
        print(f"Input identified as common name: '{user_input}'. Initiating search agent...")
        biomd_id, name = self.search_for_id(user_input)
        return biomd_id, name

    def get_name_from_id(self, biomd_id: str) -> str:
        """
        Fetches the model name given a BIOMD ID.
        """
        # We can use the search API with the ID to get the name
        params = {
            "query": biomd_id,
            "format": "json"
        }
        try:
            response = requests.get(self.search_url, params=params)
            response.raise_for_status()
            data = response.json()
            if "models" in data and len(data["models"]) > 0:
                return data["models"][0]["name"]
        except Exception as e:
            print(f"Error fetching name for ID {biomd_id}: {e}")
        return "Unknown_Model"

    def search_for_id(self, name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Searches for the circuit in biological databases using the BioModels API.

        Args:
            name (str): The common name of the biological circuit to search for.

        Returns:
            Tuple[Optional[str], Optional[str]]: (BIOMD ID, Model Name)
        """
        params = {
            "query": name,
            "format": "json"
        }
        try:
            response = requests.get(self.search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "models" in data and len(data["models"]) > 0:
                # Intelligent selection: Pick the most relevant one
                best_match = data["models"][0]
                print(f"Agent found potential match: {best_match['name']} ({best_match['id']})")
                return best_match['id'], best_match['name']
            else:
                print("Agent could not find any models matching that name.")
                return None, None
        except Exception as e:
            print(f"Error during search: {e}")
            return None, None

    def sanitize_filename(self, name: str) -> str:
        """Sanitizes a string to be safe for filenames."""
        return re.sub(r'[^\w\-_\. ]', '_', name).strip()

    def download_model(self, biomd_id: str, common_name: str) -> Optional[str]:
        """
        Downloads the SBML file for the given BIOMD ID into a specific folder.

        Args:
            biomd_id (str): The unique identifier for the BioModel.
            common_name (str): The common name of the model.

        Returns:
            Optional[str]: The absolute path to the downloaded file, or None if failed.
        """
        if not biomd_id:
            return None
            
        # Create output directory: output/{CommonName}-{BIOMD}
        safe_name = self.sanitize_filename(common_name)
        folder_name = f"{safe_name}-{biomd_id}"
        output_dir = os.path.join(self.output_root, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Agent initiating download for {biomd_id} into {os.path.relpath(output_dir)}...")
        
        url = f"{self.download_base_url}/{biomd_id}?filename={biomd_id}_url.xml"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            filename = os.path.join(output_dir, f"{biomd_id}.xml")
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded model to {os.path.relpath(filename)}")
            
            # Verify content as requested
            self.verify_sbml_content(filename)
            return filename
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None

    def verify_sbml_content(self, filename: str) -> None:
        """
        Verifies that the downloaded SBML file contains valid reaction rates and agents (species).
        Uses libsbml to parse the file structure.

        Args:
            filename (str): The file path to the local SBML (.xml) file.
        """
        try:
            import libsbml
            reader = libsbml.SBMLReader()
            document = reader.readSBML(filename)
            model = document.getModel()
            
            if model:
                species_count = model.getNumSpecies()
                reaction_count = model.getNumReactions()
                parameter_count = model.getNumParameters()
                
                print(f"\n--- Model Verification ---")
                print(f"Model ID: {model.getId()}")
                print(f"Agents (Species): {species_count}")
                print(f"Reactions: {reaction_count}")
                print(f"Parameters (Rates/Constants): {parameter_count}")
                
                if species_count > 0 and reaction_count > 0:
                    print("Status: VALID. Contains agents and reactions.")
                else:
                    print("Status: INCOMPLETE. Missing species or reactions.")
            else:
                print("Warning: Could not parse SBML model structure.")
        except ImportError:
            print("libsbml not installed, skipping content verification.")
        except Exception as e:
            print(f"Verification warning: {e}")

if __name__ == "__main__":
    agent = BioModelAgent()
    
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        print("Biological Circuit Retrieval Agent")
        user_input = input("Enter biological circuit name or BIOMD ID: ")
        
    biomd_id, common_name = agent.resolve_input(user_input)
    if biomd_id and common_name:
        agent.download_model(biomd_id, common_name)
