from logging import config
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from google.generativeai import types
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
from google import genai
from google.genai import types
#from googlesearch import search
import time
import pathlib
import pandas as pd
import psi4
import psi4
from psi4.driver.procrouting.response.scf_response import tdscf_excitations
import requests

# DEFINING THE FUNCTIONS

def load_mol(smiles: str, smarts: bool = False):
    mol = Chem.MolFromSmarts(smiles) if smarts else Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid {'SMARTS' if smarts else 'SMILES'} string: {smiles}")
    return mol

def validate_mol(smiles: str) -> bool:
  # instructions for the lLM to use
  """
  Check for if the newly generated molecular structure is valid.

  This function is designed for a molecular designer to ensure a valid molecular structure is being created.
  It takes in a SMILE string of a newly designed molecular structure made and checks for if it is a valid strucutre.

  Args:
    smiles: the newly created molecule's SMILE string

   Returns:
    If the molecule is valid or not, if not valid, resends original structure to edit to become valid.
  """
  try:
      mol = Chem.MolFromSmiles(smiles)

      if mol is None:
        raise ValueError("Molecule is invalid")

      mol = Chem.SanitizeMol(mol)

      print("Molecule is valid")
      return True

  except Exception as e:
      return False

def predict_lambda(smiles: str):
  """
  Predicts λ_max for a given SMILES string.

  It takes in a SMILE string of a newly designed molecular structure made and predicts the value of λ_max.

  Args:
    smiles: the newly created molecule's SMILE string

   Returns:
    The predicted value of λ_max.
  """
  url = "https://net-portion-wage-peoples.trycloudflare.com/predict_lambda"
  response = requests.post(url, json={
            "smiles": smiles
            })
  print(response.json())
  return response.json()['predicted_nm']

def predict_dft(smiles: str):
    """
        Predicts time-dependent density functional theory of a molecule.

        It takes in a SMILE String and converts it to its molecular coordinates and geometry.

        Then it uses the PSI4 Package to run DFT Calculations on the molecule

        Args:
            smiles: newly created SMILES string

        Returns: DFT-Calculations
    """

    print("running")

    # ISSUE: CONVERTING PROVIDED SMILE STRING TO PSI4 GEOMETRY

    # 2. Generate a 2D molecule object from the SMILES
    mol_rdkit = Chem.MolFromSmiles(smiles)

    # 3. Add explicit hydrogens and generate 3D coordinates using the MMFF94 force field
    mol_rdkit = Chem.AddHs(mol_rdkit)
    AllChem.EmbedMolecule(mol_rdkit, AllChem.ETKDGv3()) # Use a more modern conformer generation method
    AllChem.MMFFOptimizeMolecule(mol_rdkit) # Optimize with a force field

    # 4. Convert the RDKit molecule to an XYZ string format
    mol_xyz = Chem.rdmolfiles.MolToXYZBlock(mol_rdkit)

    # 5. Get rid of the first two lines of the XYZ string (which contain atom count and a comment line)
    stripped_xyz = mol_xyz.split('\n', 2)[2:][0]


    psi4_geometry_string = f"""
    {stripped_xyz}
    """

    mol_psi4 = psi4.geometry(psi4_geometry_string)

    psi4.core.set_output_file("molecule_out")
    psi4.core.set_num_threads(4)
    # Set method and basis set
    psi4.set_options({
        "save_jk": True,
        "basis": "6-31g*",
        "tdscf_states": 5  # Number of excited states to compute
    })

    # Run TD-DFT (Time-Dependent DFT)
    e, wfn = psi4.energy("HF/cc-pVDZ", return_wfn=True, molecule=mol_psi4)
    res = tdscf_excitations(wfn, states=10)

    first_key, first_value = next(iter(res[0].items()))
    print(f"{first_key} = {first_value}")
    return first_value

def predict_electronproperties(smiles: str):
    """
    Predict electron acception and electron donation properties. 

    Args:
    smiles: the newly created smiles string

   Returns:
       Returns the electron acception and donation values (homo energy, lumo energy, gap homo-lumo, ionization potential, electron affinity) 
    """

       # 2. Generate a 2D molecule object from the SMILES
    mol_rdkit = Chem.MolFromSmiles(smiles)

    # 3. Add explicit hydrogens and generate 3D coordinates using the MMFF94 force field
    mol_rdkit = Chem.AddHs(mol_rdkit)
    AllChem.EmbedMolecule(mol_rdkit, AllChem.ETKDGv3()) # Use a more modern conformer generation method
    AllChem.MMFFOptimizeMolecule(mol_rdkit) # Optimize with a force field

    # 4. Convert the RDKit molecule to an XYZ string format
    mol_xyz = Chem.rdmolfiles.MolToXYZBlock(mol_rdkit)

    # 5. Get rid of the first two lines of the XYZ string (which contain atom count and a comment line)
    stripped_xyz = mol_xyz.split('\n', 2)[2:][0]


    psi4_geometry_string = f"""
    {stripped_xyz}
    """

    mol_psi4 = psi4.geometry(psi4_geometry_string)

    # Set calculation options
    psi4.set_options({
        'basis': '6-31G*',
        'scf_type': 'pk',
        'reference': 'rhf'
    })

    # Run DFT calculation (B3LYP functional)
    energy, wfn = psi4.energy('B3LYP', return_wfn=True, molecule=mol_psi4)

    # Extract HOMO, LUMO, and gap
    eps_a = wfn.epsilon_a().to_array()
    nocc = wfn.nalpha()
    homo = eps_a[nocc - 1]
    lumo = eps_a[nocc]
    gap = lumo - homo

    psi4.set_options({'basis': '6-31G*', 'reference': 'rhf'})

    # Neutral molecule
    E_neutral = psi4.energy('B3LYP/6-31G*')

    # Cation (+1 charge) — remove 1 electron
    mol_psi4.charge = 1
    E_cation = psi4.energy('B3LYP/6-31G*')

    # Anion (-1 charge) — add 1 electron
    mol_psi4.charge = -1
    E_anion = psi4.energy('B3LYP/6-31G*')

    # Ionization potential (IP) and electron affinity (EA)
    IP = (E_cation - E_neutral) * 27.2114   # eV
    EA = (E_neutral - E_anion) * 27.2114    # eV

    print("DATA ON ELECTRON ACCEPTION AND DONATION")
    print(f"HOMO: {homo:.3f} Hartree = {homo * 27.2114:.3f} eV")
    print(f"LUMO: {lumo:.3f} Hartree = {lumo * 27.2114:.3f} eV")
    print(f"Gap:  {gap:.3f} Hartree = {gap * 27.2114:.3f} eV")
    print(f"Ionization Potential (IP): {IP:.3f} eV")
    print(f"Electron Affinity (EA): {EA:.3f} eV")
    

# DEFINING FUNCTIONS AS TOOLS

def make_tool(name, description, params):
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=name,
                description=description,
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=params,
                    required=list(params.keys())
                )
            )
        ]
    )

#'Checks if molecule is a valid structure and just needs standardization.',
validate_mol_tool = make_tool(
    "validate_mol",
    "Check whether a molecule SMILES string is chemically valid. "
    "Always use this before predicting lambda.",
    {
        "smiles": types.Schema(type=types.Type.STRING, description="Molecule SMILES string")
    }
)

predict_lambda_tool = make_tool(
    "predict_lambda",
    "Predicts the absorption wavelength (lambda_max, nm) for a valid molecule using Chemprop. "
    "Only call this after validation passes.",
    {
        "smiles": types.Schema(type=types.Type.STRING, description="Valid molecule SMILES string")
    }

)

predict_dft_tool = make_tool(
    "predict_dft",
    "Predicts the light absorption properpties of a SMILE string molecule through DFT Calculations and PSI4 package. "
    "Only call this after validation of molecule is passed.",
    {
        "smiles": types.Schema(type=types.Type.STRING, description="Valid molecule SMILES string")
    }
)

predict_electronproperties_tool = make_tool(
    "predict_electronproperties",
    "Predicts the electron acception and donation properties of the SMILE string molecule through PSI5 package"
    "Only call this after validation of molecule and light absoprtion properties predicted.",
    {
        "smiles": types.Schema(type=types.Type.STRING, description="Valid molecule SMILES string")
    }

)

# DEFINING LLM FEEDBACK LOOP MECHANISMS

LAMBDA_THRESHOLD = 350
TARGET_MOLECULE_COUNT = 5
MAX_ITERATIONS = 50

class MoleculeDesignStateMachine:

    """Manages the state and the flow of the molecule redesign process."""

    def __init__(self, threshold: float, target_count: int):
        self.threshold = threshold
        self.target_count = target_count
        self.valid_molecules = {} # {smiles: lambda_max}
        self.rejected_molecules = {} # {smiles: lambda_max}
        self.current_smiles = None
        self.iteration = 0
    
    def get_progress_message(self) -> str:
        "Progress message for the model"

        return (
            f"\n{'='*60}\n"
            f"PROGRESS: {len(self.valid_molecules)}/{self.target_count} molecules found\n"
            f"Iteration: {self.iteration}\n"
            f"{'='*60}\n"
        )
    
    def get_feedback_message(self, function_name: str, smiles: str, result) -> str:
        """Generate feedback message based on function"""
        if function_name == "validate_mol":
            if result:
                return (
                    f" Validation Passed for: {smiles}\n"
                    f"Next step: Call predict_lambda with this exact SMILES string."
                )
            else:
                return (
                    f" Validation Failed for: {smiles}\n"
                    f"The molecule is chemically invalid. Generate new\n"
                    f"Progress: {len(self.valid_molecules)}/{self.target_count}"
                )    
            
        elif function_name == "predict_lambda":
            lambda_max = result
            if lambda_max >= self.threshold:
                self.valid_molecules[smiles] = lambda_max
                msg = (
                    f" Accepted {smiles}"
                    f" Lambda max {lambda_max:.1f} nm (>= {self.threshold} nm threshold)\n"
                    f" Progress: {len(self.valid_molecules)}/{self.target_count} molecule\n"
                )
                if len(self.valid_molecules) >= self.target_count:
                    msg += f"\n TARGET REACHED! You have fonud {self.target_count} valid molecules.\n" 
                    msg += "Now provide your final summary in the request format"
                else:
                    msg += f"\nContinue: Generate molecule #{len(self.valid_molecules) + 1}"
                return msg
            else:
                self.rejected_molecules[smiles] = lambda_max
                return (
                    f" REJECTED: {smiles}\n"
                    f"Lambda Max: {lambda_max:.1f} nm (< {self.threshold} nm threshold)\n"
                    f"Progress: {len(self.valid_molecules)}/{self.target_count} molecules\n"
                    f" Generate a NEW molecule with different modifications that fit the targets"
                )
        return ""

    def is_complete(self) -> bool:
        """check if target has reached"""
        return len(self.valid_molecules) >= self.target_count
    
    def print_summary(self):
        """Print summary of results final"""
        print("\n" + "="*60)
        print("Final Results")
        print("="*60)
        print(f"\n Accepted Molecules ({len(self.valid_molecules)}):")
        for i, (smiles, lambda_max) in enumerate(self.valid_molecules.items(), 1):
            print(f" {i}. {smiles}")
            print(f" lambda_max = {lambda_max:.1f} nm")

        if self.rejected_molecules:
            print(f"Rejected molecules ({len(self.rejected.molecules)})")
            for smiles, lambda_max in self.rejected_molecules.items():
                print(f" - {smiles} (lambda max = {lambda_max:.1f} nm)")

def run_molecule_redesign(apikey: str, pdf_path: str):
    """Main function to run molecule redesign workflow"""

    client = genai.Client(api_key=apikey)
    print("Files uploading")
    pdf_file = client.files.upload(file=pdf_path)

    modelrole = f"""
    You are computational molecular designer specialized in photochemistry, structure-property relationships, and solar material design.

        GOAL: Your goal is to enhance provided molecular structures with structures that enhance photoswitching and light absorption properties in organic molecules, additionally
        add groups to the molecule that allow for the release of an electron due to structure changes under light and the ability to be given a new electron when returning to the original state pre-light absorption.

        RESOURCES:
        You have access to these files:
        - AzobenzeneLightRversibilityPaper.pdf: A research paper on azobenzene photoswitch molecular structures
        - Use your personal search tools.

        TOOLS:
        - The `validate_mol` tool, which checks if a proposed SMILES structure is valid -> returns True/False.
        - The `predict_lambda` tool, which predicts the λ_max of the proposed SMILES structure -> returns predicted λmax (nm).
        - The 'predict_dft' tool, which predicts various density-functional theories -> returns the excitation energy value (float) and prints data related to DFT properties
        - The 'predict_electronproperties' tool, which predicts various electron acception and electron donation properties -> returns nothing but prints out electron property data


        Here is the STRICT WORKFLOW you MUST follow in this specific order and flow: DO NOT RETURN TO STEP 1 UNLESS SPECIFIED
        1) Modify the molecule with a structure that you have researched will theoretically improve light absorption and photo switching properties
        2) Explain the reason why and what sources were used (the research paper provided or others)
        3) IMMEDIATELY call the validate_mol tool using the modified molecule as the input a
        4) WAIT for validation result
        4) IF VALID MOLECULE -> IMMEDIATELY CALL predict_lambda tool to provide a prediction for the light absorption
        5) ONLY IF INVALID MOLECULE -> Return to STEP 1 to ideate a new structure modification
        6) WAIT for lambda_max prediction result
        7) IF lambda_max >= 375nm -> use predict_dft to predict density-functional theory properties
        8) ONLY IF lambda_max < 375nm -> RETURN to STEP 1 to alter the molecule some more to improve lambda_max value
        7) RIGHT AFTER predict_dft is called -> call predict_electronproperties tool to predict properties related to electron transfers
        8) IF predict_electronproperties has been called: All properties have been predicted for one modified molecule, move onto next and new structure (return to Step 1) and if 
        {TARGET_MOLECULE_COUNT} has NOT been reached
        8) Stop once {TARGET_MOLECULE_COUNT} valid structures are proposed

        CRITICAL RULES:
        - Generate ONE molecule at a tinme
        - First action is ALWAYS validate_mol
        - ALWAYS validate molecule NEVER skip validate_mol
        - NEVER generate multiple molecules in one response
        - DONT Move onto designing next molecule UNTIL predict_lambda is completed for the previous molecule
        - COMPLETE ALL STEPS for EACH molecule: Use ALL tools in order: validate_mol -> predict_lambda
        -> predict_dft -> predict_electronproperties
        - Stop once {TARGET_MOLECULE_COUNT} valid structures are proposed

        After {TARGET_MOLECULE_COUNT} valid molecules have been confirmed and EACH STEP HAS BEEN COMPLETED PER MOLECULE, summarize the results as such:

        SUMMARIZED RESULTS:
    1. SMILES Molecule 1:
        a. Predicted λ: [value given from predict_lambda]
        b. Key structural modification:
        c. Reasoning behind modification:
        d. DFT-data printed from [predict_dft]
        e. Electron property data printed from [predict_electronproperties]

    2. SMILES Molecule 2:
        a. Predicted λ: [value given from predict_lambda]
        b. Key structural modification:
        c. Reasoning behind modification:
        d. DFT-data printed from [predict_dft]
        e. Electron property data printed from [predict_electronproperties]

    3. SMILES Molecule 3:
        a. Predicted λ: [value given from predict_lambda]
        b. Key structural modification:
        c. Reasoning behind modification:
        d. DFT-data printed from [predict_dft]
        e. Electron property data printed from [predict_electronproperties]
    """


    mainprompt = f"""
    Analyze the structure of the following molecule: Retinal: CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C\C(=C\C=O)\C)/C

    Using your knowledge, search tools, and research paper file provided "AzobenzeneLightReversibilityPaper.pdf" propose 
    {TARGET_MOLECULE_COUNT} valid, enhanced structures of Retinal that improve, optimize, and enhance its photoswitching and light absorption (increased absorption range) properties.
    Additionally, add groups to the molecule (retinal) that allow for the release of an electron due to structure changes under light and the ability to be given a new electron when returning to the original state pre-light absorption.
    
    For each molecule design:
     - Provide its SMILE string
     - Describe the key structural change(s) and why it improves the photoswitching and light absorption properties
     - Ensure electron donation abilities to potent under light absorption and when returning to structure pre-light absorption 
     - Stop once {TARGET_MOLECULE_COUNT} valid, enhanced structures are designed


    """

    # dictionary of tools available to LLM
    available_tools = {
        "validate_mol": validate_mol
    }


    contents_list = [mainprompt, "Here is the research paper on azobenzene photoswitches for analysis:", 
                 pdf_file]
    # allows chatbot to understand history and remember things of the past in the conversation
    history = contents_list.copy()
    state_machine = MoleculeDesignStateMachine(LAMBDA_THRESHOLD, TARGET_MOLECULE_COUNT)

    print(state_machine.get_progress_message())

    while state_machine.iteration < MAX_ITERATIONS:
        state_machine.iteration += 1

        try:
            print("Starting LLM") 
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents_list,
                config=types.GenerateContentConfig(
                tools=[validate_mol_tool, predict_lambda_tool, predict_dft_tool, predict_electronproperties_tool],
                #tools=[validate_mol_tool, predict_lambda_tool, predict_dft_tool, predict_electronproperties_tool],
                tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode='ANY')),
                system_instruction=modelrole,
                temperature=0.7
                )
            )

            # checking for function call (standard structure from googles api doc on gemini)
            if (response.candidates and
                response.candidates[0].content.parts and
                response.candidates[0].content.parts[0].function_call):

                function_call = response.candidates[0].content.parts[0].function_call
                function_name = function_call.name
                function_args = function_call.args
                smiles = function_args.get("smiles", "")

                print(f"\n[Iteration {state_machine.iteration}] Function: {function_name}")
                print(f"SMILES: {smiles}")

                # execute the function
                if function_name == "validate_mol":
                    result = validate_mol(smiles)
                    state_machine.current_smiles = smiles if result else None

                elif function_name == "predict_lambda":
                    result = predict_lambda(smiles)

                elif function_name == "predict_dft":
                    result = predict_dft(smiles)

                elif function_name == "predict_electronproperties":
                    result = predict_electronproperties(smiles)

                else:
                    print(f"Unknown function: {function_name}")
                    continue

                # add function call to history
                history.append(types.Content(
                    role="model",
                    parts=[types.Part(function_call=function_call)]
                ))

                # add function result to history
                history.append(types.Content(
                    role="user",
                    parts=[types.Part(function_response=types.FunctionResponse(
                        name=function_name,
                        response={"result": result}
                    ))]
                ))

             # add feedback message
                feedback = state_machine.get_feedback_message(function_name, smiles, result)
                print(feedback)

                history.append(types.Content(
                    role="user",
                    parts=[types.Part(text=feedback)]
                ))

            # check if complete
                if state_machine.is_complete():
                    print("\n Target reached! Requesting final summary...")
                    break

            # text instead of function call
            else:
                text = response.text if hasattr(response, 'text') else ""
                print(f"\n[Model Response]:\n{text}\n")

                if state_machine.is_complete():
                        print("\n" + "="*60)
                        print("FINAL SUMMARY FROM MODEL")
                        print("="*60)
                        print(text)
                        break

                # otherwise, prompt to continue
                else:
                        reminder = (
                            f"You must generate a molecule and call functions. "
                            f"Progress: {len(state_machine.valid_molecules)}/{TARGET_MOLECULE_COUNT}\n"
                            f"Next: Generate molecule #{len(state_machine.valid_molecules) + 1}"
                        )
                        history.append(types.Content(
                            role="user",
                            parts=[types.Part(text=reminder)]
                        ))

        except Exception as e:
            print(f"\n Error in iteration {state_machine.iteration}: {e}")
            import traceback
            traceback.print_exc()
            break

    # print final summary
    state_machine.print_summary()

    return state_machine.valid_molecules

# RUN MODEL 

pdfpath = "AzobenzeneLightReversibilityPaper.pdf"
API_KEY="AIzaSyDlP0LdGecdeNa_t972EN-7xlNRWPPnZaA"

results = run_molecule_redesign(API_KEY, pdfpath)
print("\n" + "="*60)
print("Completed Workflow")
print("=*60")
print(f"Sucessfully designed {len(results)} molecules!")