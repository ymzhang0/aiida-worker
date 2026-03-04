import numpy as np
from ..common_utils import (
    calculate_pettifor_ratio,
    calculate_pugh_ratio,
    standardize_modulus_names
)

class BornAnalyzer:
    """
    Analyzer for Born charges and elastic properties from MatdynBaseWorkChain.
    """
    
    def __init__(self, node):
        """
        Initialize with an AiiDA WorkChainNode (MatdynBaseWorkChain).
        """
        self.node = node
        self.outputs = node.outputs
        
    def get_elastic_properties(self):
        """
        Extract elastic constants and averages.
        Returns a structured dictionary of elastic properties.
        """
        if 'output_elastic_properties' not in self.outputs:
            return None
            
        raw_props = self.outputs.output_elastic_properties.get_dict()
        elastic_constants = np.array(raw_props.get("elastic_constants", []))
        
        if elastic_constants.size >= 16:
            c12 = elastic_constants[0][1]
            c44 = elastic_constants[3][3]
        else:
            c12 = c44 = 0.0

        results = {
            "elastic_constants": elastic_constants.tolist(),
            "averages": {}
        }

        for avg_name in ['voigt_average', 'reuss_average', 'VRH_average']:
            if avg_name in raw_props:
                avg_data = standardize_modulus_names(raw_props[avg_name])
                
                bulk = avg_data.get('bulk_modulus', 0.0)
                young = avg_data.get('young_modulus', 0.0)
                shear = avg_data.get('shear_modulus', 0.0)
                
                avg_data['pettifor_ratio'] = calculate_pettifor_ratio(c12, c44, young)
                avg_data['modified_pettifor_ratio'] = calculate_pettifor_ratio(c12, c44, bulk)
                avg_data['pugh_ratio'] = calculate_pugh_ratio(bulk, shear)
                
                results["averages"][avg_name] = avg_data
                
        return results

    def get_born_charges(self):
        """Extract Born effective charges."""
        if 'output_born_charges' in self.outputs:
            return self.outputs.output_born_charges.get_dict()
        return None

    def get_dielectric_tensor(self):
        """Extract dielectric tensor."""
        if 'output_dielectric_tensor' in self.outputs:
            return self.outputs.output_dielectric_tensor.get_dict()
        return None

    def check_stability(self, tolerance=-5.0):
        """
        Check phonon stability.
        Returns (is_stable, message, min_freq).
        """
        if 'output_phonon_bands' not in self.outputs:
            return None, "Phonon bands not found", 0.0
            
        THZ_TO_CM = 33.356
        bands = self.outputs.output_phonon_bands.get_bands() * THZ_TO_CM
        min_freq = np.min(bands)
        
        is_stable = min_freq >= tolerance
        msg = f"Phonon is {'stable' if is_stable else 'unstable'}. Min freq: {min_freq:.2f} cm^-1"
        
        return is_stable, msg, float(min_freq)

    def run_all(self):
        """Execute all analysis steps and return a consolidated dictionary."""
        is_stable, stability_msg, min_freq = self.check_stability()
        
        return {
            "pk": self.node.pk,
            "formula": self.node.inputs.structure.get_formula() if 'structure' in self.node.inputs else "Unknown",
            "stability": {
                "is_stable": is_stable,
                "message": stability_msg,
                "min_frequency": min_freq
            },
            "elasticity": self.get_elastic_properties(),
            "born_charges": self.get_born_charges(),
            "dielectric_tensor": self.get_dielectric_tensor()
        }
