import pandas as pd
import numpy as np
from src.data.hybrid_features import HybridFeatureEngineer

class CalibratedFeatureEngineer(HybridFeatureEngineer):
    """
    V6 CALIBRATED RESTORATION:
    Restores the full 199-feature stacking from V4 (HybridFeatureEngineer).
    
    Inheritance:
    - Inherits from V4's HybridFeatureEngineer which already implements
      the V1 Technicals + FracDiff (d=0.4) stacking.
      
    This class exists to provide a clean interface for V6 and allow for 
    any future V6-specific engineering tweaks if needed.
    """
    def __init__(self, config=None, d=0.4, floor=1e-3):
        super().__init__(config, d, floor)
        
    def get_feature_columns(self):
        # Simply return the full set (199 features)
        return super().get_feature_columns()
