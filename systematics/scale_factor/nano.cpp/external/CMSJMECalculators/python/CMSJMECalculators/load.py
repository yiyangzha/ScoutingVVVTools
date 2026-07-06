def loadJMESystematicsCalculators():
    import os
    import importlib.resources
    import importlib.metadata
    import ROOT as gbl
    incDir = importlib.resources.files("correctionlib") / "include"
    libDir = importlib.resources.files("correctionlib") / "lib"
    libName = "libcorrectionlib"
    st = gbl.gSystem.Load(os.path.join(libDir, libName))
    if st == -1:
        raise RuntimeError("Library {0} could not be found".format(libName))
    elif st == -2:
        raise RuntimeError("Version match for library {0}".format(libName))
    gbl.gInterpreter.AddIncludePath(str(incDir))
    gbl.gROOT.ProcessLine('#include "correction.h"')
    try:  # pip version
        incDir = importlib.resources.files("CMSJMECalculators") / "include"
        libDir = importlib.resources.files("CMSJMECalculators") / "lib"
        libName = "libCMSJMECalculators"
        st = gbl.gSystem.Load(os.path.join(libDir, libName))
        if st == -1:
            raise RuntimeError(f"Library {libName} could not be found")
        elif st == -2:
            raise RuntimeError(f"Version mismatch for library {libName}")
        gbl.gInterpreter.AddIncludePath(str(incDir))
        headers = [
            "CMSCalculatorsUtilities.h",
            "JetMETVariationsCalculatorBase.h",
            "JetVariationsCalculator.h",
            "FatJetVariationsCalculator.h",
            "Type1METVariationsCalculator.h",
            "EGammaVariationsCalculator.h",
            "TauVariationsCalculator.h",
            "MuonVariationsCalculator.h"
        ]
        for header in headers:
            gbl.gROOT.ProcessLine(f'#include "{header}"')
    except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError):  # fallback: load directly
        libName = "libCMSJMECalculatorsDict"
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        gbl.gSystem.AddDynamicPath(base_path)
        st = gbl.gSystem.Load(libName)
        if st == -1:
            raise RuntimeError(f"Library {libName} could not be found")
        elif st == -2:
            raise RuntimeError(f"Version mismatch for library {libName}")
    getattr(gbl, "JetVariationsCalculator::result_t")  # Trigger dictionary generation (if needed)
