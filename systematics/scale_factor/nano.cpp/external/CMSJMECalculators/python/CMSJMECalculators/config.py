from dataclasses import dataclass, field, fields
from typing import ClassVar, List, Tuple
from pathlib import Path

@dataclass
class ConfigBase:
    jsonFile: str

    def __post_init__(self):
        if isinstance(self.jsonFile, Path):
            self.jsonFile = str(self.jsonFile)

    @property
    def params(self):
        pass

    def toCppStr(self, param):
        if isinstance(param, str):
            return f'"{param}"'
        elif isinstance(param, bool):
            return ("true" if param else "false")
        elif isinstance(param, float) or isinstance(param, int):
            return str(param)
        elif hasattr(param, "__iter__"):
            return f'{{{", ".join(self.toCppStr(p) for p in param)}}}'
        else:
            raise ValueError(f"Parameter of unsupported type: {param!r}")

    def toArg(self, value, destType):
        from cppyy import gbl
        if destType in (str, int, float, bool):
            return value
        elif destType == List[str]:
            vc = gbl.std.vector["std::string"]()
            for elm in value:
                vc.push_back(elm)
            return vc
        elif destType == List[float]:
            vc = gbl.std.vector["double"]()
            for elm in value:
                vc.push_back(elm)
            return vc
        elif destType == Tuple[float, float, float, float, float, float]:
            arr = gbl.std.array["double,6"]()
            for i, v in enumerate(value):
                arr[i] = v
            return arr
        else:
            raise RuntimeError(f"Cannot deal with type {destType!r} for value {value}")

    @property
    def cppConstruct(self):
        return (f"{self.calcClass}::create"
                f"({', '.join(self.toCppStr(getattr(self, p)) for p in self.params)})")

    def create(self):
        fields_by_name = {f.name: f for f in fields(self)}
        from cppyy import gbl
        return getattr(gbl, self.calcClass).create(*(
            self.toArg(getattr(self, p), fields_by_name[p].type)
            for p in self.params
        ))


@dataclass
class JetMETVariationsConfigBase(ConfigBase):
    jetAlgo: str = "AK4PFPuppi"
    jecTag: str = ""
    jecLevel: str = ""
    jesUncertainties: List[str] = field(default_factory=list)
    jerTag: str = ""
    jsonFileSmearingTool: str = ""
    smearingToolName: str = "JERSmear"
    splitJER: bool = False
    useGenMatch: bool = True
    genMatchDR: float = 0.2
    genMatchDPt: float = 3.

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.jsonFileSmearingTool, Path):
            self.jsonFileSmearingTool = str(self.jsonFileSmearingTool)

    @property
    def jerParams(self):
        return ["jerTag", "jsonFileSmearingTool", "smearingToolName", "splitJER", "useGenMatch", "genMatchDR", "genMatchDPt"]


@dataclass
class TauVariations(ConfigBase):
    calcClass: ClassVar[str] = "TauVariationsCalculator"
    tauCorr: str = "tau_energy_scale"
    tauAlgo: str = "DeepTau2017v2p1"
    tauWP: str = ""
    tauWPvsE: str = ""
    addSystematics: bool = True
    splitSystematics: bool = False

    @property
    def params(self):
        return ["jsonFile", "tauCorr", "tauAlgo", "tauWP", "tauWPvsE", "addSystematics", "splitSystematics"]


@dataclass
class MuonVariations(ConfigBase):
    calcClass: ClassVar[str] = "MuonVariationsCalculator"
    addSystematics: bool = True
    isMC: bool = False
    jsonFileSmearingTool: str = ""
    smearingTool: str = "stdflat"

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.jsonFileSmearingTool, Path):
            self.jsonFileSmearingTool = str(self.jsonFileSmearingTool)

    @property
    def params(self):
        return ["jsonFile", "addSystematics", "isMC",
                "jsonFileSmearingTool", "smearingTool"]


@dataclass
class EGammaVariations(ConfigBase):
    calcClass: ClassVar[str] = "EGammaVariationsCalculator"
    scale: str = ""
    smearing: str = ""
    isMC: bool = False
    addSystematics: bool = True
    jsonFileSmearingTool: str = ""
    smearingTool: str = "stdnormal"

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.jsonFileSmearingTool, Path):
            self.jsonFileSmearingTool = str(self.jsonFileSmearingTool)

    @property
    def params(self):
        return ["jsonFile", "scale", "smearing", "isMC", "addSystematics",
                "jsonFileSmearingTool", "smearingTool"]


@dataclass
class JetVariations(JetMETVariationsConfigBase):
    calcClass: ClassVar[str] = "JetVariationsCalculator"
    addHEM2018Issue: bool = False

    @property
    def params(self):
        return ["jsonFile", "jetAlgo", "jecTag", "jecLevel",
                "jesUncertainties", "addHEM2018Issue"] + self.jerParams


@dataclass
class FatJetVariations(JetVariations):
    calcClass: ClassVar[str] = "FatJetVariationsCalculator"
    addHEM2018Issue: bool = False
    genMatchDR: float = 0.4
    jsonFileSubjet: str = ""
    jetAlgoSubjet: str = ""
    jecTagSubjet: str = ""
    jecLevelSubjet: str = ""

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.jsonFileSubjet, Path):
            self.jsonFileSubjet = str(self.jsonFileSubjet)

    @property
    def params(self):
        return (["jsonFile", "jetAlgo", "jecTag", "jecLevel",
                "jesUncertainties", "addHEM2018Issue"] + self.jerParams 
                + ["jsonFileSubjet", "jetAlgoSubjet", "jecTagSubjet", "jecLevelSubjet"])


@dataclass
class METVariationsConfigBase(JetMETVariationsConfigBase):
    l1JecTag: str = "L1FastJet"
    unclEnThreshold: float = 15.
    emEnFracThreshold: float = 0.9
    isT1SmearedMET: bool = False
    isXYCorrMET: bool = False
    jsonXYCorr: str = ""
    eraForXYCorrMET: str = ""
    isMC: bool = False

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.jsonXYCorr, Path):
            self.jsonXYCorr = str(self.jsonXYCorr)


@dataclass
class METVariations(METVariationsConfigBase):
    calcClass: ClassVar[str] = "Type1METVariationsCalculator"
    addHEM2018Issue: bool = False
    isT1SmearedMET: bool = False
    unclEnThreshold: float = 15.
    emEnFracThreshold: float = 0.9

    @property
    def params(self):
        return ["jsonFile", "jetAlgo", "jecTag", "jecLevel",
                "l1JecTag", "unclEnThreshold", "emEnFracThreshold",
                "jesUncertainties", "addHEM2018Issue", "isT1SmearedMET",
                "isXYCorrMET", "jsonXYCorr", "eraForXYCorrMET", "isMC"] + self.jerParams

