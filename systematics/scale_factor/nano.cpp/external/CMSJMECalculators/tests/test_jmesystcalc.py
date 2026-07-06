import pytest
import os, os.path

testData = os.path.join(os.path.dirname(__file__), "data")

if "CMSSW_BASE" in os.environ:  # CMSSW/scram version
    from UserCode.CMSJMECalculators.CMSJMECalculators import loadJMESystematicsCalculators
    from UserCode.CMSJMECalculators.CMSJMECalculators.utils import (
        toRVecFloat,
        toRVecInt,
        getJetMETArgs,
        getFatJetArgs,
    )
    from UserCode.CMSJMECalculators.CMSJMECalculators import config as calcConfigs
else:  # pip version
    from CMSJMECalculators import loadJMESystematicsCalculators
    from CMSJMECalculators.utils import (
        toRVecFloat,
        toRVecInt,
        getJetMETArgs,
        getFatJetArgs,
    )
    from CMSJMECalculators import config as calcConfigs

def getEventWith(f, condition=lambda ev : True, treeName="Events"):
    tup = f.Get(treeName)
    tup.GetEntry(0)
    i = 0
    while not condition(tup):
        i += 1
        tup.GetEntry(i)
    yield tup

@pytest.fixture(scope="module")
def nanojetargsMC18():
    import ROOT as gbl
    f = gbl.TFile.Open(os.path.join(testData, "QCD_Run2018.root"))
    for tup in getEventWith(f, (lambda tup : tup.nJet >= 5)):
        yield getJetMETArgs(tup, isMC=True, forMET=False)

@pytest.fixture(scope="module")
def nanojetargsMC18_postvalues():
    import ROOT as gbl
    f = gbl.TFile.Open(os.path.join(testData, "QCD_Run2018.root"))
    tup = f.Get("Events")
    res = []
    for i in range(tup.GetEntries()):
        tup.GetEntry(i)
        jet_vars = {
            "nominal" : (toRVecFloat(tup.Jet_pt_nominal    ), toRVecFloat(tup.Jet_mass_nominal    )),
            }
        from itertools import chain
        jet_vars.update(dict(chain.from_iterable(
            { f"jer{i:d}{vdir}" : tuple(toRVecFloat(getattr(tup, f"Jet_{ivar}_jer{i:d}{vdir}")) for ivar in ("pt", "mass"))
                for vdir in ("up", "down") }.items() for i in range(6))))
        jet_vars.update(dict(chain.from_iterable(
            { f"jes{src}{vdir}" : tuple(toRVecFloat(getattr(tup, f"Jet_{ivar}_jes{src}{vdir}".format(ivar, src))) for ivar in ("pt", "mass"))
                for vdir in ("up", "down") }.items() for src in ("Total", "FlavorQCD", "TimePtEta")
            )))
        res.append((getJetMETArgs(tup, isMC=True, forMET=False), jet_vars))
    yield res

@pytest.fixture(scope="module")
def nanoMETargsMC18_postvalues():
    import ROOT as gbl
    RVec_float = getattr(gbl, "ROOT::VecOps::RVec<float>")
    f = gbl.TFile.Open(os.path.join(testData, "QCD_Run2018.root"))
    tup = f.Get("Events")
    res = []
    for i in range(tup.GetEntries()):
        tup.GetEntry(i)
        met_vars = {
            "nominal" : (tup.MET_T1_pt_nominal , tup.MET_T1_phi_nominal ),
            "unclustEnup"   : (tup.MET_T1_pt_unclustEnup  , tup.MET_T1_phi_unclustEnup  ),
            "unclustEndown" : (tup.MET_T1_pt_unclustEndown, tup.MET_T1_phi_unclustEndown)
            }
        from itertools import chain
        met_vars.update(dict(
            ("{0}{1}".format(nm, var), (getattr(tup, "MET_T1_pt_{0}{1}".format(nm, var)), getattr(tup, "MET_T1_phi_{0}{1}".format(nm, var))))
            for var in ("up", "down") for nm in [ "jes{0}".format(jsnm) for jsnm in ("Total", "FlavorQCD", "TimePtEta") ]
            ))
        res.append((tuple(getJetMETArgs(tup, isMC=True, forMET=True)), met_vars))
    yield res

@pytest.fixture(scope="module")
def nanoSmearMETargsMC18_postvalues():
    import ROOT as gbl
    RVec_float = getattr(gbl, "ROOT::VecOps::RVec<float>")
    f = gbl.TFile.Open(os.path.join(testData, "QCD_Run2018.root"))
    tup = f.Get("Events")
    res = []
    for i in range(tup.GetEntries()):
        tup.GetEntry(i)
        met_vars = {
            "nominal" : (tup.MET_T1Smear_pt_nominal , tup.MET_T1Smear_phi_nominal ),
            "unclustEnup"   : (tup.MET_T1Smear_pt_unclustEnup  , tup.MET_T1Smear_phi_unclustEnup  ),
            "unclustEndown" : (tup.MET_T1Smear_pt_unclustEndown, tup.MET_T1Smear_phi_unclustEndown)
            }
        from itertools import chain
        met_vars.update(dict(
            ("{0}{1}".format(nm, var), (getattr(tup, "MET_T1Smear_pt_{0}{1}".format(nm, var)), getattr(tup, "MET_T1Smear_phi_{0}{1}".format(nm, var))))
            for var in ("up", "down") for nm in [f"jer{i:d}" for i in range(6)]+[ "jes{0}".format(jsnm) for jsnm in ("Total", "FlavorQCD", "TimePtEta") ]
            ))
        res.append((tuple(getJetMETArgs(tup, isMC=True, forMET=True)), met_vars))
    yield res

@pytest.fixture(scope="module")
def nanojetargsMC18_postvalues_hem():
    import ROOT as gbl
    f = gbl.TFile.Open(os.path.join(testData, "QCD_Run2018.root"))
    tup = f.Get("Events")
    res = []
    for i in range(tup.GetEntries()):
        tup.GetEntry(i)
        jet_vars = {
            "nominal" : (toRVecFloat(tup.Jet_pt_nominal    ), toRVecFloat(tup.Jet_mass_nominal    )),
            }
        from itertools import chain
        jet_vars.update(dict(chain.from_iterable(
            { f"{src}{vdir}" : tuple(toRVecFloat(getattr(tup, f"Jet_{ivar}_{src}{vdir}".format(ivar, src))) for ivar in ("pt", "mass"))
                for vdir in ("up", "down") }.items() for src in [f"jer{i:d}" for i in range(6)]+[ "jes{0}".format(jsnm) for jsnm in ("Total", "FlavorQCD", "TimePtEta") ]
            )))
        res.append((getJetMETArgs(tup, isMC=True, forMET=False, addHEM2018Issue=True), jet_vars))
    yield res

@pytest.fixture(scope="module")
def nanofatjetargsMC22_postvalues():
    import ROOT as gbl
    f = gbl.TFile.Open(os.path.join(testData, "QCD_Run2022.root"))
    tup = f.Get("Events")
    res = []
    for i in range(tup.GetEntries()):
        tup.GetEntry(i)
        jet_vars = {
            "nominal" : (toRVecFloat(tup.FatJet_pt_nominal), toRVecFloat(tup.FatJet_mass_nominal), toRVecFloat(tup.FatJet_msoftdrop_nominal)),
            }
        from itertools import chain
        jet_vars.update(dict(chain.from_iterable(
            { f"jer{i:d}{vdir}" : tuple(toRVecFloat(getattr(tup, f"FatJet_{ivar}_jer{i:d}{vdir}")) for ivar in ("pt", "mass", "msoftdrop"))
                for vdir in ("up", "down") }.items() for i in range(6))))
        jet_vars.update(dict(chain.from_iterable(
            { f"jes{src}{vdir}" : tuple(toRVecFloat(getattr(tup, f"FatJet_{ivar}_jes{src}{vdir}".format(ivar, src))) for ivar in ("pt", "mass", "msoftdrop"))
                for vdir in ("up", "down") }.items() for src in ("Total", "FlavorQCD", "TimePtEta")
            )))
        res.append((getFatJetArgs(tup, isMC=True), jet_vars))
    yield res

@pytest.fixture(scope="module")
def nanofatjetargsMC22():
    import ROOT as gbl
    f = gbl.TFile.Open(os.path.join(testData, "QCD_Run2022.root"))
    for tup in getEventWith(f):
        yield getFatJetArgs(tup, isMC=True, addHEM2018Issue=False)

@pytest.fixture(scope="module")
def jetvarcalcMC18_smear():
    configCls = calcConfigs.JetVariations
    jsonFile = os.path.join(testData,"jet_jerc_2018UL.json.gz")
    config = configCls(jsonFile)
    config.jetAlgo = "AK4PFchs"
    config.jecTag = "Summer19UL18_V5_MC"
    config.jerTag = "Summer19UL18_JRV2_MC"
    config.jecLevel = "L1L2L3Res"
    config.jsonFileSmearingTool = os.path.join(testData,"jer_smear.json.gz")
    config.smearingToolName = "JERSmear"
    config.splitJER = True
    loadJMESystematicsCalculators()
    yield config.create()

@pytest.fixture(scope="module")
def jetvarcalcMC18_jec():
    configCls = calcConfigs.JetVariations
    jsonFile = os.path.join(testData,"jet_jerc_2018UL.json.gz")
    config = configCls(jsonFile)
    config.jetAlgo = "AK4PFchs"
    config.jecTag = "Summer19UL18_V5_MC"
    config.jerTag = "Summer19UL18_JRV2_MC"
    config.jecLevel = "L1L2L3Res"
    config.jsonFileSmearingTool = os.path.join(testData,"jer_smear.json.gz")
    config.smearingToolName = "JERSmear"
    config.splitJER = True
    loadJMESystematicsCalculators()
    yield config.create()

@pytest.fixture(scope="module")
def jetvarcalcMC18_jesunc():
    configCls = calcConfigs.JetVariations
    jsonFile = os.path.join(testData,"jet_jerc_2018UL.json.gz")
    config = configCls(jsonFile)
    config.jetAlgo = "AK4PFchs"
    config.jecTag = "Summer19UL18_V5_MC"
    config.jerTag = "Summer19UL18_JRV2_MC"
    config.jecLevel = "L1L2L3Res"
    config.jesUncertainties = ["Total","FlavorQCD","TimePtEta"]
    config.splitJER = True
    config.jsonFileSmearingTool = os.path.join(testData,"jer_smear.json.gz")
    config.smearingToolName = "JERSmear"
    loadJMESystematicsCalculators()
    yield config.create()

@pytest.fixture(scope="module")
def metvarcalcMC18_jesunc():
    configCls = calcConfigs.METVariations
    jsonFile = os.path.join(testData,"jet_jerc_2018UL.json.gz")
    config = configCls(jsonFile)
    config.jetAlgo = "AK4PFchs"
    config.jecTag = "Summer19UL18_V5_MC"
    config.jerTag = "Summer19UL18_JRV2_MC"
    config.jecLevel = "L1L2L3Res"
    config.jesUncertainties = ["Total","FlavorQCD","TimePtEta"]
    config.splitJER = True
    config.jsonFileSmearingTool = os.path.join(testData,"jer_smear.json.gz")
    config.smearingToolName = "JERSmear"
    loadJMESystematicsCalculators()
    yield config.create()

@pytest.fixture(scope="module")
def jetvarcalcMC18_hem():
    configCls = calcConfigs.JetVariations
    jsonFile = os.path.join(testData,"jet_jerc_2018UL.json.gz")
    config = configCls(jsonFile)
    config.jetAlgo = "AK4PFchs"
    config.jecTag = "Summer19UL18_V5_MC"
    config.jerTag = "Summer19UL18_JRV2_MC"
    config.jecLevel = "L1L2L3Res"
    config.jesUncertainties = ["Total","FlavorQCD","TimePtEta"]
    config.splitJER = True
    config.jsonFileSmearingTool = os.path.join(testData,"jer_smear.json.gz")
    config.smearingToolName = "JERSmear"
    config.addHEM2018Issue = True
    loadJMESystematicsCalculators()
    yield config.create()

@pytest.fixture(scope="module")
def fatjetvarcalcMC22():
    configCls = calcConfigs.FatJetVariations
    jsonFile = os.path.join(testData,"fatjet_jerc_2022EE.json.gz")
    config = configCls(jsonFile)
    config.jetAlgo = "AK8PFPuppi"
    config.jerTag = "Summer22EEPrompt22_JRV1_MC"
    config.splitJER = True
    config.jsonFileSmearingTool = os.path.join(testData,"jer_smear.json.gz")
    config.smearingToolName = "JERSmear"
    config.jecTag = "Summer22EEPrompt22_V1_MC"
    config.jecLevel = "L1L2L3Res"
    config.jesUncertainties = ["Total","FlavorQCD","TimePtEta"]
    config.jsonFileSubjet = os.path.join(testData,"jet_jerc_2022EE.json.gz")
    config.jetAlgoSubjet = "AK4PFPuppi"
    config.jecTagSubjet = "Summer22EEPrompt22_V1_MC"
    config.jecLevelSubjet = "L1L2L3Res"
    loadJMESystematicsCalculators()
    yield config.create()

def test_jetvarcalcMC18_smear(jetvarcalcMC18_smear):
    assert jetvarcalcMC18_smear

def test_jetvarcalcMC18_nano_smear(jetvarcalcMC18_smear, nanojetargsMC18):
    res = jetvarcalcMC18_smear.produce(*nanojetargsMC18)
    assert res

def test_jetvarcalcMC18_nano_jec(jetvarcalcMC18_jec, nanojetargsMC18):
    res = jetvarcalcMC18_jec.produce(*nanojetargsMC18)
    assert res

def test_jetvarcalcMC18_nano_jesunc(jetvarcalcMC18_jesunc, nanojetargsMC18):
    res = jetvarcalcMC18_jesunc.produce(*nanojetargsMC18)
    assert res

def test_fatjetvarcalcMC22_nano(fatjetvarcalcMC22, nanofatjetargsMC22):
    res = fatjetvarcalcMC22.produce(*nanofatjetargsMC22)
    assert res

import math
def isclose_float(a, b, tol=1.):
    import ROOT as gbl
    return math.isclose(a, b, rel_tol=tol*getattr(gbl, "std::numeric_limits<float>").epsilon())

def compareJets(names, calcRes, postValues, tol=1.):
    hasDiff = False
    for ky,(post_pt, post_mass) in postValues.items():
        idx = names.index(ky)
        print(ky, "pt", calcRes.pt(idx), post_pt)
        print(ky, "m ", calcRes.mass(idx), post_mass)
        if not ( all(isclose_float(a,b, tol=tol) for a,b in zip(post_pt, calcRes.pt(idx))) and all(isclose_float(a,b, tol=tol) for a,b in zip(post_mass, calcRes.mass(idx))) ):
            print(f"FAIL: Difference for {ky}")
            hasDiff = True
    return not hasDiff

def compareFatJets(namesAll, namesM, calcRes, postValues, tol=1.):
    hasDiff = False
    for ky,post in postValues.items():
        if ky in namesAll:
            idx = namesAll.index(ky)
            (post_pt, post_mass, post_msd) = post
            print(ky, "pt ", calcRes.pt(idx), post_pt)
            print(ky, "m  ", calcRes.mass(idx), post_mass)
            print(ky, "msd", calcRes.msoftdrop(idx), post_msd)
            eq_pt  = all(isclose_float(a,b, tol=tol) for a,b in zip(post_pt, calcRes.pt(idx)))
            eq_m   = all(isclose_float(a,b, tol=tol) for a,b in zip(post_mass, calcRes.mass(idx)))
            eq_msd = all(isclose_float(a,b, tol=tol) for a,b in zip(post_msd, calcRes.msoftdrop(idx)))
            if not ( eq_pt and eq_m and eq_msd ):
                what = ", ".join((["pt"] if not eq_pt else [])+(["mass"] if not eq_m else [])+(["msd"] if not eq_msd else []))
                print(f"FAIL: Difference for {ky} (in {what})")
                hasDiff = True
        else:
            idx = len(namesAll)+namesM.index(ky)
            (post_mass, post_msd) = post
            print(ky, "m  ", calcRes.mass(idx), post_mass)
            print(ky, "msd", calcRes.msoftdrop(idx), post_msd)
            eq_m   = all(isclose_float(a,b, tol=tol) for a,b in zip(post_mass, calcRes.mass(idx)))
            eq_msd = all(isclose_float(a,b, tol=tol) for a,b in zip(post_msd, calcRes.msoftdrop(idx)))
            if not ( eq_m and eq_msd ):
                what = ", ".join((["mass"] if not eq_m else [])+(["msd"] if not eq_msd else []))
                print(f"FAIL: Difference for {ky} (in {what})")
                hasDiff = True
    return not hasDiff

def compareMET(names, calcRes, postValues, reltol_pt=1.e-6, reltol_phi=1.e-6, abstol_phi=1.e-6):
    hasDiff = False
    for ky,(post_pt, post_phi) in postValues.items():
        idx = names.index(ky)
        print(ky, "pt ", calcRes.pt(idx), post_pt)
        print(ky, "phi", calcRes.phi(idx), post_phi)
        if not ( math.isclose(calcRes.pt(idx), post_pt, rel_tol=1.e-6) and math.isclose(calcRes.phi(idx), post_phi, rel_tol=1.e-6, abs_tol=1.e-6) ):
            print(f"FAIL: Difference for {ky}")
            hasDiff = True
    return not hasDiff

def test_jetvarcalc_nanopost_jesunc(jetvarcalcMC18_jesunc, nanojetargsMC18_postvalues):
    for nanojetargsMC18, postValues in nanojetargsMC18_postvalues:
        assert compareJets([ str(nm) for nm in jetvarcalcMC18_jesunc.available() ],
                jetvarcalcMC18_jesunc.produce(*nanojetargsMC18),
                postValues)

def test_metvarcalc_nanopost_jesunc(metvarcalcMC18_jesunc, nanoMETargsMC18_postvalues):
    metvarcalcMC18_jesunc.setIsT1SmearedMET(False)
    for nanoMETargsMC18, postValues in nanoMETargsMC18_postvalues:
        assert compareMET([ str(nm) for nm in metvarcalcMC18_jesunc.available() ],
                metvarcalcMC18_jesunc.produce(*nanoMETargsMC18),
                postValues)

def test_metvarcalc_nanopost_jesunc_T1Smear(metvarcalcMC18_jesunc, nanoSmearMETargsMC18_postvalues):
    metvarcalcMC18_jesunc.setIsT1SmearedMET(True)
    for nanoMETargsMC18, postValues in nanoSmearMETargsMC18_postvalues:
        assert compareMET([ str(nm) for nm in metvarcalcMC18_jesunc.available() ],
                metvarcalcMC18_jesunc.produce(*nanoMETargsMC18), postValues)

def test_jetvarcalc_nanopost_jesunc_HEM(jetvarcalcMC18_hem, nanojetargsMC18_postvalues_hem):
    for nanojetargsMC18, postValues in nanojetargsMC18_postvalues_hem:
        assert compareJets([ str(nm) for nm in jetvarcalcMC18_hem.available() ],
                jetvarcalcMC18_hem.produce(*nanojetargsMC18),
                postValues, tol=2.)

def test_fatjetvarcalc_nanopost_jesunc(fatjetvarcalcMC22, nanofatjetargsMC22_postvalues):
    for fatjetargs, postValues in nanofatjetargsMC22_postvalues:
        avlAll = [ str(nm) for nm in fatjetvarcalcMC22.available() ]
        avlM = [ str(nm) for nm in fatjetvarcalcMC22.available("mass") if nm not in avlAll ]
        assert compareFatJets(avlAll, avlM,
                fatjetvarcalcMC22.produce(*fatjetargs),
                postValues)
