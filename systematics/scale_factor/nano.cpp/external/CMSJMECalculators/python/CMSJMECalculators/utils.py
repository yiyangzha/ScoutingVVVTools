import logging
logger = logging.getLogger(__name__)

import ROOT as gbl

RVec_float = getattr(gbl, "ROOT::VecOps::RVec<float>")
RVec_int = getattr(gbl, "ROOT::VecOps::RVec<int>")


def toRVecFloat(values):
    """ Convert values to an RVec<float> """
    res = RVec_float(len(values), 0.)
    for i,val in enumerate(values):
        res[i] = val
    return res


def toRVecInt(values):
    """ Convert values to an RVec<int> """
    res = RVec_int(len(values), 0)
    for i,val in enumerate(values):
        res[i] = val
    return res


def getMETUnclDeltaXY(ptNom, phiNom, ptUp, phiUp):
    """ Calculate MetUnclustEnUpDeltaX and MetUnclustEnUpDeltaY from MET_ptUnclusteredUp and MET_phiUnclusteredUp"""
    p4Nom = gbl.TLorentzVector()
    p4Nom.SetPtEtaPhiE(ptNom, 0., phiNom, 0.)
    p4Up = gbl.TLorentzVector()
    p4Up.SetPtEtaPhiE(ptUp, 0., phiUp, 0.)
    p4 = p4Up - p4Nom
    return p4.Px(), p4.Py()


def getTauArgs(tup):
    """ Get the input values for tau variations calculator from a tree (PyROOT-style) """
    args = [
        toRVecFloat(tup.Tau_pt),
        toRVecFloat(tup.Tau_eta),
        toRVecFloat(tup.Tau_mass),
        toRVecInt(tup.Tau_decayMode),
        toRVecInt(tup.Tau_genPartFlav)
    ]

    return args


def getMuonArgs(tup):
    """ Get the input values for tau variations calculator from a tree (PyROOT-style) """
    args = [
        toRVecFloat(tup.Muon_pt),
        toRVecFloat(tup.Muon_eta),
        toRVecFloat(tup.Muon_phi),
        toRVecInt(tup.Muon_charge),
        toRVecInt(tup.Muon_nTrackerLayers),
        tup.event
    ]

    return args


def getElectronArgs(tup):
    """ Get the input values for electron variations calculator from a tree (PyROOT-style) """
    args = [
        toRVecFloat(tup.Electron_pt),
        toRVecFloat(tup.Electron_eta),
        toRVecFloat(tup.Electron_deltaEtaSC),
        toRVecFloat(tup.Electron_phi),
        toRVecFloat(tup.Electron_r9),
        toRVecInt(tup.Electron_seedGain),
        tup.run,
        tup.event
    ]

    return args


def getJetMETArgs(tup, isMC=True, forMET=False, PuppiMET=False, addHEM2018Issue=False):
    """ Get the input values for the jet/met variations calculator from a tree (PyROOT-style) """
    args = [
        toRVecFloat(tup.Jet_pt),
        toRVecFloat(tup.Jet_eta),
        toRVecFloat(tup.Jet_phi),
        toRVecFloat(tup.Jet_mass),
        toRVecFloat(tup.Jet_rawFactor),
        toRVecFloat(tup.Jet_area),
        ]
    if forMET:
        args += [
            toRVecFloat(tup.Jet_muonSubtrFactor),
            toRVecFloat(tup.Jet_neEmEF),
            toRVecFloat(tup.Jet_chEmEF),
            ]
    args.append(toRVecInt(tup.Jet_jetId if addHEM2018Issue else []))
    if hasattr(tup, "fixedGridRhoFastjetAll"):
        args.append(tup.fixedGridRhoFastjetAll)
    elif hasattr(tup, "Rho_fixedGridRhoFastjetAll"):
        args.append(tup.Rho_fixedGridRhoFastjetAll)
    else:
        raise RuntimeError("Could not find fixedGridRhoFastjetAll or Rho_fixedGridRhoFastjetAll in tree")
    if isMC:
        args += [
            toRVecInt(tup.Jet_genJetIdx),
            toRVecInt(tup.Jet_partonFlavour),
            tup.event,
            tup.run,
            toRVecFloat(tup.GenJet_pt),
            toRVecFloat(tup.GenJet_eta),
            toRVecFloat(tup.GenJet_phi),
            toRVecFloat(tup.GenJet_mass)
            ]
    else:
        args += [ toRVecInt([]), toRVecInt([]), 0, toRVecFloat([]), toRVecFloat([]), toRVecFloat([]), toRVecFloat([]) ]
    if forMET:
        if PuppiMET:
            args += [ tup.RawPuppiMET_phi, tup.RawPuppiMET_pt ]
        else:
            args += [ tup.RawMET_phi, tup.RawMET_pt ]
        args += [ toRVecFloat(getattr(tup, "CorrT1METJet_{0}".format(varNm)))
                for varNm in ("rawPt", "eta", "phi", "area", "muonSubtrFactor") ]
        args += [ toRVecFloat([]), toRVecFloat([]) ]
        if PuppiMET:
            if getattr(tup, "PuppiMET_ptUnclusteredUp") is not None and getattr(tup, "PuppiMET_phiUnclusteredUp") is not None:
                MetUnclustEnUpDeltaX, MetUnclustEnUpDeltaY = getMETUnclDeltaXY(
                    tup.PuppiMET_pt, tup.PuppiMET_phi, tup.PuppiMET_ptUnclusteredUp, tup.PuppiMET_phiUnclusteredUp)
                args += [ MetUnclustEnUpDeltaX, MetUnclustEnUpDeltaY]
            elif getattr(tup, "PuppiMET_MetUnclustEnUpDeltaX") is not None and getattr(tup, "PuppiMET_MetUnclustEnUpDeltaY") is not None:
                args += [ tup.PuppiMET_MetUnclustEnUpDeltaX, tup.PuppiMET_MetUnclustEnUpDeltaY]
            else:
                raise RuntimeError(f"None of the variables related to MET Uncl. energy found.")
        else:
            args += [ tup.MET_MetUnclustEnUpDeltaX, tup.MET_MetUnclustEnUpDeltaY]
        args += [tup.PV_npvsGood]

    return args


def getFatJetArgs(tup, isMC=True, addHEM2018Issue=False):
    """ Get the input values for the jet variations calculator for a fat jet from a tree (PyROOT-style) """
    args = [
        toRVecFloat(tup.FatJet_pt),
        toRVecFloat(tup.FatJet_eta),
        toRVecFloat(tup.FatJet_phi),
        toRVecFloat(tup.FatJet_mass),
        toRVecFloat(tup.FatJet_rawFactor),
        toRVecFloat(tup.FatJet_area),
        toRVecFloat(tup.FatJet_msoftdrop),
        toRVecInt(tup.FatJet_subJetIdx1),
        toRVecInt(tup.FatJet_subJetIdx2),
        toRVecFloat(tup.SubJet_pt),
        toRVecFloat(tup.SubJet_eta),
        toRVecFloat(tup.SubJet_phi),
        toRVecFloat(tup.SubJet_mass),
        toRVecFloat(tup.SubJet_rawFactor),
        toRVecInt(tup.FatJet_jetId if addHEM2018Issue else [])
        ]
    if hasattr(tup, "fixedGridRhoFastjetAll"):
        args.append(tup.fixedGridRhoFastjetAll)
    elif hasattr(tup, "Rho_fixedGridRhoFastjetAll"):
        args.append(tup.Rho_fixedGridRhoFastjetAll)
    else:
        raise RuntimeError("Could not find fixedGridRhoFastjetAll or Rho_fixedGridRhoFastjetAll in tree")
    if isMC:
        args += [
            toRVecInt(tup.FatJet_genJetAK8Idx),
            tup.event,
            tup.run,
            toRVecFloat(tup.GenJetAK8_pt),
            toRVecFloat(tup.GenJetAK8_eta),
            toRVecFloat(tup.GenJetAK8_phi),
            toRVecFloat(tup.GenJetAK8_mass)
            ]
    else:
        args += [ toRVecInt([]), 0, tup.run, toRVecFloat([]), toRVecFloat([]), toRVecFloat([]), toRVecFloat([])]

    return args


def getJetMETArgsCustom(
        tup, isMC=True, forMET=False, PuppiMET=False, addHEM2018Issue=False,
        JetCollection="Jet", JetMuonSubtrFactor="muonSubtrFactor", JetNeEmEF="neEmEF",
        JetcChEmEF="chEmEF", GenJetCollection="GenJet", JetGenJetIdx="genJetIdx",
        RawMETCollection="RawPuppiMET", LowPtJetCollection="CorrT1METJet"):
    """ 
    Get the input values for the jet/met variations calculator from a tree (PyROOT-style)
    This configuration is more general and can be used for custom NanoAOD (e.g. JMENano) 
    as well as non-standard jet/met collections. 
    One needs to provide this info: JetCollection, GenJetCollection, RawMETCollection, and LowPtJetCollection.
    A dummy value is used for uncertainties related to the unclustered energy.
    """
    rvec_zero_int = RVec_int(len(getattr(tup, "{0}_pt".format(JetCollection))), 0)
    rvec_mOne_int = RVec_int(len(getattr(tup, "{0}_pt".format(JetCollection))), -1)
    rvec_zero_float = RVec_float(len(getattr(tup, "{0}_pt".format(JetCollection))), 0.)

    args = [ toRVecFloat(getattr(tup, "{0}_{1}".format(JetCollection,varNm)))
            for varNm in ("pt", "eta", "phi", "mass", "rawFactor", "area") ]
    if forMET:
        if not JetMuonSubtrFactor:
            args.append(rvec_zero_float)
        else:
            args.append(toRVecFloat(getattr(tup, "{0}_{1}".format(JetCollection,JetMuonSubtrFactor))))
        if not JetNeEmEF:
            args.append(rvec_zero_float)
        else:
            args.append(toRVecFloat(getattr(tup, "{0}_{1}".format(JetCollection,JetNeEmEF))))
        if not JetcChEmEF:
            args.append(rvec_zero_float)
        else:
            args.append(toRVecFloat(getattr(tup, "{0}_{1}".format(JetCollection,JetcChEmEF))))
    args.append(toRVecInt(getattr(tup, "{0}_{1}".format(JetCollection,"jetId")) if addHEM2018Issue else []))
    rhoNames = ["fixedGridRhoFastjetAll","Rho_fixedGridRhoFastjetAll"]
    for rhoName in rhoNames:
        try:
            rho = getattr(tup, rhoName)
            args.append(rho)
            break
        except AttributeError:
            pass
    else:
        raise RuntimeError(f"None of the attributes {rhoNames} found.")
    if isMC:
        if not JetGenJetIdx:
            args.append(rvec_mOne_int)
        else:
            args.append(tup,"{0}_{1}".format(JetCollection,JetGenJetIdx))
        if not JetPartonFlavour:
            args.append(rvec_zero_int)
        else:
            args.append(toRVecInt(getattr(tup, "{0}_{1}".format(JetCollection,JetPartonFlavour))))
        args.append(tup.event)
        args.append(tup.run)
        args += [ toRVecFloat(getattr(tup, "{0}_{1}".format(GenJetCollection,varNm)))
                for varNm in ("pt", "eta", "phi", "mass") ]
    else:
        args += [ toRVecInt([]), toRVecInt([]), 0, tup.run, toRVecFloat([]), toRVecFloat([]), toRVecFloat([]), toRVecFloat([]) ]
    if forMET:
        args += [getattr(tup,"{0}_{1}".format(RawMETCollection,"phi")), getattr(tup,"{0}_{1}".format(RawMETCollection,"pt"))]

    if not LowPtJetCollection:
        args += [ toRVecFloat([]), toRVecFloat([]), toRVecFloat([]), toRVecFloat([]),
                 toRVecFloat([]), toRVecFloat([]), toRVecFloat([])]
    else:
        args += [ toRVecFloat(getattr(tup, "{0}_{1}".format(LowPtJetCollection,varNm)))
                for varNm in ("rawPt", "eta", "phi", "area", "muonSubtrFactor") ]
        args += [ toRVecFloat([]), toRVecFloat([]) ]
    ## Unclustered Energy - let use dummy value 0,0
    args.append(0.)
    args.append(0.)
    args += [tup.PV_npvsGood]

    return args

def getJetMETArgsNames(isMC=True, forMET=False, PuppiMET=False, addHEM2018Issue=False):
    """ Get the input values for the jet/met variations calculator as a list of strings """
    args = [
        "Jet_pt",
        "Jet_eta",
        "Jet_phi",
        "Jet_mass",
        "Jet_rawFactor",
        "Jet_area",
    ]
    if forMET:
        args += [
            "Jet_muonSubtrFactor",
            "Jet_neEmEF",
            "Jet_chEmEF",
        ]
    args.append("Jet_jetId" if addHEM2018Issue else [])
    args.append("fixedGridRhoFastjetAll")
    if isMC:
        args += [
            "Jet_genJetIdx",
            "Jet_partonFlavour",
            "event",
            "run",
            "GenJet_pt",
            "GenJet_eta",
            "GenJet_phi",
            "GenJet_mass",
        ]
    else:
        args += [ "RVec<int>()", "RVec<int>()", 0, "run", "RVec<float>()", "RVec<float>()", "RVec<float>()", "RVec<float>()" ]
    if forMET:
        if PuppiMET:
            args += ["RawPuppiMET_phi", "RawPuppiMET_pt"]
        else:
            args += ["RawMET_phi", "RawMET_pt"]
        args += ["CorrT1METJet_rawPt", "CorrT1METJet_eta", "CorrT1METJet_phi", "CorrT1METJet_area", "CorrT1METJet_muonSubtrFactor"]
        args += ["RVec<float>()", "RVec<float>()"]
        if PuppiMET:
            args += ["PuppiMET_MetUnclustEnUpDeltaX", "PuppiMET_MetUnclustEnUpDeltaY"]
        else:
            args += ["MET_MetUnclustEnUpDeltaX", "MET_MetUnclustEnUpDeltaY"]

    return args
