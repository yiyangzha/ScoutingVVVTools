#include "FatJetVariationsCalculator.h"
#include "CMSCalculatorsUtilities.h"

#include <Math/GenVector/LorentzVector.h>
#include <Math/GenVector/PtEtaPhiM4D.h>
#include "Math/VectorUtil.h"

#include <cassert>
#include <algorithm>

#include <cstdint>
#include <vector>
#include <variant>
#include <string>
#include <stdexcept>

//#define BAMBOO_JME_DEBUG // uncomment to debug

#ifdef BAMBOO_JME_DEBUG
#define LogDebug std::cout
#else
#define LogDebug if (false) std::cout
#endif

using namespace correction;

namespace {
  correction::Correction::Ref requireCorrection(
    const std::unique_ptr<CorrectionSet>& cset,
    const std::string& key,
    const std::string& context)
  {
    try {
      return std::move(cset->at(key));
    } catch (const std::out_of_range&) {
      throw std::runtime_error(context + " missing correction key: " + key);
    }
  }

  template<typename CALC>
  void configureFatjetCalc(CALC& calc,
    const std::unique_ptr<CorrectionSet>& cset,
    const std::unique_ptr<CorrectionSet>& csetJerSmear,
    const std::string& jetAlgo,
    const std::string& jecTag,
    const std::string& jecLevel,
    const std::vector<std::string>& jesUncertainties,
    const std::string& jerTag, bool splitJER,
    const std::string& smearingToolName,
    bool doGenMatch, float genMatch_maxDR, float genMatch_maxDPT,
    const std::unique_ptr<CorrectionSet>& csetSubjet,
    const std::string& jetAlgoSubjet,
    const std::string& jecTagSubjet,
    const std::string& jecLevelSubjet)
  {
    if (!jecLevel.empty()) {
      const std::string key = jecTag + "_" + jecLevel + "_" + jetAlgo;
      auto maybe_corr = std::find_if(cset->begin(), cset->end(),
            [key](const auto& elem){ return elem.first == key; } );
      if ( maybe_corr != cset->end() ) {
        calc.setJEC(std::move(cset->at(key)));
      } else {
        calc.setJEC(std::move(cset->compound().at(key)));
      }
    }
    for (const auto& unc: jesUncertainties) {
      const std::string key = jecTag + "_" + unc + "_" + jetAlgo;
      calc.addJESUncertainty(unc, requireCorrection(cset, key, "FatJetVariationsCalculator JES"));
    }
    if (!jerTag.empty()) {
      const std::string resKey = jerTag + "_PtResolution_" + jetAlgo;
      const std::string sfKey = jerTag + "_ScaleFactor_" + jetAlgo;
      const std::string sfUncKey = jerTag + "_SFUncertainty_" + jetAlgo;
      const std::string jsKey = smearingToolName;
      correction::Correction::Ref sfUnc{};
      auto maybe_sf_unc = std::find_if(cset->begin(), cset->end(),
            [sfUncKey](const auto& elem){ return elem.first == sfUncKey; } );
      if (maybe_sf_unc != cset->end()) {
        sfUnc = std::move(cset->at(sfUncKey));
      }
      calc.setSmearing(requireCorrection(cset, resKey, "FatJetVariationsCalculator JER resolution"),
              requireCorrection(cset, sfKey, "FatJetVariationsCalculator JER scale factor"),
              requireCorrection(csetJerSmear, jsKey, "FatJetVariationsCalculator JER smearing tool"),
              std::move(sfUnc),
              splitJER, doGenMatch, genMatch_maxDR, genMatch_maxDPT);
      calc.setIsMC(true);
    }
    else {
      if (jecTag.size() >= 3 && jecTag.compare(jecTag.size() - 3, 3, "_MC") == 0) {
        calc.setIsMC(true);
      }
      else {
        calc.setIsMC(false);
      }
    }
    // subjets
    if (!jecLevelSubjet.empty()) {
      const std::string key = jecTagSubjet + "_" + jecLevelSubjet + "_" + jetAlgoSubjet;
      auto maybe_corr = std::find_if(csetSubjet->begin(), csetSubjet->end(),
            [key](const auto& elem){ return elem.first == key; } );
      if (maybe_corr != csetSubjet->end()) {
        calc.setJECSubjet(std::move(csetSubjet->at(key)));
      } else {
        calc.setJECSubjet(std::move(csetSubjet->compound().at(key)));
      }
      for (const auto& unc: jesUncertainties) {
        const std::string key = jecTagSubjet + "_" + unc + "_" + jetAlgoSubjet;
        calc.addJESUncertaintySubjet(unc, requireCorrection(csetSubjet, key, "FatJetVariationsCalculator subjet JES"));
      }
    }
  }
}

FatJetVariationsCalculator FatJetVariationsCalculator::create(
  const std::string& jsonFile,
  const std::string& jetAlgo,
  const std::string& jecTag,
  const std::string& jecLevel,
  const std::vector<std::string>& jesUncertainties,
  bool addHEM2018Issue,
  const std::string& jerTag,
  const std::string& jsonFileSmearingTool,
  const std::string& smearingToolName,
  bool splitJER,
  bool doGenMatch,
  float genMatch_maxDR,
  float genMatch_maxDPT,
  const std::string& jsonFileSubjet,
  const std::string& jetAlgoSubjet,
  const std::string& jecTagSubjet,
  const std::string& jecLevelSubjet)
{
  FatJetVariationsCalculator inst{};
  auto cset = CorrectionSet::from_file(jsonFile);

  std::unique_ptr<correction::CorrectionSet> csetJerSmear = jsonFileSmearingTool.empty() ?
  nullptr : CorrectionSet::from_file(jsonFileSmearingTool);

  std::unique_ptr<correction::CorrectionSet> csetSubjet = jsonFileSubjet.empty() ?
  nullptr : CorrectionSet::from_file(jsonFileSubjet);

  configureFatjetCalc(inst, cset, csetJerSmear, jetAlgo, jecTag, jecLevel, jesUncertainties,
    jerTag, splitJER, smearingToolName, doGenMatch, genMatch_maxDR, genMatch_maxDPT,
    csetSubjet, jetAlgoSubjet, jecTagSubjet, jecLevelSubjet);
  inst.setAddHEM2018Issue(addHEM2018Issue);
  return std::move(inst);
}


FatJetVariationsCalculator::result_t FatJetVariationsCalculator::produce(
  const p4compv_t& jet_pt, const p4compv_t& jet_eta, const p4compv_t& jet_phi, const p4compv_t& jet_mass,
  const p4compv_t& jet_rawcorr, const p4compv_t& jet_area, const p4compv_t& jet_msoftdrop,
  const p4compv_int& jet_subJetIdx1, const p4compv_int& jet_subJetIdx2, const p4compv_t& subjet_pt,
  const p4compv_t& subjet_eta, const p4compv_t& subjet_phi, const p4compv_t& subjet_mass, const p4compv_t& subjet_rawFactor,
  const p4compv_int& jet_jetId, const float rho, const p4compv_int& jet_genJetIdx, const int seed, const int run,
  const p4compv_t& genjet_pt, const p4compv_t& genjet_eta, const p4compv_t& genjet_phi, const p4compv_t& genjet_mass) const
{
  const auto nVariations = 1+( m_doSmearing ? 2*( m_splitJER ? 6 : 1 ) : 0 )+2*m_jesUncSources.size()+( m_addHEM2018Issue ? 2 : 0 ); // 1(nom)+2(JER)+2*len(JES)[+2(HEM)]
  LogDebug << "JME:: hello from FatJetVariations produce. Got " << jet_pt.size() << " jets" << std::endl;
  LogDebug << "JME:: variations for PT: " << nVariations << std::endl;
  // Only JEC L1Fastjet depends on area and we dont have subjet_are
  // No problem, since subjets are Puppijets and L1Fastjet is not needed for them 
  p4compv_t subjet_area(subjet_pt.size(), 0.);
  
  const auto nJets = jet_pt.size();
  result_t out{nVariations, jet_pt, jet_mass, jet_msoftdrop};
  ROOT::VecOps::RVec<double> pt_nom{jet_pt}, mass_nom{jet_mass};
  ROOT::VecOps::RVec<double> pt_sj_nom{subjet_pt}, mass_sj_nom{subjet_mass};
  using LVectorM = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;
  auto jet_groomedP4 = std::vector<LVectorM>{nJets};
  ROOT::VecOps::RVec<double> msd_nom(nJets, 0.);

  if ( m_doJEC ) {
    LogDebug << "JME:: reapplying JEC" << std::endl;
    for ( std::size_t i{0}; i != nJets; ++i ) {
      float corr = applyJEC(m_jesSF, m_jecInputAreaRho,
                            m_jecInputRunAreaRho, m_jecInputAreaRhoRun,
                            m_jecInputAreaRhoPhi, m_jecInputAreaRhoPhiRun,
                            jet_area[i], jet_eta[i], jet_phi[i], jet_pt[i],
                            rho, jet_rawcorr[i], run, m_isMC);
      if ( corr > 0. ) {
        const double newc = (1.-jet_rawcorr[i])*corr;
        pt_nom[i]   *= newc;
        mass_nom[i] *= newc;
      }
    }
#ifdef BAMBOO_JME_DEBUG
    LogDebug << "JME:: with reapplied JEC: ";
    for ( std::size_t i{0}; i != nJets; ++i ) {
      LogDebug << "(PT=" << pt_nom[i] << ", ETA=" << jet_eta[i] << ", PHI=" << jet_phi[i] << ", M=" << mass_nom[i] << ") ";
    }
    LogDebug << std::endl;
#endif
  } else {
    LogDebug << "JME:: Not reapplying JEC" << std::endl;
  }

  if ( m_doJECSubjet ) {
    // calculate groomed P4 (and mass)
    for ( std::size_t j{0}; j != nJets; ++j ) {
      if ( jet_subJetIdx1[j] >= 0 && jet_subJetIdx2[j] >= 0 ) {
        
        float corrS1 = applyJEC(m_jesSFSubjet, m_jecInputAreaRhoSubjet,
                                m_jecInputRunAreaRhoSubjet, m_jecInputAreaRhoRunSubjet,
                                m_jecInputAreaRhoPhiSubjet, m_jecInputAreaRhoPhiRunSubjet,
                                subjet_area[jet_subJetIdx1[j]], subjet_eta[jet_subJetIdx1[j]],
                                subjet_phi[jet_subJetIdx1[j]], subjet_pt[jet_subJetIdx1[j]],
                                rho, subjet_rawFactor[jet_subJetIdx1[j]], run, m_isMC);
        
        float corrS2 = applyJEC(m_jesSFSubjet, m_jecInputAreaRhoSubjet,
                                m_jecInputRunAreaRhoSubjet, m_jecInputAreaRhoRunSubjet,
                                m_jecInputAreaRhoPhiSubjet, m_jecInputAreaRhoPhiRunSubjet,
                                subjet_area[jet_subJetIdx2[j]], subjet_eta[jet_subJetIdx2[j]],
                                subjet_phi[jet_subJetIdx2[j]], subjet_pt[jet_subJetIdx2[j]],
                                rho, subjet_rawFactor[jet_subJetIdx2[j]], run, m_isMC);

        if( corrS1 > 0.){
          const double newcS1 = (1.-subjet_rawFactor[jet_subJetIdx1[j]])*corrS1;
          pt_sj_nom[jet_subJetIdx1[j]] *= newcS1;
          mass_sj_nom[jet_subJetIdx1[j]] *= newcS1;
        }

        if( corrS2 > 0.){
          const double newcS2 = (1.-subjet_rawFactor[jet_subJetIdx2[j]])*corrS2;
          pt_sj_nom[jet_subJetIdx2[j]] *= newcS2;
          mass_sj_nom[jet_subJetIdx2[j]] *= newcS2;
        }
        jet_groomedP4[j] = (
            LVectorM(pt_sj_nom[jet_subJetIdx1[j]], subjet_eta[jet_subJetIdx1[j]], subjet_phi[jet_subJetIdx1[j]], mass_sj_nom[jet_subJetIdx1[j]])
          + LVectorM(pt_sj_nom[jet_subJetIdx2[j]], subjet_eta[jet_subJetIdx2[j]], subjet_phi[jet_subJetIdx2[j]], mass_sj_nom[jet_subJetIdx2[j]]));
          msd_nom[j] = jet_groomedP4[j].M();
        if ( msd_nom[j] < 0.0) {
          msd_nom[j] *= -1.;
        }
      }
    }
  }

#ifdef BAMBOO_JME_DEBUG
  LogDebug << "JME:: Groomed momenta: ";
  for ( std::size_t i{0}; i != nJets; ++i ) {
    const auto& p4_g = jet_groomedP4[i];
    LogDebug << "(PT=" << p4_g.Pt() << ", ETA=" << p4_g.Eta() << ", PHI=" << p4_g.Phi() << ") ";
    if ( m_doJECSubjet ) {
      LogDebug <<"(mSD=" <<msd_nom[i]<< ") ";
    }
  }
  LogDebug << std::endl;
#endif
  // smearing and JER
  std::size_t iVar = 1; // after nominal
  if ( m_doSmearing ) {
    LogDebug << "JME:: Smearing (seed=" << seed << ")" << std::endl;
    p4compv_t pt_jerUp(pt_nom.size(), 0.), mass_jerUp(mass_nom.size(), 0.);
    p4compv_t pt_jerDown(pt_nom.size(), 0.), mass_jerDown(mass_nom.size(), 0.);
    for ( std::size_t i{0}; i != nJets; ++i ) {
      enum JERSFIndex {Nominal = 0, Up, Down};
      std::array<double, 3> smearFactor = applyJERSmearing(m_jetPtRes, m_jerSmear, m_jetEResSF, m_smearDoGenMatch,
                                                           pt_nom[i], jet_eta[i], jet_phi[i], jet_genJetIdx[i],
                                                           genjet_pt, genjet_eta, genjet_phi, seed, rho);
      pt_jerDown[i]   = pt_nom[i]*smearFactor[Down];
      mass_jerDown[i] = mass_nom[i]*smearFactor[Down];
      pt_jerUp[i]     = pt_nom[i]*smearFactor[Up];
      mass_jerUp[i]   = mass_nom[i]*smearFactor[Up];
      pt_nom[i]       *= smearFactor[Nominal];
      mass_nom[i]     *= smearFactor[Nominal];
    }
    if ( m_splitJER ) {
      ROOT::VecOps::RVec<int> jerBin(pt_nom.size(), -1);
      for ( std::size_t j{0}; j != nJets; ++j ) {
        jerBin[j] = jerSplitID(pt_nom[j], jet_eta[j]);
      }
      for ( int i{0}; i != 6; ++i ) {
        p4compv_t pt_jeriUp{pt_nom}, mass_jeriUp{mass_nom};
        p4compv_t pt_jeriDown{pt_nom}, mass_jeriDown{mass_nom};
        for ( std::size_t j{0}; j != nJets; ++j ) {
          if ( jerBin[j] == i ) {
            pt_jeriUp[j] = pt_jerUp[j];
            pt_jeriDown[j] = pt_jerDown[j];
            mass_jeriUp[j] = mass_jerUp[j];
            mass_jeriDown[j] = mass_jerDown[j];
          }
        }
        out.set(iVar++, std::move(pt_jeriUp)  , std::move(mass_jeriUp), std::move(msd_nom));
        out.set(iVar++, std::move(pt_jeriDown), std::move(mass_jeriDown), std::move(msd_nom));
      }
    } else {
      out.set(iVar++, std::move(pt_jerUp)  , std::move(mass_jerUp), std::move(msd_nom));
      out.set(iVar++, std::move(pt_jerDown), std::move(mass_jerDown), std::move(msd_nom));
    }
    LogDebug << "JME:: Done with smearing" << std::endl;
  } else {
    LogDebug << "JME:: No smearing" << std::endl;
  }

  // Nominal = first entry in result
  out.set(0, pt_nom, mass_nom, msd_nom);

  // HEM issue 2018, see https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
  if ( m_addHEM2018Issue ) {
    p4compv_t pt_down(pt_nom.size(), 0.), mass_down(mass_nom.size(), 0.), msd_down(msd_nom.size(), 0.);
    auto jet_groomedP4_HEM = std::vector<LVectorM>{nJets};
    for ( std::size_t j{0}; j != nJets; ++j ) {
      const auto delta = deltaHEM2018Issue(pt_nom[j], jet_jetId[j], jet_phi[j], jet_eta[j]);
      pt_down[j] = pt_nom[j]*delta;
      mass_down[j] = mass_nom[j]*delta;
      if ( m_doJECSubjet ) {
        if ( jet_subJetIdx1[j] >= 0 && jet_subJetIdx2[j] >= 0 ) {
          const auto delta1 = deltaHEM2018Issue(pt_sj_nom[jet_subJetIdx1[j]], jet_jetId[j], subjet_phi[jet_subJetIdx1[j]], subjet_eta[jet_subJetIdx1[j]]);
          const auto delta2 = deltaHEM2018Issue(pt_sj_nom[jet_subJetIdx2[j]], jet_jetId[j], subjet_phi[jet_subJetIdx2[j]], subjet_eta[jet_subJetIdx2[j]]);
          jet_groomedP4_HEM[j] = (
              LVectorM(delta1*pt_sj_nom[jet_subJetIdx1[j]], subjet_eta[jet_subJetIdx1[j]], subjet_phi[jet_subJetIdx1[j]], delta1*mass_sj_nom[jet_subJetIdx1[j]])
            + LVectorM(delta2*pt_sj_nom[jet_subJetIdx2[j]], subjet_eta[jet_subJetIdx2[j]], subjet_phi[jet_subJetIdx2[j]], delta2*mass_sj_nom[jet_subJetIdx2[j]]));
          msd_down[j] = jet_groomedP4_HEM[j].M();
          if ( msd_down[j] < 0.0 ) {
            msd_down[j] *= -1.;
          }
        }
      }
    }
    if ( m_doJECSubjet ) {
      out.set(iVar++, pt_nom, mass_nom, msd_nom);
      out.set(iVar++, std::move(pt_down), std::move(mass_down), std::move(msd_down));
    } else {
      out.set(iVar++, pt_nom, mass_nom, msd_nom);
      out.set(iVar++, std::move(pt_down), std::move(mass_down), msd_nom);
    }
  }
  // JES uncertainties
  for ( auto& jesUnc : m_jesUncSources ) {
    LogDebug << "JME:: evaluating JES uncertainty: " << jesUnc.first << std::endl;
    p4compv_t pt_jesDown(pt_nom.size(), 0.), mass_jesDown(mass_nom.size(), 0.), msd_jesDown(msd_nom.size(), 0.);
    p4compv_t pt_jesUp(pt_nom.size(), 0.), mass_jesUp(mass_nom.size(), 0.), msd_jesUp(msd_nom.size(), 0.);
    auto jet_groomedP4_jesDown = std::vector<LVectorM>{nJets};
    auto jet_groomedP4_jesUp = std::vector<LVectorM>{nJets};
    for ( std::size_t i{0}; i != nJets; ++i ) {
      const auto delta = jesUnc.second->evaluate({jet_eta[i], pt_nom[i]});
      LogDebug << "JME:: jet " << i << ", delta = " << delta << std::endl;
      pt_jesDown[i]   = pt_nom[i]*(1.-delta);
      mass_jesDown[i] = mass_nom[i]*(1.-delta);
      pt_jesUp[i]     = pt_nom[i]*(1.+delta);
      mass_jesUp[i]   = mass_nom[i]*(1.+delta);
      if ( m_doJECSubjet ) {
        if ( jet_subJetIdx1[i] >= 0 && jet_subJetIdx2[i] >= 0 ) {
          const auto delta1 = jesUnc.second->evaluate({subjet_eta[jet_subJetIdx1[i]], pt_sj_nom[jet_subJetIdx1[i]]});
          const auto delta2 = jesUnc.second->evaluate({subjet_eta[jet_subJetIdx2[i]], pt_sj_nom[jet_subJetIdx2[i]]});
          jet_groomedP4_jesDown[i] = (
              LVectorM((1.-delta1)*pt_sj_nom[jet_subJetIdx1[i]], subjet_eta[jet_subJetIdx1[i]], subjet_phi[jet_subJetIdx1[i]], (1.-delta1)*mass_sj_nom[jet_subJetIdx1[i]])
            + LVectorM((1.-delta2)*pt_sj_nom[jet_subJetIdx2[i]], subjet_eta[jet_subJetIdx2[i]], subjet_phi[jet_subJetIdx2[i]], (1.-delta2)*mass_sj_nom[jet_subJetIdx2[i]]));
          msd_jesDown[i] = jet_groomedP4_jesDown[i].M();
          if ( msd_jesDown[i] < 0.0 ) {
            msd_jesDown[i] *= -1.;
          }
          jet_groomedP4_jesUp[i] = (
              LVectorM((1.+delta1)*pt_sj_nom[jet_subJetIdx1[i]], subjet_eta[jet_subJetIdx1[i]], subjet_phi[jet_subJetIdx1[i]], (1.+delta1)*mass_sj_nom[jet_subJetIdx1[i]])
            + LVectorM((1.+delta2)*pt_sj_nom[jet_subJetIdx2[i]], subjet_eta[jet_subJetIdx2[i]], subjet_phi[jet_subJetIdx2[i]], (1.+delta2)*mass_sj_nom[jet_subJetIdx2[i]]));
          msd_jesUp[i] = jet_groomedP4_jesUp[i].M();
          if ( msd_jesUp[i] < 0.0 ) {
            msd_jesUp[i] *= -1.;
          }
        }
      }
    }
    if ( m_doJECSubjet ) {
      out.set(iVar++, std::move(pt_jesUp)  , std::move(mass_jesUp)  , std::move(msd_jesUp));
      out.set(iVar++, std::move(pt_jesDown), std::move(mass_jesDown), std::move(msd_jesDown));
    } else {
      out.set(iVar++, std::move(pt_jesUp)  , std::move(mass_jesUp), msd_nom);
      out.set(iVar++, std::move(pt_jesDown), std::move(mass_jesDown), msd_nom);
    }
  }

#ifdef BAMBOO_JME_DEBUG
  assert(iVar == out.size());
  LogDebug << "JME:: returning " << out.size() << " modified jet collections" << std::endl;
  const auto varNames = available();
  assert(varNames.size() == nVariations);
  for ( std::size_t i{0}; i != nVariations; ++i ) {
    LogDebug << "JME:: Jet_" << varNames[i] << ": ";
    for ( std::size_t j{0}; j != nJets; ++j ) {
      LogDebug << "(PT=" << out.pt(i)[j] << ", ETA=" << jet_eta[j] << ", PHI=" << jet_phi[j] << ", M=" << out.mass(i)[j] << ") ";
      if ( m_doJECSubjet ) {
        LogDebug << "(mSD=" << out.msoftdrop(i)[j] <<") ";
      }
    }
    LogDebug << std::endl;
  }
#endif
  return out;
}


std::vector<std::string> FatJetVariationsCalculator::available(const std::string&) const
{
  std::vector<std::string> products = { "nominal" };
  if ( m_doSmearing ) {
    if ( m_splitJER ) {
      for ( int i = 0; i != 6; ++i ) {
        products.emplace_back("jer"+std::to_string(i)+"up");
        products.emplace_back("jer"+std::to_string(i)+"down");
      }
    } else {
      products.emplace_back("jerup");
      products.emplace_back("jerdown");
    }
  }
  if ( m_addHEM2018Issue ) {
    products.emplace_back("jesHEMIssueup");
    products.emplace_back("jesHEMIssuedown");
  }
  for ( const auto& src : m_jesUncSources ) {
    products.emplace_back("jes"+src.first+"up");
    products.emplace_back("jes"+src.first+"down");
  }
  return products;
}
