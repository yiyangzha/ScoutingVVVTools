#include "JetVariationsCalculator.h"
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
  void configureBaseCalc(CALC& calc,
    const std::unique_ptr<CorrectionSet>& cset,
    const std::unique_ptr<CorrectionSet>& csetJerSmear,
    const std::string& jetAlgo,
    const std::string& jecTag,
    const std::string& jecLevel,
    const std::vector<std::string>& jesUncertainties,
    const std::string& jerTag, bool splitJER,
    const std::string& smearingToolName,
    bool doGenMatch, float genMatch_maxDR, float genMatch_maxDPT)
  {
    if (!jecLevel.empty()) {
      const std::string key = jecTag + "_" + jecLevel + "_" + jetAlgo;
      auto maybe_corr = std::find_if(cset->begin(), cset->end(),
            [key](const auto& elem){ return elem.first == key; } );
      if (maybe_corr != cset->end()) {
        calc.setJEC(std::move(cset->at(key)));
      } else {
        calc.setJEC(std::move(cset->compound().at(key)));
      }
    }
    for (const auto& unc: jesUncertainties) {
      const std::string key = jecTag + "_" + unc + "_" + jetAlgo;
      calc.addJESUncertainty(unc, requireCorrection(cset, key, "JetVariationsCalculator JES"));
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
      calc.setSmearing(requireCorrection(cset, resKey, "JetVariationsCalculator JER resolution"),
              requireCorrection(cset, sfKey, "JetVariationsCalculator JER scale factor"),
              requireCorrection(csetJerSmear, jsKey, "JetVariationsCalculator JER smearing tool"),
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
  }
}

JetVariationsCalculator JetVariationsCalculator::create(
  const std::string& jsonFile,
  const std::string& jetAlgo,
  const std::string& jecTag,
  const std::string& jecLevel,
  const std::vector<std::string>& jesUncertainties,
  bool addHEM2018Issue,
  const std::string& jerTag, const std::string& jsonFileSmearingTool,
  const std::string& smearingToolName, bool splitJER,
  bool doGenMatch, float genMatch_maxDR, float genMatch_maxDPT)
{
  JetVariationsCalculator inst{};
  auto cset = CorrectionSet::from_file(jsonFile);
  std::unique_ptr<correction::CorrectionSet> csetJerSmear = jsonFileSmearingTool.empty() ?
  nullptr : CorrectionSet::from_file(jsonFileSmearingTool);
  configureBaseCalc(inst, cset, csetJerSmear, jetAlgo, jecTag, jecLevel, jesUncertainties,
    jerTag, splitJER, smearingToolName, doGenMatch, genMatch_maxDR, genMatch_maxDPT);
  inst.setAddHEM2018Issue(addHEM2018Issue);
  return std::move(inst);
}

JetVariationsCalculator::result_t JetVariationsCalculator::produce(
  const p4compv_t& jet_pt, const p4compv_t& jet_eta, const p4compv_t& jet_phi, const p4compv_t& jet_mass,
  const p4compv_t& jet_rawcorr, const p4compv_t& jet_area, const p4compv_int& jet_jetId,
  const float rho, const p4compv_int& jet_genJetIdx, const p4compv_int& jet_partonFlavour,
  const int seed, const int run, const p4compv_t& genjet_pt, const p4compv_t& genjet_eta,
  const p4compv_t& genjet_phi, const p4compv_t& genjet_mass ) const
{
  const auto nVariations = 1+( m_doSmearing ? 2*( m_splitJER ? 6 : 1 ) : 0 )+2*m_jesUncSources.size()+( m_addHEM2018Issue ? 2 : 0 ); // 1(nom)+2(JER)+2*len(JES)[+2(HEM)]
  LogDebug << "JME:: hello from JetVariations produce. Got " << jet_pt.size() << " jets" << std::endl;
  const auto nJets = jet_pt.size();
  result_t out{nVariations, jet_pt, jet_mass};
  ROOT::VecOps::RVec<double> pt_nom{jet_pt}, mass_nom{jet_mass};
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
  // smearing and JER
  std::size_t iVar = 1; // after nominal
  if ( m_doSmearing ) {
    LogDebug << "JME:: Smearing (seed=" << seed << ")" << std::endl;
    p4compv_t pt_jerUp(pt_nom.size(), 0.), mass_jerUp(mass_nom.size(), 0.);
    p4compv_t pt_jerDown(pt_nom.size(), 0.), mass_jerDown(mass_nom.size(), 0.);
    for ( std::size_t i{0}; i != nJets; ++i ) {
      enum JERSFIndex {Nominal = 0, Up, Down};
      std::array<double, 3> smearFactor = applyJERSmearing(m_jetPtRes, m_jerSmear, m_jetEResSF,
                                                          m_smearDoGenMatch,    
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
        out.set(iVar++, std::move(pt_jeriUp)  , std::move(mass_jeriUp)  );
        out.set(iVar++, std::move(pt_jeriDown), std::move(mass_jeriDown));
      }
    } else {
      out.set(iVar++, std::move(pt_jerUp)  , std::move(mass_jerUp)  );
      out.set(iVar++, std::move(pt_jerDown), std::move(mass_jerDown));
    }
    LogDebug << "JME:: Done with smearing" << std::endl;
  } else {
    LogDebug << "JME:: No smearing" << std::endl;
  }

  // Nominal = first entry in result
  out.set(0, pt_nom, mass_nom);

  // HEM issue 2018, see https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
  if ( m_addHEM2018Issue ) {
    p4compv_t pt_down(pt_nom.size(), 0.), mass_down(mass_nom.size(), 0.);
    for ( std::size_t j{0}; j != nJets; ++j ) {
      const auto delta = deltaHEM2018Issue(pt_nom[j], jet_jetId[j], jet_phi[j], jet_eta[j]);
      pt_down[j] = pt_nom[j]*delta;
      mass_down[j] = mass_nom[j]*delta;
    }
    out.set(iVar++, pt_nom, mass_nom);
    out.set(iVar++, std::move(pt_down), std::move(mass_down));
  }
  // JES uncertainties
  for ( auto& jesUnc : m_jesUncSources ) {
    LogDebug << "JME:: evaluating JES uncertainty: " << jesUnc.first << std::endl;
    p4compv_t pt_jesDown(pt_nom.size(), 0.), mass_jesDown(mass_nom.size(), 0.);
    p4compv_t pt_jesUp(pt_nom.size(), 0.), mass_jesUp(mass_nom.size(), 0.);
    for ( std::size_t i{0}; i != nJets; ++i ) {
      float delta = 0.;
      const auto partonFlav = std::abs(jet_partonFlavour[i]);
      if (isValidFlavorJESUncertainty(jesUnc.first, partonFlav)) {
          delta = jesUnc.second->evaluate({jet_eta[i], pt_nom[i]});
      }
      LogDebug << "JME:: jet " << i << ", parton flavour = " << partonFlav << ", delta = " << delta << std::endl;
      pt_jesDown[i]   = pt_nom[i]*(1.-delta);
      mass_jesDown[i] = mass_nom[i]*(1.-delta);
      pt_jesUp[i]     = pt_nom[i]*(1.+delta);
      mass_jesUp[i]   = mass_nom[i]*(1.+delta);
    }
    out.set(iVar++, std::move(pt_jesUp)  , std::move(mass_jesUp)  );
    out.set(iVar++, std::move(pt_jesDown), std::move(mass_jesDown));
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
    }
    LogDebug << std::endl;
  }
#endif
  return out;
}

std::vector<std::string> JetVariationsCalculator::available(const std::string&) const
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
