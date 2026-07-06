#include "Type1METVariationsCalculator.h"
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
  void configureMETCalc_common(CALC& calc,
    const std::unique_ptr<CorrectionSet>& cset,
    const std::unique_ptr<CorrectionSet>& csetJerSmear,
    const std::unique_ptr<CorrectionSet>& csetXYCorr,
    const std::string& jetAlgo,
    const std::string& jecTag,
    const std::string& jecLevel,
    const std::string& l1jec,
    float unclEnThreshold,
    float emEnFracThreshold,
    const std::vector<std::string>& jesUncertainties,
    bool isT1SmearedMET,
    const std::string& jerTag, bool splitJER,
    const std::string& smearingToolName,
    bool doGenMatch, float genMatch_maxDR, float genMatch_maxDPT,
    bool isXYCorrMET)
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
      calc.addJESUncertainty(unc, requireCorrection(cset, key, "Type1METVariationsCalculator JES"));
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
      calc.setSmearing(requireCorrection(cset, resKey, "Type1METVariationsCalculator JER resolution"),
                requireCorrection(cset, sfKey, "Type1METVariationsCalculator JER scale factor"),
                requireCorrection(csetJerSmear, jsKey, "Type1METVariationsCalculator JER smearing tool"),
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
    if (!l1jec.empty()){
      const std::string l1Key = jecTag + "_" + l1jec + "_" + jetAlgo;
      calc.setL1JEC(requireCorrection(cset, l1Key, "Type1METVariationsCalculator L1 JEC"));
    }
    if (isXYCorrMET){
      const std::string key = "met_xy_corrections";
      calc.setXYCorrMET(std::move(csetXYCorr->at(key)));
    }
  }
}

Type1METVariationsCalculator Type1METVariationsCalculator::create(
  const std::string& jsonFile,
  const std::string& jetAlgo,
  const std::string& jecTag,
  const std::string& jecLevel,
  const std::string& l1jec,
  float unclEnThreshold,
  float emEnFracThreshold,
  const std::vector<std::string>& jesUncertainties,
  bool addHEM2018Issue,
  bool isT1SmearedMET,
  bool isXYCorrMET,
  const std::string& jsonXYCorrMET,
  const std::string& eraForXYCorrMET,
  bool isMC,
  const std::string& jerTag, const std::string& jsonFileSmearingTool,
  const std::string& smearingToolName, bool splitJER,
  bool doGenMatch, float genMatch_maxDR, float genMatch_maxDPT)
{
 Type1METVariationsCalculator inst{};
 auto cset = CorrectionSet::from_file(jsonFile);
 std::unique_ptr<correction::CorrectionSet> csetJerSmear = jsonFileSmearingTool.empty() ?
 nullptr : CorrectionSet::from_file(jsonFileSmearingTool);
 std::unique_ptr<correction::CorrectionSet> csetXYcorr = jsonXYCorrMET.empty() ?
 nullptr : CorrectionSet::from_file(jsonXYCorrMET);
 configureMETCalc_common(inst, cset, csetJerSmear, csetXYcorr, jetAlgo, jecTag,
  jecLevel, l1jec, unclEnThreshold, emEnFracThreshold, jesUncertainties,
  isT1SmearedMET, jerTag, splitJER, smearingToolName, doGenMatch,
  genMatch_maxDR, genMatch_maxDPT, isXYCorrMET);
 inst.setAddHEM2018Issue(addHEM2018Issue);
 inst.setUnclusteredEnergyTreshold(unclEnThreshold);
 inst.setEmEnergyFracThreshold(emEnFracThreshold);
 inst.setIsT1SmearedMET(isT1SmearedMET);
 inst.setIsXYcorrMET(isXYCorrMET);
 inst.setEraXYCorrMET(eraForXYCorrMET);
 inst.setIsMCXYCorrMET(isMC);
 return inst;
}

Type1METVariationsCalculator::result_t Type1METVariationsCalculator::produce(
  const p4compv_t& jet_pt, const p4compv_t& jet_eta, const p4compv_t& jet_phi, const p4compv_t& jet_mass,
  const p4compv_t& jet_rawcorr, const p4compv_t& jet_area, const p4compv_t& jet_muonSubtrFactor,
  const p4compv_t& jet_neEmEF, const p4compv_t& jet_chEmEF, const p4compv_int& jet_jetId,
  const float rho, const p4compv_int& jet_genJetIdx, const p4compv_int& jet_partonFlavour,
  const int seed, const int run,
  const p4compv_t& genjet_pt, const p4compv_t& genjet_eta, const p4compv_t& genjet_phi,
  const p4compv_t& genjet_mass, const float rawmet_phi, const float rawmet_pt, const p4compv_t& lowptjet_rawpt,
  const p4compv_t& lowptjet_eta, const p4compv_t& lowptjet_phi, const p4compv_t& lowptjet_area,
  const p4compv_t& lowptjet_muonSubtrFactor, const p4compv_t& lowptjet_neEmEF, const p4compv_t& lowptjet_chEmEF,
  const float met_unclustenupdx, const float met_unclustenupdy, const unsigned char npvGood
  ) const
{
  const auto nJets = jet_pt.size();
  const auto nVariations = 3+( m_doSmearing ? 2*( m_splitJER ? 6 : 1 ) : 0 )+2*m_jesUncSources.size()+( m_addHEM2018Issue ? 2 : 0 ); // 1(nom)+2(unclust)+2(JER[*6])+2*len(JES)[+2(HEM)]
  result_t out{nVariations, rawmet_pt*std::cos(rawmet_phi), rawmet_pt*std::sin(rawmet_phi)};
  LogDebug << "JME:: hello from Type1METVariations produce. Got " << jet_pt.size() << " jets and " << lowptjet_rawpt.size() << " low-PT jets" << std::endl;
  LogDebug << "JME:: Smearing (seed=" << seed << ")" << std::endl;
  LogDebug << "JME:: RawMET pt: " << rawmet_pt << ", phi: " << rawmet_phi << std::endl;

  // normal jets
  addVariations(out, jet_pt, jet_eta, jet_phi, jet_mass, jet_rawcorr, jet_area,
                jet_muonSubtrFactor, jet_neEmEF, jet_chEmEF, jet_jetId, rho,
                jet_genJetIdx, jet_partonFlavour, genjet_pt, genjet_eta,
                genjet_phi, genjet_mass, seed, run);

  //low-PT jets
  p4compv_t lowptjet_zero(lowptjet_rawpt.size(), 0.);
  p4compv_int lowptjet_zero_int(lowptjet_rawpt.size(), 0);
  p4compv_int lowptjet_mOne_int(lowptjet_rawpt.size(), -1);
  addVariations(out, lowptjet_rawpt, lowptjet_eta, lowptjet_phi, lowptjet_zero,
      lowptjet_zero, lowptjet_area, lowptjet_muonSubtrFactor,
      ( lowptjet_neEmEF.empty() ? lowptjet_zero : lowptjet_neEmEF  ),
      ( lowptjet_chEmEF.empty() ? lowptjet_zero : lowptjet_chEmEF  ),
      lowptjet_zero_int, rho, lowptjet_zero_int, lowptjet_mOne_int,
        genjet_pt, genjet_eta, genjet_phi, genjet_mass, seed, run);

  // unclustered energy, base on nominal(0)
  out.setXY(nVariations-2, out.px(0)+met_unclustenupdx, out.py(0)+met_unclustenupdy);
  out.setXY(nVariations-1, out.px(0)-met_unclustenupdx, out.py(0)-met_unclustenupdy);

  if(m_isXYcorrMET){
    applyXYcorrection(out, npvGood);
  }

#ifdef BAMBOO_JME_DEBUG
  LogDebug << "JME:: returning " << out.size() << " modified METs" << std::endl;
  const auto varNames = available();
  assert(varNames.size() == nVariations);
  for ( std::size_t i{0}; i != nVariations; ++i ) {
    LogDebug << "JME:: MET_" << varNames[i] << ": PT=" << out.pt(i) << ", PHI=" << out.phi(i) << std::endl;
  }
#endif
  return out;
}

void Type1METVariationsCalculator::applyXYcorrection(
  Type1METVariationsCalculator::result_t& out,
  const unsigned char npvGood
  ) const
{
  for ( std::size_t i{0}; i != out.size(); ++i ) {
    auto pt = out.pt(i);
    auto phi = out.phi(i);
    float phiXYCorr = m_metXYcorr->evaluate({"phi", "PuppiMET", m_eraForXYCorrMET, m_dataType, "nom", pt, phi, (double)npvGood});
    float ptXYCorr = m_metXYcorr->evaluate({"pt", "PuppiMET", m_eraForXYCorrMET, m_dataType, "nom", pt, phi, (double)npvGood});
    out.setXY(i, ptXYCorr*std::cos(phiXYCorr), ptXYCorr*std::sin(phiXYCorr));
  }
}

// for a single jet collection
void Type1METVariationsCalculator::addVariations(Type1METVariationsCalculator::result_t& out,
  const p4compv_t& jet_pt, const p4compv_t& jet_eta, const p4compv_t& jet_phi, const p4compv_t& jet_mass,
  const p4compv_t& jet_rawcorr, const p4compv_t& jet_area, const p4compv_t& jet_muonSubtrFactor,
  const p4compv_t& jet_neEmEF, const p4compv_t& jet_chEmEF, const p4compv_int& jet_jetId, const float rho,
  const p4compv_int& jet_genJetIdx, const p4compv_int& jet_partonFlavour, 
  const p4compv_t& genjet_pt, const p4compv_t& genjet_eta, const p4compv_t& genjet_phi, const p4compv_t& genjet_mass,
  const int seed, const int run
) const
{
  const auto nJets = jet_pt.size();
  for( std::size_t i{0}; i != nJets; ++i) {
    // L1 and full (L1L2L3Res) JEC for muon-subtracted jet
    float corr_L1L2L3Res = applyJEC(m_jesSF, m_jecInputAreaRho,
                                    m_jecInputRunAreaRho, m_jecInputAreaRhoRun,
                                    m_jecInputAreaRhoPhi, m_jecInputAreaRhoPhiRun,
                                    jet_area[i], jet_eta[i], jet_phi[i], jet_pt[i],
                                    rho, jet_rawcorr[i], run, m_isMC);
    float corr_L1 = 1;
    if ( m_doL1JEC ){
      if (m_l1JecInputAreaRho) {
        corr_L1 = m_jetLevel1->evaluate({jet_area[i], jet_eta[i], jet_pt[i]*(1.-jet_rawcorr[i]), rho});
      } else if (m_l1JecInputRunAreaRho) {
        corr_L1 = m_jetLevel1->evaluate({(double)run, jet_area[i], jet_eta[i], jet_pt[i]*(1.-jet_rawcorr[i]), rho});
      } else if (m_l1JecInputAreaRhoRun) {
        corr_L1 = m_jetLevel1->evaluate({jet_area[i], jet_eta[i], jet_pt[i]*(1.-jet_rawcorr[i]), rho, (double)run});
      } else {
        corr_L1 = m_jetLevel1->evaluate({jet_eta[i], jet_pt[i]*(1.-jet_rawcorr[i])});
      }
    }
    if (corr_L1L2L3Res <= 0.){corr_L1L2L3Res = 1.;}
    if (corr_L1 <= 0.){corr_L1 = 1.;}

    const auto jet_pt_raw = jet_pt[i]*(1-jet_rawcorr[i]);
    const double jet_pt_raw_nomu = jet_pt_raw*(1-jet_muonSubtrFactor[i]);
    const double muon_pt = jet_pt_raw*jet_muonSubtrFactor[i];
    const auto jet_pt_nomuL1L2L3 = jet_pt_raw_nomu*corr_L1L2L3Res;
    const auto jet_pt_nomuL1     = jet_pt_raw_nomu*corr_L1;
    const auto jet_pt_L1L2L3 = jet_pt_nomuL1L2L3 + muon_pt;
    const auto jet_pt_L1     = jet_pt_nomuL1     + muon_pt;
    const auto jet_mass_L1L2L3 = jet_mass[i]*(1-jet_rawcorr[i])*corr_L1L2L3Res;
    LogDebug << "JME:: jet_muonSubtrFactor[i]: " << jet_muonSubtrFactor[i] << std::endl; 
    LogDebug << "JME:: jecL1L2L3=" << corr_L1L2L3Res << ", jecL1=" << corr_L1 << "; PT_L1L2L3=" << jet_pt_L1L2L3 << ", PT_L1=" << jet_pt_L1 << ", PT_mu=" << muon_pt << std::endl;

    enum JERSFIndex {Nominal = 0, Up, Down};
    std::array<double, 3> smearFactor;
    if (m_doSmearing){
      smearFactor = applyJERSmearing(m_jetPtRes, m_jerSmear, m_jetEResSF,
                                     m_smearDoGenMatch,
                                     jet_pt_L1L2L3, jet_eta[i], jet_phi[i], jet_genJetIdx[i],
                                     genjet_pt, genjet_eta, genjet_phi, seed, rho);
    } else {
      smearFactor = {1., 1., 1.};
    }

    if ( ( jet_pt_nomuL1L2L3 > m_unclEnThreshold ) && ( (jet_neEmEF[i]+jet_chEmEF[i]) < m_emEnFracThreshold ) ) {
      std::size_t iVar = 0;
      const auto jet_cosPhi = std::cos(jet_phi[i]);
      const auto jet_sinPhi = std::sin(jet_phi[i]);
      if ( ! ( m_doSmearing && m_isT1SmearedMET) ) {
        out.addR_proj(iVar++, jet_cosPhi, jet_sinPhi, jet_pt_L1 - jet_pt_L1L2L3);             // nominal
      }
      auto jet_pt_L1p = jet_pt_L1; // with optional offset for JES uncertainty calculation if nominal is smeared
      if ( m_doSmearing ) {
        const auto dr_jernom = jet_pt_L1 - jet_pt_L1L2L3 * smearFactor[Nominal];
        if ( m_isT1SmearedMET ) {
          const auto dr_jerup   = jet_pt_L1 - jet_pt_L1L2L3*smearFactor[Up];
          const auto dr_jerdown = jet_pt_L1 - jet_pt_L1L2L3*smearFactor[Down];
          out.addR_proj(iVar++, jet_cosPhi, jet_sinPhi, dr_jernom);                         // smeared nominal
          if ( m_splitJER ) {
            const auto jerBin = jerSplitID(jet_pt_L1L2L3 * smearFactor[Nominal], jet_eta[i]);
            for ( int k{0}; k != 6; ++k ) {
              if ( jerBin == k ) { // vary
                out.addR_proj(iVar++, jet_cosPhi, jet_sinPhi, dr_jerup);                    // JER[k]-up
                out.addR_proj(iVar++, jet_cosPhi, jet_sinPhi, dr_jerdown);                  // JER[k]-down
              } else { // keep nominal
                out.addR_proj(iVar++, jet_cosPhi, jet_sinPhi, dr_jernom);                   // JER[k]-up
                out.addR_proj(iVar++, jet_cosPhi, jet_sinPhi, dr_jernom);                   // JER[k]-down
              }
            }
          }
          else{
            out.addR_proj(iVar++, jet_cosPhi, jet_sinPhi, dr_jerup);                        // JER-up
            out.addR_proj(iVar++, jet_cosPhi, jet_sinPhi, dr_jerdown);                      // JER-down
          }
          jet_pt_L1p += jet_pt_L1L2L3*(1.-smearFactor[Nominal]); // offset for JES uncertainties, since the nominal is smeared
        } else{
          for ( std::size_t k{0}; k != ( m_splitJER ? 6 : 1 ); ++k ) {
            out.addR_proj(iVar++, jet_cosPhi, jet_sinPhi, dr_jernom);                     // JER[k]-up
            out.addR_proj(iVar++, jet_cosPhi, jet_sinPhi, dr_jernom);                     // JER[k]-down
          }
        }
      }
      if ( m_addHEM2018Issue ) {
        const auto delta = deltaHEM2018Issue(jet_pt_L1L2L3 * smearFactor[Nominal], jet_jetId[i], jet_phi[i], jet_eta[i]);
        out.addR_proj(iVar++, jet_cosPhi, jet_sinPhi, jet_pt_L1p - jet_pt_L1L2L3);           // up = nominal
        out.addR_proj(iVar++, jet_cosPhi, jet_sinPhi, jet_pt_L1p - jet_pt_L1L2L3*delta);     // down
      }
      // JES uncertainties
      for ( auto& jesUnc : m_jesUncSources ) {
        LogDebug << "JME:: evaluating JES uncertainty: " << jesUnc.first << std::endl;
        float delta = 0.;
        const auto partonFlav = std::abs(jet_partonFlavour[i]);
        if (isValidFlavorJESUncertainty(jesUnc.first, partonFlav)) {
          delta = jesUnc.second->evaluate({jet_eta[i], jet_pt_L1L2L3});
        }
        LogDebug << " JES uncertainty " << jesUnc.first << " : " << delta << std::endl;
        out.addR_proj(iVar++, jet_cosPhi, jet_sinPhi, jet_pt_L1p - jet_pt_L1L2L3*(1+delta)); // JES_i-up
        out.addR_proj(iVar++, jet_cosPhi, jet_sinPhi, jet_pt_L1p - jet_pt_L1L2L3*(1-delta)); // JES_i-down
      }
    }
  }
}


std::vector<std::string> Type1METVariationsCalculator::available(const std::string&) const
{
 if ((!m_doJEC) || (!m_doL1JEC)) {
   throw std::runtime_error("The calculator is not fully configured (for MET variations both setJEC and setL1JEC need to be called)");
 }
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
 products.emplace_back("unclustEnup");
 products.emplace_back("unclustEndown");
 return products;
}
