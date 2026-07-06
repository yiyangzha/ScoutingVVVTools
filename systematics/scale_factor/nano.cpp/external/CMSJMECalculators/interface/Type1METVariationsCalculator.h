#ifndef CMSCALCULATORS_TYPE1METVARIATIONSCALCULATOR_H
#define CMSCALCULATORS_TYPE1METVARIATIONSCALCULATOR_H
#include "JetMETVariationsCalculatorBase.h"
#include "CMSCalculatorsUtilities.h"

class Type1METVariationsCalculator : public JetMETVariationsCalculatorBase {
public:
  using result_t = rdfhelpers::ModifiedMET;

  Type1METVariationsCalculator() = default;

  static Type1METVariationsCalculator create(
    const std::string& jsonFile,
    const std::string& jetAlgo,
    const std::string& jecTag,
    const std::string& jecLevel,
    const std::string& l1JecTag,
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
    bool doGenMatch, float genMatch_maxDR, float genMatch_maxDPT);

  // additional settings: L1-only JEC
  void setUnclusteredEnergyTreshold(float threshold) { m_unclEnThreshold = threshold; }
  void setEmEnergyFracThreshold(float EMthreshold) {m_emEnFracThreshold = EMthreshold; }
  void setIsT1SmearedMET(bool isT1SmearedMET) { m_isT1SmearedMET = isT1SmearedMET; }
  void setL1JEC(const correction::Correction::Ref&& l1JecTag) {
    m_doL1JEC = true;
    const auto inputs = l1JecTag->inputs();
    const auto nInputs = inputs.size();
    const auto first = nInputs > 0 ? inputs.front().name() : std::string{};
    const auto last = nInputs > 0 ? inputs.back().name() : std::string{};
    m_l1JecInputAreaRho = (nInputs == 4);
    m_l1JecInputRunAreaRho = (nInputs == 5 && first == "run");
    m_l1JecInputAreaRhoRun = (nInputs == 5 && last == "run");
    m_jetLevel1 = std::move(l1JecTag);
  }
  void setIsXYcorrMET(bool isXYCorrMET) { m_isXYcorrMET = isXYCorrMET; }
  void setXYCorrMET(const correction::Correction::Ref&& xyCorrMET){
    m_metXYcorr = std::move(xyCorrMET);
  }
  void setEraXYCorrMET(const std::string eraForXYCorrMET){
    m_eraForXYCorrMET = eraForXYCorrMET;
  }
  void setIsMCXYCorrMET(bool isMC){
    if (isMC){
      m_dataType = "MC";
    } else {
      m_dataType = "DATA";
    }
  }

  std::vector<std::string> available(const std::string& attr = {}) const;
  // interface for NanoAOD
  result_t produce(
    const p4compv_t& jet_pt, const p4compv_t& jet_eta, const p4compv_t& jet_phi, const p4compv_t& jet_mass,
    const p4compv_t& jet_rawcorr, const p4compv_t& jet_area, const p4compv_t& jet_muonSubtrFactor, 
    const p4compv_t& jet_neEmEF, const p4compv_t& jet_chEmEF, const p4compv_int& jet_jetId, const float rho,
    // MC-only
    const p4compv_int& jet_genJetIdx, const p4compv_int& jet_partonFlavour, const int seed, const int run,
    const p4compv_t& genjet_pt, const p4compv_t& genjet_eta, const p4compv_t& genjet_phi, const p4compv_t& genjet_mass,
    // MET-specific
    const float rawmet_phi, const float rawmet_pt, const p4compv_t& lowptjet_rawpt, const p4compv_t& lowptjet_eta,
    const p4compv_t& lowptjet_phi, const p4compv_t& lowptjet_area, const p4compv_t& lowptjet_muonSubtrFactor,
    const p4compv_t& lowptjet_neEmEF, const p4compv_t& lowptjet_chEmEF,
    const float met_unclustenupdx, const float met_unclustenupdy, const unsigned char npvGood
    ) const;

protected:
  float m_unclEnThreshold = 15.;
  float m_emEnFracThreshold = 0.9;
  bool m_isT1SmearedMET = false;
  bool m_doL1JEC{false};
  bool m_l1JecInputAreaRho{false};
  bool m_l1JecInputRunAreaRho{false};
  bool m_l1JecInputAreaRhoRun{false};
  correction::Correction::Ref m_jetLevel1;
  bool m_isXYcorrMET = false;
  correction::Correction::Ref m_metXYcorr;
  std::string m_eraForXYCorrMET, m_dataType;
  void addVariations(Type1METVariationsCalculator::result_t& out,
    const p4compv_t& jet_pt, const p4compv_t& jet_eta, const p4compv_t& jet_phi, const p4compv_t& jet_mass,
    const p4compv_t& jet_rawcorr, const p4compv_t& jet_area, const p4compv_t& jet_muonSubtrFactor,
    const p4compv_t& jet_neEmEF, const p4compv_t& jet_chEmEF, const p4compv_int& jet_jetId, const float rho,
    const p4compv_int& jet_genJetIdx, const p4compv_int& jet_partonFlavour, const p4compv_t& genjet_pt,
    const p4compv_t& genjet_eta, const p4compv_t& genjet_phi, const p4compv_t& genjet_mass,
    const int seed, const int run
    ) const;

  void applyXYcorrection(Type1METVariationsCalculator::result_t& out,
    const unsigned char npvGood
    ) const;
};

#endif
