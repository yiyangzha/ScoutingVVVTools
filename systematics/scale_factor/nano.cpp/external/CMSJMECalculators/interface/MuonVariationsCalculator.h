#ifndef CMSCALCULATORS_MUONVARIATIONSCALCULATOR_H
#define CMSCALCULATORS_MUONVARIATIONSCALCULATOR_H
#include "CMSCalculatorsUtilities.h"

#include <boost/math/special_functions/erf.hpp>
#include <cmath>
#include <iostream>

class MuonVariationsCalculator {
public:
  using p4compv_t = ROOT::VecOps::RVec<float>;
  using p4compv_int = ROOT::VecOps::RVec<int>;
  using result_t = rdfhelpers::ModifiedPtCollection;

  MuonVariationsCalculator() = default;
  
  void setCBParams(correction::Correction::Ref&& cb){
    m_cbParams = std::move(cb);
  }

  void setPolyParams(correction::Correction::Ref&& poly){
    m_polyParams = std::move(poly);
  }

  void setParamAForScaling(correction::Correction::Ref&& parA){
    m_aSF = std::move(parA);
  }

  void setParamMForScaling(correction::Correction::Ref&& parM){
    m_mSF = std::move(parM);
  }

  void setParamKData(correction::Correction::Ref&& kDataSF){
    m_kDataSF = std::move(kDataSF);
  }

  void setParamKMC(correction::Correction::Ref&& kMcSF){
    m_kMcSF = std::move(kMcSF);
  }

  void setSmearingTool(correction::Correction::Ref&& smearingTool) {
    m_smearingTool = std::move(smearingTool);
  }

  void setSystematics(bool addSystematics){m_Systematics = addSystematics;}
  void setIsMC(bool isMC){
    m_isMC = isMC;
  }

  static MuonVariationsCalculator create(
    const std::string& jsonFile,
    bool addSystematics, bool isMC,
    const std::string& jsonFileSmearingTool,
    const std::string& smearingTool);

  std::vector<std::string> available(const std::string& attr = {}) const;
  
  // Interface for NanoAOD
  result_t produce(
    const p4compv_t& muon_pt,
    const p4compv_t& muon_eta,
    const p4compv_t& muon_phi,
    const p4compv_int& muon_charge,
    const p4compv_int& muon_nLayers,
    const int seed
    ) const;

private:
  // Parameters and helpers
  correction::Correction::Ref m_aSF, m_mSF;
  correction::Correction::Ref m_cbParams, m_polyParams;
  correction::Correction::Ref m_kDataSF, m_kMcSF;
  correction::Correction::Ref m_smearingTool;

  std::string m_dtmc;
  bool m_Systematics{false}, m_isMC{false};

};

#endif
