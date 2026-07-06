#ifndef CMSCALCULATORS_EGAMMAVARIATIONSCALCULATOR_H
#define CMSCALCULATORS_EGAMMAVARIATIONSCALCULATOR_H
#include "CMSCalculatorsUtilities.h"

class EGammaVariationsCalculator {
public:
  using p4compv_t = ROOT::VecOps::RVec<float>;
  using p4compv_int = ROOT::VecOps::RVec<int>;
  using result_t = rdfhelpers::ModifiedPtCollection;

  EGammaVariationsCalculator() = default;

  void setEGammaScale(const std::variant<correction::Correction::Ref, correction::CompoundCorrection::Ref>&& scale) {
    m_egammaScale = std::move(scale);
  }

  void setEGammaSmearing(correction::Correction::Ref&& smearing) {
    m_egammaSmearing = std::move(smearing);
  }

  void setEGammaSmearingTool(correction::Correction::Ref&& smearingTool) {
    m_egammaSmearingTool = std::move(smearingTool);
  }

  void setSystematics(bool addSystematics){m_Systematics = addSystematics;}
  void setIsMC(bool isMC){m_isMC = isMC;}

  static EGammaVariationsCalculator create(
    const std::string& jsonFile,
    const std::string& scale,
    const std::string& smearing,
    bool isMC,
    bool addSystematics,
    const std::string& jsonFileSmearingTool,
    const std::string& smearingTool);

  std::vector<std::string> available(const std::string& attr = {}) const;
  // interface for NanoAOD
  result_t produce(
    const p4compv_t& egamma_pt,
    const p4compv_t& egamma_eta,
    const p4compv_t& egamma_deltaSC,
    const p4compv_t& egamma_phi,
    const p4compv_t& egamma_r9,
    const p4compv_int& egamma_seedGain,
    const int run,
    const int seed
    ) const;

private:
  // Parameters and helpers
  std::variant<correction::Correction::Ref, correction::CompoundCorrection::Ref> m_egammaScale;
  correction::Correction::Ref m_egammaSmearing;
  correction::Correction::Ref m_egammaSmearingTool;
  bool m_Systematics{false};
  bool m_isMC{false};
};

#endif
