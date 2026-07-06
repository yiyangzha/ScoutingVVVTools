#ifndef CMSCALCULATORS_TAUVARIATIONSCALCULATOR_H
#define CMSCALCULATORS_TAUVARIATIONSCALCULATOR_H
#include "CMSCalculatorsUtilities.h"

class TauVariationsCalculator {
public:
  using p4compv_t = ROOT::VecOps::RVec<float>;
  using p4compv_int = ROOT::VecOps::RVec<int>;
  using result_t = rdfhelpers::ModifiedPtMCollection;

  TauVariationsCalculator() = default;

  void setTES(correction::Correction::Ref&& tes) {
    m_tesSF = std::move(tes);
  }

  void setTauAlgo(const std::string& name) {
    m_tauAlgo = name;
  }

  void setTauWP(const std::string& name) {
    m_wp = name;
  }

  void setTauWPvsE(const std::string& name){
    m_wp_VSE = name;
  }

  void setSystematics(bool addSystematics) {
    m_Systematics = addSystematics;
  }

  void setSplitSystematics(bool splitSystematics){
    m_splitSystematics = splitSystematics;
  }

  void setRun3False() {
    m_Run3 = false;
  }

  static TauVariationsCalculator create(
    const std::string& jsonFile,
    const std::string& tauCorr,
    const std::string& tauAlgo,
    const std::string& tauWP,
    const std::string& tauWPvsE,
    bool addSystematics,
    bool splitSystematics);

  std::vector<std::string> available(const std::string& attr = {}) const;

  // Interface for NanoAOD
  result_t produce(
    const p4compv_t& tau_pt,
    const p4compv_t& tau_eta,
    const p4compv_t& tau_mass,
    const p4compv_int& tau_decayMode,
    const p4compv_int& tau_genMatch
    ) const;

private:
  // Parameters and helpers
  correction::Correction::Ref m_tesSF;
  std::string m_tauAlgo, m_wp, m_wp_VSE;
  bool m_Systematics{false};
  bool m_splitSystematics{false};
  bool m_Run3{true};
};

#endif
