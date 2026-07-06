#ifndef CMSCALCULATORS_JETVARIATIONSCALCULATOR_H
#define CMSCALCULATORS_JETVARIATIONSCALCULATOR_H
#include "JetMETVariationsCalculatorBase.h"
#include "CMSCalculatorsUtilities.h"

class JetVariationsCalculator : public JetMETVariationsCalculatorBase {
public:
  using result_t = rdfhelpers::ModifiedPtMCollection;

  JetVariationsCalculator() = default;

  static JetVariationsCalculator create(
    const std::string& jsonFile,
    const std::string& jetAlgo,
    const std::string& jecTag,
    const std::string& jecLevel,
    const std::vector<std::string>& jesUncertainties,
    bool addHEM2018Issue,
    const std::string& jerTag, const std::string& jsonFileSmearingTool,
    const std::string& smearingToolName, bool splitJER,
    bool doGenMatch, float genMatch_maxDR, float genMatch_maxDPT);

  std::vector<std::string> available(const std::string& attr = {}) const;
  // interface for NanoAOD
  result_t produce(
    const p4compv_t& jet_pt, const p4compv_t& jet_eta, const p4compv_t& jet_phi,
    const p4compv_t& jet_mass, const p4compv_t& jet_rawcorr, const p4compv_t& jet_area,
    const p4compv_int& jet_jetId, const float rho,
    // MC-only
    const p4compv_int& jet_genJetIdx, const p4compv_int& jet_partonFlavour,
    const int seed, const int run,
    const p4compv_t& genjet_pt, const p4compv_t& genjet_eta,
    const p4compv_t& genjet_phi, const p4compv_t& genjet_mass) const;
};

#endif
