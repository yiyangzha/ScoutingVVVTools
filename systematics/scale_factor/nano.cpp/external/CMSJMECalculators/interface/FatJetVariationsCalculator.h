#ifndef CMSCALCULATORS_FATJETVARIATIONSCALCULATOR_H
#define CMSCALCULATORS_FATJETVARIATIONSCALCULATOR_H
#include "JetMETVariationsCalculatorBase.h"
#include "CMSCalculatorsUtilities.h"

class FatJetVariationsCalculator : public JetMETVariationsCalculatorBase {
public:
  using result_t = rdfhelpers::ModifiedPtMMsdCollection;

  FatJetVariationsCalculator() = default;

  static FatJetVariationsCalculator create(
    const std::string& jsonFile,
    const std::string& jetAlgo,
    const std::string& jecTag,
    const std::string& jecLevel,
    const std::vector<std::string>& jesUncertainties,
    bool addHEM2018Issue,
    const std::string& jerTag, const std::string& jsonFileSmearingTool,
    const std::string& smearingToolName, bool splitJER,
    bool doGenMatch, float genMatch_maxDR, float genMatch_maxDPT,
    const std::string& jsonFileSubjet,
    const std::string& jetAlgoSubjet,
    const std::string& jecTagSubjet,
    const std::string& jecLevelSubjet);

  std::vector<std::string> available(const std::string& attr = {}) const;
  // interface for NanoAOD
  result_t produce(
    const p4compv_t& jet_pt, const p4compv_t& jet_eta, const p4compv_t& jet_phi, const p4compv_t& jet_mass,
    const p4compv_t& jet_rawcorr, const p4compv_t& jet_area, const p4compv_t& jet_msoftdrop,
    const p4compv_int& jet_subJetIdx1, const p4compv_int& jet_subJetIdx2, const p4compv_t& subjet_pt,
    const p4compv_t& subjet_eta, const p4compv_t& subjet_phi, const p4compv_t& subjet_mass, const p4compv_t& subjet_rawFactor,
    const p4compv_int& jet_jetId, const float rho,
    // MC-only
    const p4compv_int& jet_genJetIdx, const int seed, const int run,
    const p4compv_t& genjet_pt, const p4compv_t& genjet_eta, const p4compv_t& genjet_phi, const p4compv_t& genjet_mass
    ) const;

  void addJESUncertaintySubjet(const std::string& name, const correction::Correction::Ref&& params)
  {
    m_jesUncSourcesSubjet.emplace(std::piecewise_construct,
      std::forward_as_tuple(name), std::forward_as_tuple(params));
  }
  void setJECSubjet(const std::variant<correction::Correction::Ref, correction::CompoundCorrection::Ref>&& jesSF) {
    m_doJECSubjet = true;
    auto setInputFlags = [this](const auto& inputs) {
      const auto nInputs = inputs.size();
      const auto first = nInputs > 0 ? inputs.front().name() : std::string{};
      const auto last = nInputs > 0 ? inputs.back().name() : std::string{};
      m_jecInputAreaRhoSubjet = (nInputs == 4);
      m_jecInputRunAreaRhoSubjet = (nInputs == 5 && first == "run");
      m_jecInputAreaRhoRunSubjet = (nInputs == 5 && last == "run");
      m_jecInputAreaRhoPhiSubjet = (nInputs == 5 && last != "run" && first != "run");
      m_jecInputAreaRhoPhiRunSubjet = (nInputs == 6);
    };
    if (auto corrObj = std::get_if<correction::Correction::Ref>(&jesSF)) {
      setInputFlags((*corrObj)->inputs());
    } else {
      setInputFlags(std::get<correction::CompoundCorrection::Ref>(jesSF)->inputs());
    }
    m_jesSFSubjet = std::move(jesSF);
  }

protected:
  std::variant<correction::Correction::Ref, correction::CompoundCorrection::Ref> m_jesSFSubjet;
  std::map<std::string, correction::Correction::Ref> m_jesUncSourcesSubjet;
  bool m_jecInputAreaRhoSubjet{false}, m_jecInputRunAreaRhoSubjet{false}, m_jecInputAreaRhoRunSubjet{false};
  bool m_jecInputAreaRhoPhiSubjet{false}, m_jecInputAreaRhoPhiRunSubjet{false}, m_doJECSubjet{false};
};

#endif
