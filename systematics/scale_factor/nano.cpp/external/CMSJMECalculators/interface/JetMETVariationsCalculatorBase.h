#ifndef CMSCALCULATORS_JETMETVARIATIONSCALCULATORBASE_H
#define CMSCALCULATORS_JETMETVARIATIONSCALCULATORBASE_H
#include "JetMETVariationsCalculatorBase.h"
#include "CMSCalculatorsUtilities.h"

class JetMETVariationsCalculatorBase {
public:
  using p4compv_t = ROOT::VecOps::RVec<float>;
  using p4compv_int = ROOT::VecOps::RVec<int>;


  JetMETVariationsCalculatorBase() = default;

  // set up smearing (and JER systematics)
  void setSmearing(
    const correction::Correction::Ref&& jetPtRes,
    const correction::Correction::Ref&& jetEResSF,
    const correction::Correction::Ref&& jerSmear,
    const correction::Correction::Ref&& jetEResSFUnc,
    bool splitJER,
    bool doGenMatch,
    float genMatch_maxDR=-1.,
    float genMatch_maxDPT=-1.)
  {
    m_doSmearing = true;
    m_jetPtRes = std::move(jetPtRes);
    m_jetEResSF = std::move(jetEResSF);
    m_jerSmear = std::move(jerSmear);
    m_jetEResSFUnc = std::move(jetEResSFUnc);
    m_hasJetEResSFUnc = static_cast<bool>(m_jetEResSFUnc);
    m_splitJER = splitJER;
    m_smearDoGenMatch = doGenMatch;
    m_genMatch_dR2max = genMatch_maxDR*genMatch_maxDR;
    m_genMatch_dPtmax = genMatch_maxDPT;
    const auto sfInputs = jetEResSF->inputs();
    m_jersfInputEtaPtSyst = (sfInputs.size() == 3); // jetEta, jetPt, syst
    m_jersfInputEtaPt = (sfInputs.size() == 2 && sfInputs[1].type() == correction::Variable::VarType::real); // jetEta, jetPt
  }

  void setJEC(const std::variant<correction::Correction::Ref, correction::CompoundCorrection::Ref>&& jesSF) {
    m_doJEC = true;
    auto setInputFlags = [this](const auto& inputs) {
      const auto nInputs = inputs.size();
      const auto first = nInputs > 0 ? inputs.front().name() : std::string{};
      const auto last = nInputs > 0 ? inputs.back().name() : std::string{};
      m_jecInputAreaRho = (nInputs == 4);
      m_jecInputRunAreaRho = (nInputs == 5 && first == "run");
      m_jecInputAreaRhoRun = (nInputs == 5 && last == "run");
      m_jecInputAreaRhoPhi = (nInputs == 5 && last != "run" && first != "run");
      m_jecInputAreaRhoPhiRun = (nInputs == 6);
    };
    if (auto corrObj = std::get_if<correction::Correction::Ref>(&jesSF)) {
      setInputFlags((*corrObj)->inputs());
    } else {
      setInputFlags(std::get<correction::CompoundCorrection::Ref>(jesSF)->inputs());
    }
    m_jesSF = std::move(jesSF);
  }
  void setAddHEM2018Issue(bool enable) { m_addHEM2018Issue = enable; }

  void setIsMC(bool enable) {m_isMC = enable; }

  void addJESUncertainty(const std::string& name, const correction::Correction::Ref&& params)
  {
    m_jesUncSources.emplace(std::piecewise_construct,
        std::forward_as_tuple(name),
        std::forward_as_tuple(params));
  }
protected:
  std::size_t findGenMatch(
    const double pt, const float eta,
    const float phi, const std::size_t genJetIdx,
    const p4compv_t& gen_pt, const p4compv_t& gen_eta,
    const p4compv_t& gen_phi, const double resolution 
    ) const;

  int jerSplitID(
    const float pt, const float eta) const;

  bool isValidFlavorJESUncertainty(
    const std::string& jesName,
    const int partonFlav) const;

  float deltaHEM2018Issue(
    const float pt_nom, const int jetId,
    const float phi, const float eta ) const;

  float applyJEC(
    const std::variant<correction::Correction::Ref, correction::CompoundCorrection::Ref>& m_jesSF,
    const bool m_jecInputAreaRho, const bool m_jecInputRunAreaRho, const bool m_jecInputAreaRhoRun,
    const bool m_jecInputAreaRhoPhi, const bool m_jecInputAreaRhoPhiRun,
    const float jet_area, const float jet_eta, const float jet_phi, const float jet_pt, const float rho,
    const float jet_rawcorr, const int run, const bool m_isMC ) const;

  std::array<double, 3> applyJERSmearing(
    const correction::Correction::Ref& m_jetPtRes,
    const correction::Correction::Ref& m_jerSmear,
    const correction::Correction::Ref& m_jetEResSF,
    const bool m_smearDoGenMatch,
    const double pt_nom, const float jet_eta, const float jet_phi,
    const int jet_genJetIdx,
    const p4compv_t& genjet_pt, const p4compv_t& genjet_eta,
    const p4compv_t& genjet_phi, const int seed, const float rho) const;

  // config options
  bool m_doSmearing{false}, m_smearDoGenMatch;      // default: yes, yes
  bool m_addHEM2018Issue{false}, m_splitJER{false}; // default: no, no
  float m_genMatch_dR2max, m_genMatch_dPtmax;       // default: R/2 (0.2) and 3
  // parameters and helpers
  correction::Correction::Ref m_jetPtRes;
  correction::Correction::Ref m_jetEResSF;
  correction::Correction::Ref m_jetEResSFUnc;
  correction::Correction::Ref m_jerSmear;
  bool m_doJEC{false}, m_jecInputAreaRho{false}, m_jersfInputEtaPt{false}, m_jersfInputEtaPtSyst{false};
  bool m_hasJetEResSFUnc{false};
  bool m_jecInputRunAreaRho{false}, m_jecInputAreaRhoRun{false};
  bool m_jecInputAreaRhoPhi{false}, m_jecInputAreaRhoPhiRun{false};
  bool m_isMC{false};
  std::variant<correction::Correction::Ref, correction::CompoundCorrection::Ref> m_jesSF;
  std::map<std::string, correction::Correction::Ref> m_jesUncSources;
};

#endif
