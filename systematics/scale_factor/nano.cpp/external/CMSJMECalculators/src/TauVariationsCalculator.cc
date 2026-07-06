#include "TauVariationsCalculator.h"
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
#include <array>

using namespace correction;

namespace {
  template<typename CALC>
  void configureTauCalc(CALC& calc,
    const std::unique_ptr<CorrectionSet>& cset,
    const std::string& tauCorr)
  {
    calc.setTES(std::move(cset->at(tauCorr)));
  }

  inline bool isValidTauDecayMode(int mode) {
    return !(mode < 0 || (mode > 1 && mode < 10) || mode > 11);
  }
}

TauVariationsCalculator TauVariationsCalculator::create(
  const std::string& jsonFile,
  const std::string& tauCorr,
  const std::string& tauAlgo,
  const std::string& tauWP,
  const std::string& tauWPvsE,
  bool AddSystematics,
  bool splitSystematics)
{
  TauVariationsCalculator inst{};
  auto cset = CorrectionSet::from_file(jsonFile);
  configureTauCalc(inst, cset, tauCorr);
  inst.setTauAlgo(tauAlgo);
  if (tauWP.empty() && tauWPvsE.empty()) {
    inst.setRun3False();
  } else {
    inst.setTauWP(tauWP);
    inst.setTauWPvsE(tauWPvsE);
  }
  inst.setSystematics(AddSystematics);
  inst.setSplitSystematics(splitSystematics);
  return std::move(inst);
}

TauVariationsCalculator::result_t TauVariationsCalculator::produce(
  const p4compv_t& tau_pt, const p4compv_t& tau_eta,
  const p4compv_t& tau_mass, const p4compv_int& tau_decayMode,
  const p4compv_int& tau_genMatch ) const
{
  const auto nVariations = 1 + (m_Systematics ? 2 : 0) * (m_splitSystematics ? 2 : 1); // 1(nom)+2(up/down)*2(split syst)
  LogDebug << "CMSCalc:: hello from TauVariations produce. Got " << tau_pt.size() << " tau" << std::endl;
  const auto nTaus = tau_pt.size();
  result_t out{nVariations, tau_pt, tau_mass};

  std::vector<ROOT::VecOps::RVec<double>> pt_variations(nVariations, tau_pt);
  std::vector<ROOT::VecOps::RVec<double>> mass_variations(nVariations, tau_mass);

  enum VariationIndex { Nominal = 0, Up, Down };
  int endVariation = m_Systematics ? Down : Nominal;
  std::array<std::string, 3> variationTypes = {"nom", "up", "down"};

  for (std::size_t i{0}; i != nTaus; ++i) {
    if (!isValidTauDecayMode(tau_decayMode[i]))
      continue;
    for (int j = Nominal; j <= endVariation; ++j) {
      float corrections = 0.;
      if ( m_Run3 ){
        corrections = m_tesSF->evaluate({tau_pt[i], tau_eta[i], tau_decayMode[i], tau_genMatch[i], m_tauAlgo, m_wp, m_wp_VSE, variationTypes[j]});
      } else {
        corrections = m_tesSF->evaluate({tau_pt[i], tau_eta[i], tau_decayMode[i], tau_genMatch[i], m_tauAlgo, variationTypes[j]});
      }
      if (corrections > 0.) {
        if(j==0){
          pt_variations[j][i] *= corrections;
          mass_variations[j][i] *= corrections;
        }
        else{
          if(m_splitSystematics){
            if(tau_genMatch[i]==5){
              pt_variations[j][i] *= corrections;
              mass_variations[j][i] *= corrections;
              pt_variations[j+2][i] = pt_variations[Nominal][i];
              mass_variations[j+2][i] = mass_variations[Nominal][i];
            }
            else if(tau_genMatch[i]==1 || tau_genMatch[i]==3){
              pt_variations[j][i] = pt_variations[Nominal][i];
              mass_variations[j][i] = mass_variations[Nominal][i];
              pt_variations[j+2][i] *= corrections;
              mass_variations[j+2][i] *= corrections;
            }
            else{
              pt_variations[j][i] = pt_variations[Nominal][i];
              mass_variations[j][i] = mass_variations[Nominal][i];
              pt_variations[j+2][i] = pt_variations[Nominal][i];
              mass_variations[j+2][i] = mass_variations[Nominal][i];
            }
          }
          else{
            pt_variations[j][i] *= corrections;
            mass_variations[j][i] *= corrections;
          }
        }
      }
    }
  }

  for (int j = Nominal; j < nVariations; ++j) {
    out.set(j, pt_variations[j], mass_variations[j]);
  }

  return out;
}

std::vector<std::string> TauVariationsCalculator::available(const std::string&) const
{
  std::vector<std::string> products = { "nominal" };
  if(m_Systematics){
    if(m_splitSystematics){
      products.emplace_back("tesGenuineTauup");
      products.emplace_back("tesGenuineTaudown");
      products.emplace_back("tesEtoTauup");
      products.emplace_back("tesEtoTaudown");
    }
    else{
      products.emplace_back("tesup");
      products.emplace_back("tesdown");
    }
  }
  return products;
}
