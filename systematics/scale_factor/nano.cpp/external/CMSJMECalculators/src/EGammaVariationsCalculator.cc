#include "EGammaVariationsCalculator.h"
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
  void configureEGammaCalc(CALC& calc,
    const std::unique_ptr<CorrectionSet>& cset,
    const std::unique_ptr<CorrectionSet>& csetSmearingTool,
    const std::string& scale,
    const std::string& smearing,
    const std::string& smearingTool)
  {
    if( !scale.empty() )
      calc.setEGammaScale(std::move(cset->compound().at(scale)));
    if( !smearing.empty() )
      calc.setEGammaSmearing(std::move(cset->at(smearing)));
    if( !smearingTool.empty() )
      calc.setEGammaSmearingTool(std::move(csetSmearingTool->at(smearingTool)));
  }
}

EGammaVariationsCalculator EGammaVariationsCalculator::create(
  const std::string& jsonFile,
  const std::string& scale,
  const std::string& smearing,
  bool isMC, bool addSystematics,
  const std::string& jsonFileSmearingTool,
  const std::string& smearingTool)
{
  EGammaVariationsCalculator inst{};
  auto cset = CorrectionSet::from_file(jsonFile);
  std::unique_ptr<CorrectionSet> csetSmearingTool = jsonFileSmearingTool.empty()
    ? nullptr : CorrectionSet::from_file(jsonFileSmearingTool);
  configureEGammaCalc(inst, cset, csetSmearingTool, scale, smearing, smearingTool);
  inst.setSystematics(addSystematics);
  inst.setIsMC(isMC);
  return std::move(inst);
}

EGammaVariationsCalculator::result_t EGammaVariationsCalculator::produce(
  const p4compv_t& egamma_pt, const p4compv_t& egamma_eta, const p4compv_t& egamma_deltaSC,
  const p4compv_t& egamma_phi, const p4compv_t& egamma_r9, const p4compv_int& egamma_seedGain,
  const int run, const int seed) const
{
  const auto nVariations = 1 + (m_Systematics ? 4 : 0); // 1(nom)+2(up/down for scale and smearing)
  LogDebug << "CMSCalc:: hello from EGammaVariations produce. Got " << egamma_pt.size() << " EGamma objects" << std::endl;
  const auto nEGammas = egamma_pt.size();
  result_t out{nVariations, egamma_pt};

  enum VariationIndex {Nominal = 0, ScaleDown, ScaleUp, SmearUp, SmearDown};
  std::vector<ROOT::VecOps::RVec<double>> pt_variations(nVariations, egamma_pt);

  auto apply_correction = [&](auto& pt_var, auto correction){
    if( correction>0. ){
      pt_var *= correction;
    }
  };

  for ( std::size_t i{0}; i < nEGammas; ++i ) {
    auto etaSC = egamma_eta[i] + egamma_deltaSC[i];
    float corrNom = 0., smearing = 0., rho = 0.;
    if ( !m_isMC ) {
      corrNom = std::get<CompoundCorrection::Ref>(m_egammaScale)->evaluate({"scale", (double)run, etaSC, egamma_r9[i], egamma_pt[i], (double)egamma_seedGain[i]});
    } else {
      rho = m_egammaSmearing->evaluate({"smear", egamma_pt[i], egamma_r9[i], etaSC});
      smearing = m_egammaSmearingTool->evaluate({egamma_pt[i], etaSC, egamma_phi[i], seed});
      corrNom = 1. + rho * smearing;
      
      if( m_Systematics ){
        float corrVar = m_egammaSmearing->evaluate({"escale", egamma_pt[i], egamma_r9[i], etaSC});
        float rhoErr = m_egammaSmearing->evaluate({"esmear", egamma_pt[i], egamma_r9[i], etaSC});
        auto corrSmearingUp = 1. + (rho + rhoErr) * smearing;
        auto corrSmearingDown = 1. + (rho - rhoErr) * smearing;
        auto corrNom = 1. + rho * smearing;
        auto corrScaleUp = (1. + corrVar) * corrNom;
        auto corrScaleDown = (1. - corrVar) * corrNom;
        
        apply_correction(pt_variations[ScaleUp][i], corrScaleUp);
        apply_correction(pt_variations[ScaleDown][i], corrScaleDown);
        apply_correction(pt_variations[SmearUp][i], corrSmearingUp);
        apply_correction(pt_variations[SmearDown][i], corrSmearingDown);
      }
    }
    apply_correction(pt_variations[Nominal][i], corrNom);
  }

  out.set(Nominal, pt_variations[Nominal]);

  if ( m_Systematics ) {
    for ( int v = ScaleDown; v <= SmearDown; ++v ) {
      out.set(v, pt_variations[v]);
    }
  }
  return out;
}

std::vector<std::string> EGammaVariationsCalculator::available(const std::string&) const
{
  std::vector<std::string> products = { "nominal" };
  if( m_isMC && m_Systematics ){
    constexpr std::array<std::string_view, 4> systematics = {
      "egammascaleup", "egammascaledown", 
      "egammasmearingup", "egammasmearingdown"
    };
    products.insert(products.end(), systematics.begin(), systematics.end());  
  }
  return products;
}
