#include "MuonVariationsCalculator.h"
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

  std::vector<double> getK(
    const correction::Correction::Ref kDataSF,
    const correction::Correction::Ref kMcSF,
    float eta, bool addSystematics) {

    auto compute_k = [](float kd, float km) {
      float val = (kd*kd) - (km*km);
      return (val > 0.0) ? std::sqrt(val) : 0.0;
    };

    float k_data = kDataSF->evaluate({std::abs(eta), "nom"});
    float k_mc = kMcSF->evaluate({std::abs(eta), "nom" });
    auto k = compute_k(k_data, k_mc);

    if (!addSystematics) return {k};

    float k_mc_stat = kMcSF->evaluate({std::abs(eta), "stat"});
    auto k_Up = compute_k(k_data, k_mc + k_mc_stat);
    auto k_Down = compute_k(k_data, k_mc - k_mc_stat);

    return {k, k_Up, k_Down};
  };

  double getScale(
    const correction::Correction::Ref aSF,
    const correction::Correction::Ref mSF,
    double pt, float eta, float phi, int charge){

    auto compute_scale = [&](float mVar, float aVar) {
      return (1.0/(mVar/pt + aVar*charge))/pt;
    };

    float a = aSF->evaluate({eta, phi, "nom"});
    float m = mSF->evaluate({eta, phi, "nom"});

    auto scale = compute_scale(m, a);

    return scale;
  }

  double getScaleUnc(
    const correction::Correction::Ref aSF,
    const correction::Correction::Ref mSF,
    double pt, float eta, float phi, int charge){

    float aStat = aSF->evaluate({eta, phi, "stat"});
    float mStat = mSF->evaluate({eta, phi, "stat"});
    float mRhoStat = mSF->evaluate({eta, phi, "rho_stat"});

    auto scaleUnc = pt*pt*sqrt(mStat*mStat/(pt*pt)+aStat*aStat+2*charge*mRhoStat*mStat*aStat/pt);

    return scaleUnc;
  };

  double getStd(
    const correction::Correction::Ref polyParams,
    double pt, float eta, int nL) {
    // obtain paramters from correctionlib
    float param_0 = polyParams->evaluate({std::abs(eta), (float)nL, 0});
    float param_1 = polyParams->evaluate({std::abs(eta), (float)nL, 1});
    float param_2 = polyParams->evaluate({std::abs(eta), (float)nL, 2});

    return std::max(0.0, param_0 + param_1 * pt + param_2 * pt*pt);
  };

  struct CrystalBall{
    double pi=3.14159;
    double sqrtPiOver2=sqrt(pi/2.0);
    double sqrt2=sqrt(2.0);
    double m, s, a, n;
    double B, C, D, N, NA, Ns, NC, F, G, k;
    double cdfMa, cdfPa;
    CrystalBall():m(0),s(1),a(10),n(10){
      init();
  }

  CrystalBall(double mean, double sigma, double alpha, double n)
    :m(mean),s(sigma),a(alpha),n(n){
    init();
  }

  void init(){
    double fa = std::abs(a);
    double ex = std::exp(-fa*fa/2);
    double A  = std::pow(n/fa, n) * ex;
    double C1 = n/fa/(n-1) * ex;
    double D1 = 2 * sqrtPiOver2 * std::erf(fa/sqrt2);
    B = n/fa-fa;
    C = (D1+2*C1)/C1;
    D = (D1+2*C1)/2;
    N = 1.0/s/(D1+2*C1);
    k = 1.0/(n-1);
    NA = N*A;
    Ns = N*s;
    NC = Ns*C1;
    F = 1-fa*fa/n;
    G = s*n/fa;
    cdfMa = cdf(m-a*s);
    cdfPa = cdf(m+a*s);
  }

  double pdf(double x) const{
    double d=(x-m)/s;
    if(d<-a) return NA*std::pow(B-d, -n);
    if(d>a) return NA*std::pow(B+d, -n);
    return N*exp(-d*d/2);
  }

  double pdf(double x, double ks, double dm) const{
    double d=(x-m-dm)/(s*ks);
    if(d<-a) return NA/ks*std::pow(B-d, -n);
    if(d>a) return NA/ks*std::pow(B+d, -n);
    return N/ks*std::exp(-d*d/2);
  }

  double cdf(double x) const{
    double d = (x-m)/s;
    if(d<-a) return NC / std::pow(F-s*d/G, n-1);
    if(d>a) return NC * (C - std::pow(F+s*d/G, 1-n) );
    return Ns * (D - sqrtPiOver2 * std::erf(-d/sqrt2));
  }

  double invcdf(double u) const{
    if(u<cdfMa) return m + G*(F - std::pow(NC/u, k));
    if(u>cdfPa) return m - G*(F - std::pow(C-u/NC, -k) );
    return m - sqrt2 * s * boost::math::erf_inv((D - u/Ns )/sqrtPiOver2);
  }
};

template<typename CALC>
  void configureMuonCalc(CALC& calc,
    const std::unique_ptr<CorrectionSet>& cset,
    const std::unique_ptr<CorrectionSet>& csetSmearingTool,
    bool isMC, const std::string& smearingTool)
  {
    calc.setParamKMC(std::move(cset->at("k_mc")));
    calc.setParamKData(std::move(cset->at("k_data")));
    calc.setCBParams(std::move(cset->at("cb_params")));
    calc.setPolyParams(std::move(cset->at("poly_params")));

    const std::string prefix = isMC ? "mc" : "data";
    calc.setParamAForScaling(std::move(cset->at("a_" + prefix)));
    calc.setParamMForScaling(std::move(cset->at("m_" + prefix)));

    if(!smearingTool.empty())
      calc.setSmearingTool(std::move(csetSmearingTool->at(smearingTool)));
  }
}


MuonVariationsCalculator MuonVariationsCalculator::create(
  const std::string& jsonFile,
  bool addSystematics, bool isMC,
    const std::string& jsonFileSmearingTool,
    const std::string& smearingTool)
{
  MuonVariationsCalculator inst{};
  auto cset = CorrectionSet::from_file(jsonFile);
  std::unique_ptr<CorrectionSet> csetSmearingTool = jsonFileSmearingTool.empty()
    ? nullptr : CorrectionSet::from_file(jsonFileSmearingTool);
  configureMuonCalc(inst, cset, csetSmearingTool, isMC, smearingTool);
  inst.setSystematics(addSystematics);
  inst.setIsMC(isMC);

  return std::move(inst);
}

MuonVariationsCalculator::result_t MuonVariationsCalculator::produce(
  const p4compv_t& muon_pt, const p4compv_t& muon_eta,
  const p4compv_t& muon_phi, const p4compv_int& muon_charge,
  const p4compv_int& muon_nLayers, const int seed )const
{
  const auto nVariations = 1 + (m_Systematics ? 4 : 0); // 1(nom)+2(up/down)x2(scale/smear)
  LogDebug << "CMSCalc:: hello from MuonVariations produce. Got " << muon_pt.size() << " Muon objects" << std::endl;
  const auto nMuons = muon_pt.size();

  enum VariationIndex {
    Nominal = 0,
    ScaleUp, ScaleDown,
    SmearUp, SmearDown
  };

  auto apply_scale = [](double value, double scale){
      auto scaled = value * scale;
      if (!std::isnan(scaled) && scaled >= 0. && scale >= 0.1 && scale <= 2.) {
          return scaled;
      }
      return value;
  };

  result_t out{nVariations, muon_pt};

  std::vector<ROOT::VecOps::RVec<double>> ptVariations(nVariations, muon_pt);

  for ( std::size_t i{0}; i != nMuons; ++i) {

    LogDebug << "CMSCalc:: Muon properties:  pT =  " << muon_pt[i] << ", eta: " << muon_eta[i] <<
                ", phi: " << muon_phi[i] << ", charge: " << muon_charge[i] <<
                ", nLayers: " << muon_nLayers[i] << std::endl;

    if (!m_isMC) { //data
      auto corrNom = getScale(m_aSF, m_mSF, muon_pt[i], muon_eta[i], muon_phi[i], muon_charge[i]);
      ptVariations[Nominal][i] = apply_scale(ptVariations[Nominal][i], corrNom);
    }
    else{
      auto scale = getScale(m_aSF, m_mSF, muon_pt[i], muon_eta[i], muon_phi[i], muon_charge[i]);
      double pt = apply_scale(muon_pt[i], scale);

      const float absEta = std::abs(muon_eta[i]);
      std::array<float, 4> cbParams = {
        m_cbParams->evaluate({absEta, static_cast<float>(muon_nLayers[i]), 0}), //mean
        m_cbParams->evaluate({absEta, static_cast<float>(muon_nLayers[i]), 1}), //sigma
        m_cbParams->evaluate({absEta, static_cast<float>(muon_nLayers[i]), 2}), //n
        m_cbParams->evaluate({absEta, static_cast<float>(muon_nLayers[i]), 3})  //alpha
      };

      // instantiate CB and get random number following the CB
      CrystalBall cb(cbParams[0], cbParams[1], cbParams[3], cbParams[2]);
      float rndm = cb.invcdf(m_smearingTool->evaluate({muon_pt[i], muon_eta[i], muon_phi[i], seed}));

      // Compute standard deviations
      auto stdDevs = getStd(m_polyParams, pt, muon_eta[i], muon_nLayers[i]);

      std::vector<double> k = getK(m_kDataSF, m_kMcSF, muon_eta[i], m_Systematics);
      auto corrNom = 1. + k[0] * stdDevs * rndm;

      ptVariations[Nominal][i] = apply_scale(ptVariations[Nominal][i], corrNom*scale);

      if ( m_Systematics ) {
        auto scaleUnc = getScaleUnc(m_aSF, m_mSF, pt, muon_eta[i], muon_phi[i], muon_charge[i]);
        auto stdDevsUp = getStd(m_polyParams, pt+scaleUnc, muon_eta[i], muon_nLayers[i]);
        auto stdDevsDown = getStd(m_polyParams, pt-scaleUnc, muon_eta[i], muon_nLayers[i]);

        std::array<double, 4> corrections = {
          1. + k[0] * stdDevsUp * rndm, // ScaleUp
          1. + k[0] * stdDevsDown * rndm, // ScaleDown
          1. + k[1] * stdDevs * rndm, // SmearUp
          1. + k[2] * stdDevs * rndm, // SmearDown
        };

        ptVariations[ScaleUp][i] = apply_scale((pt+scaleUnc), corrections[0]);
        ptVariations[ScaleDown][i] = apply_scale((pt-scaleUnc), corrections[1]);
        ptVariations[SmearUp][i] = apply_scale(ptVariations[SmearUp][i], corrections[2]*scale);
        ptVariations[SmearDown][i] = apply_scale(ptVariations[SmearDown][i], corrections[3]*scale);
      } 
    }
  }
  out.set(Nominal, ptVariations[Nominal]);

  if (m_Systematics) {
    for (int i = ScaleUp; i <= SmearDown; ++i) {
      out.set(i, ptVariations[i]);
    }
  }
  
  return out;
}


std::vector<std::string> MuonVariationsCalculator::available(const std::string&) const
{
  std::vector<std::string> products = { "nominal" };
  if(m_isMC && m_Systematics){
    static const std::vector<std::string> variations = {
      "mesScaleup", "mesScaledown", "mesSmearup", "mesSmeardown"
    };
    products.insert(products.end(), variations.begin(), variations.end());
  }
  return products;
}
