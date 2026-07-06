#include "JetMETVariationsCalculatorBase.h"
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

using namespace correction;

namespace {
  // because something goes wrong with linking ROOT::Math::VectorUtil::Phi_mpi_pi
  template<typename T>
  T phi_mpi_pi(T angle) {
    if ( angle <= M_PI && angle > -M_PI ) {
      return angle;
    }
    if ( angle > 0 ) {
      const int n = static_cast<int>(.5*(angle*M_1_PI+1.));
      angle -= 2*n*M_PI;
    } else {
      const int n = static_cast<int>(-.5*(angle*M_1_PI-1.));
      angle += 2*n*M_PI;
    }
    return angle;
  }
}

bool JetMETVariationsCalculatorBase::isValidFlavorJESUncertainty(
  const std::string& jesName, const int partonFlav) const
{
  return !(jesName == "FlavorPureGluon" && partonFlav != 21) &&
         !(jesName == "FlavorPureQuark" && !(partonFlav >= 1 && partonFlav <= 3)) &&
         !(jesName == "FlavorPureCharm" && partonFlav != 4) &&
         !(jesName == "FlavorPureBottom" && partonFlav != 5);
}

float JetMETVariationsCalculatorBase::deltaHEM2018Issue(
  const float pt_nom, const int jetId,
  const float phi, const float eta ) const
{
  float delta = 1.;
  if ( pt_nom > 15. && ( jetId & 0x2 ) && phi > -1.57 && phi < -0.87 ) {
    if ( eta > -2.5 && eta < -1.3 ) {
      delta = 0.8;
    } else if ( eta <= -2.5 && eta > -3. ) {
      delta = 0.65;
    }
  }
  return delta;
}

int JetMETVariationsCalculatorBase::jerSplitID(
  const float pt, const float eta) const
{
  const auto aEta = std::abs(eta);
  if ( aEta < 1.93 )
    return 0;
  else if ( aEta < 2.5 )
    return 1;
  else if ( aEta < 3. )
    if ( pt < 50. )
      return 2;
    else
      return 3;
  else
    if ( pt < 50. )
      return 4;
    else
      return 5;
}

float JetMETVariationsCalculatorBase::applyJEC(
  const std::variant<correction::Correction::Ref, correction::CompoundCorrection::Ref>& m_jesSF,
  const bool m_jecInputAreaRho, const bool m_jecInputRunAreaRho, const bool m_jecInputAreaRhoRun,
  const bool m_jecInputAreaRhoPhi, const bool m_jecInputAreaRhoPhiRun,
  const float jet_area, const float jet_eta, const float jet_phi, const float jet_pt, const float rho,
  const float jet_rawcorr, const int run, const bool m_isMC) const
{
  static_cast<void>(m_isMC);
  float corr_L1L2L3Res;
  if (auto corrObj = std::get_if<Correction::Ref>(&m_jesSF)) {
    if (m_jecInputAreaRho)
      corr_L1L2L3Res = (*corrObj)->evaluate({jet_area, jet_eta, jet_pt*(1.-jet_rawcorr), rho});
    else if (m_jecInputRunAreaRho)
      corr_L1L2L3Res = (*corrObj)->evaluate({(double)run, jet_area, jet_eta, jet_pt*(1.-jet_rawcorr), rho});
    else if (m_jecInputAreaRhoRun)
      corr_L1L2L3Res = (*corrObj)->evaluate({jet_area, jet_eta, jet_pt*(1.-jet_rawcorr), rho, (double)run});
    else if (m_jecInputAreaRhoPhi)
      corr_L1L2L3Res = (*corrObj)->evaluate({jet_area, jet_eta, jet_pt*(1.-jet_rawcorr), rho, jet_phi});
    else if (m_jecInputAreaRhoPhiRun)
      corr_L1L2L3Res = (*corrObj)->evaluate({jet_area, jet_eta, jet_pt*(1.-jet_rawcorr), rho, jet_phi, (double)run});
    else
      corr_L1L2L3Res = (*corrObj)->evaluate({jet_eta, jet_pt*(1.-jet_rawcorr)});
  } else {
    if (m_jecInputAreaRho)
      corr_L1L2L3Res = std::get<CompoundCorrection::Ref>(m_jesSF)->evaluate({jet_area, jet_eta, jet_pt*(1.-jet_rawcorr), rho});
    else if (m_jecInputRunAreaRho)
      corr_L1L2L3Res = std::get<CompoundCorrection::Ref>(m_jesSF)->evaluate({(double)run, jet_area, jet_eta, jet_pt*(1.-jet_rawcorr), rho});
    else if (m_jecInputAreaRhoRun)
      corr_L1L2L3Res = std::get<CompoundCorrection::Ref>(m_jesSF)->evaluate({jet_area, jet_eta, jet_pt*(1.-jet_rawcorr), rho, (double)run});
    else if (m_jecInputAreaRhoPhi)
      corr_L1L2L3Res = std::get<CompoundCorrection::Ref>(m_jesSF)->evaluate({jet_area, jet_eta, jet_pt*(1.-jet_rawcorr), rho, jet_phi});
    else if (m_jecInputAreaRhoPhiRun)
      corr_L1L2L3Res = std::get<CompoundCorrection::Ref>(m_jesSF)->evaluate({jet_area, jet_eta, jet_pt*(1.-jet_rawcorr), rho, jet_phi, (double)run});
    else
      corr_L1L2L3Res = std::get<CompoundCorrection::Ref>(m_jesSF)->evaluate({jet_eta, jet_pt*(1.-jet_rawcorr)});
  }
  return corr_L1L2L3Res;
}

std::array<double, 3> JetMETVariationsCalculatorBase::applyJERSmearing(
  const correction::Correction::Ref& m_jetPtRes,
  const correction::Correction::Ref& m_jerSmear,
  const correction::Correction::Ref& m_jetEResSF,
  const bool m_smearDoGenMatch,
  const double pt_nom, const float jet_eta, const float jet_phi,
  const int jet_genJetIdx,
  const p4compv_t& genjet_pt, const p4compv_t& genjet_eta,
  const p4compv_t& genjet_phi,
  const int seed, const float rho) const
{
  double smearFactor_nom{1.}, smearFactor_down{1.}, smearFactor_up{1.};
  if (pt_nom < 0.){
    return {smearFactor_nom, smearFactor_up, smearFactor_down};
  }

  const auto ptRes  = m_jetPtRes->evaluate({jet_eta, pt_nom, rho});
  LogDebug << "JME:: JetParameters: pt=" << pt_nom << ", eta=" << jet_eta << ", rho=" << rho << "; ptRes=" << ptRes << std::endl;
  LogDebug << "JME:: ";
  float genPt = -1;
  if ( m_smearDoGenMatch ) {
    const auto iGen = findGenMatch(pt_nom, jet_eta, jet_phi, jet_genJetIdx, genjet_pt, genjet_eta, genjet_phi, ptRes*pt_nom);
    if ( iGen != genjet_pt.size() ) {
      genPt = genjet_pt[iGen];
      LogDebug << "genPt=" << genPt << " ";
    }
  }
  if ( m_jersfInputEtaPtSyst ) {
    smearFactor_nom  = m_jerSmear->evaluate({pt_nom, jet_eta, genPt, rho, seed, ptRes, m_jetEResSF->evaluate({jet_eta, pt_nom, "nom"})});
    smearFactor_up  = m_jerSmear->evaluate({pt_nom, jet_eta, genPt, rho, seed, ptRes, m_jetEResSF->evaluate({jet_eta, pt_nom, "up"})});
    smearFactor_down  = m_jerSmear->evaluate({pt_nom, jet_eta, genPt, rho, seed, ptRes, m_jetEResSF->evaluate({jet_eta, pt_nom, "down"})});
  } else if ( m_jersfInputEtaPt ) {
    const auto jerSF = m_jetEResSF->evaluate({jet_eta, pt_nom});
    const auto jerSFUnc = m_hasJetEResSFUnc ? m_jetEResSFUnc->evaluate({jet_eta, pt_nom}) : 0.;
    smearFactor_nom  = m_jerSmear->evaluate({pt_nom, jet_eta, genPt, rho, seed, ptRes, jerSF});
    smearFactor_up  = m_jerSmear->evaluate({pt_nom, jet_eta, genPt, rho, seed, ptRes, jerSF + jerSFUnc});
    smearFactor_down  = m_jerSmear->evaluate({pt_nom, jet_eta, genPt, rho, seed, ptRes, jerSF - jerSFUnc});
  } else {
    smearFactor_nom  = m_jerSmear->evaluate({pt_nom, jet_eta, genPt, rho, seed, ptRes, m_jetEResSF->evaluate({jet_eta, "nom"})});
    smearFactor_up  = m_jerSmear->evaluate({pt_nom, jet_eta, genPt, rho, seed, ptRes, m_jetEResSF->evaluate({jet_eta, "up"})});
    smearFactor_down  = m_jerSmear->evaluate({pt_nom, jet_eta, genPt, rho, seed, ptRes, m_jetEResSF->evaluate({jet_eta, "down"})});
  }
  LogDebug << "JME::  SmearFactors are: NOMINAL=" << smearFactor_nom << ", DOWN=" << smearFactor_down << ", UP=" << smearFactor_up << std::endl;

  return {smearFactor_nom, smearFactor_up, smearFactor_down};
}

// TODO with orig MET and jets (sumpx,sumpy): calc modif MET(sig), produce bigger results type
std::size_t JetMETVariationsCalculatorBase::findGenMatch(
  const double pt, const float eta, const float phi,
  const std::size_t genJetIdx, const p4compv_t& gen_pt,
  const p4compv_t& gen_eta, const p4compv_t& gen_phi,
  const double resolution ) const
{
  auto get_dr2 = [](float phi, float eta, float gen_phi, float gen_eta) -> float {
    const auto dphi = phi_mpi_pi(gen_phi - phi);
    const auto deta = gen_eta - eta;
    return dphi*dphi + deta*deta;
  };
  auto check_resolution = [this, resolution](float pt, float gen_pt) -> bool {
    return std::abs(gen_pt - pt) < m_genMatch_dPtmax*resolution;
  };

  // First check if matched genJet from NanoAOD is acceptable
  if (genJetIdx >= 0) {
      const float dr2 = get_dr2(phi, eta, gen_phi[genJetIdx], gen_eta[genJetIdx]);
      if ((dr2 < m_genMatch_dR2max) && check_resolution(pt, gen_pt[genJetIdx])) {
        LogDebug << "Using matched genJet from NanoAOD, dr2=" << dr2 << std::endl;
        return genJetIdx;
      }
  }

  std::size_t igBest{gen_pt.size()};
  auto dr2Min = std::numeric_limits<float>::max();
  LogDebug << "(DRs: ";
  for ( std::size_t ig{0}; ig != gen_pt.size(); ++ig ) {
    const auto dr2 = get_dr2(phi, eta, gen_phi[ig], gen_eta[ig]);
    LogDebug << "dr2=" << dr2;
    if ( ( dr2 < dr2Min ) && ( dr2 < m_genMatch_dR2max ) ) {
      LogDebug << "->dpt=" << std::abs(gen_pt[ig]-pt) << ",res=" << resolution;
      if (check_resolution(pt, gen_pt[ig])) {
        LogDebug << "->best:" << ig;
        dr2Min = dr2;
        igBest = ig;
      }
    }
    LogDebug << ", ";
  }
  LogDebug << ")";
  return igBest;
}
