#include "nano/producers/HeavyFlavScoutingMuonSampleProducer.h"

#include "nano/core/Collection.h"
#include "nano/core/Helpers.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nano {

namespace {

std::vector<ObjectView> sort_by_pt(std::vector<ObjectView> objects) {
  std::sort(objects.begin(), objects.end(), [](const auto &a, const auto &b) { return a.pt() > b.pt(); });
  return objects;
}

float safe_div(float num, float den) {
  return std::abs(den) > 1e-12f ? num / den : 0.0f;
}

bool pass_scouting_muon_id(const ObjectView &mu) {
  return safe_object_int(mu, "nValidStandAloneMuonHits", 0) > 0 &&
         safe_object_float(mu, "normchi2", 99.0f) < 3.0f &&
         safe_object_int(mu, "nTrackerLayersWithMeasurement", 0) > 5 &&
         safe_object_int(mu, "nValidPixelHits", 0) > 0 &&
         safe_object_int(mu, "nRecoMuonMatchedStations", 0) >= 2;
}

bool pass_scouting_ak4_id(const ObjectView &jet) {
  const auto abs_eta = std::abs(jet.eta());
  if (safe_object_int(jet, "nConstituents", 0) <= 1) {
    return false;
  }
  if (safe_object_float(jet, "neHEF", 1.0f) >= 0.99f || safe_object_float(jet, "neEmEF", 1.0f) >= 0.90f) {
    return false;
  }
  if (abs_eta <= 2.4f) {
    return safe_object_float(jet, "chHEF", 0.0f) > 0.01f &&
           safe_object_int(jet, "chHadMultiplicity", 0) > 0;
  }
  return true;
}

bool pass_scouting_ak8_id(const ObjectView &jet) {
  const auto abs_eta = std::abs(jet.eta());
  if (safe_object_int(jet, "nConstituents", 0) <= 1) {
    return false;
  }
  if (safe_object_float(jet, "neHEF", 1.0f) >= 0.99f || safe_object_float(jet, "neEmEF", 1.0f) >= 0.90f) {
    return false;
  }
  if (abs_eta <= 2.4f) {
    return safe_object_float(jet, "chHEF", 0.0f) > 0.01f &&
           safe_object_int(jet, "nCh", 0) > 0;
  }
  return true;
}

}  // namespace

HeavyFlavScoutingMuonSampleProducer::HeavyFlavScoutingMuonSampleProducer(ProducerConfig config)
    : HeavyFlavBaseProducer([&config] {
        config.channel = "scouting_muon";
        return config;
      }()) {
  fatjet_name_ = "ScoutingFatPFJetRecluster";
  genfatjet_name_ = "GenJet";
}

void HeavyFlavScoutingMuonSampleProducer::begin_file() {
  HeavyFlavBaseProducer::begin_file();
  out_.branch("passMuTrig", false);
  out_.branch("muon_pt", 0.0f);
  out_.branch("muon_eta", 0.0f);
  out_.branch("muon_miniIso", 0.0f);
  out_.branch("leptonicW_pt", 0.0f);
}

float HeavyFlavScoutingMuonSampleProducer::tagger_score(const ObjectView &fj, const std::string &name) const {
  const auto qcd = safe_object_float(fj, "scoutGlobalParT_prob_QCD", 0.0f);
  const auto xud = safe_object_float(fj, "scoutGlobalParT_prob_Xud", 0.0f);
  const auto xcs = safe_object_float(fj, "scoutGlobalParT_prob_Xcs", 0.0f);
  const auto xqq = safe_object_float(fj, "scoutGlobalParT_prob_Xqq", 0.0f);
  const auto xbb = safe_object_float(fj, "scoutGlobalParT_prob_Xbb", 0.0f);
  const auto xcc = safe_object_float(fj, "scoutGlobalParT_prob_Xcc", 0.0f);
  const auto xss = safe_object_float(fj, "scoutGlobalParT_prob_Xss", 0.0f);

  if (name == "WvsQCD") {
    const auto sig = xud + xcs;
    return safe_div(sig, sig + qcd);
  }
  if (name == "ZvsQCD") {
    const auto sig = xbb + xcc + xss + xqq;
    return safe_div(sig, sig + qcd);
  }
  if (name == "VvsQCD") {
    const auto sig = xud + xcs + xqq + xbb + xcc + xss;
    return safe_div(sig, sig + qcd);
  }
  if (name == "XcsVsQCD") {
    return safe_div(xcs, xcs + qcd);
  }
  if (name == "XudVsQCD") {
    return safe_div(xud, xud + qcd);
  }
  if (name == "XbbVsQCD") {
    return safe_div(xbb, xbb + qcd);
  }
  if (name == "XccVsQCD") {
    return safe_div(xcc, xcc + qcd);
  }
  if (name == "XssVsQCD") {
    return safe_div(xss, xss + qcd);
  }
  if (name == "XqqVsQCD") {
    return safe_div(xqq, xqq + qcd);
  }
  return safe_object_float(fj, name, -99.0f);
}

bool HeavyFlavScoutingMuonSampleProducer::analyze_common(Event &event) {
  auto muons = event.collection("ScoutingMuonVtx").objects();
  std::vector<ObjectView> selected_muons;
  for (auto &mu : muons) {
    const auto pt = mu.pt();
    const auto rel_iso = safe_div(safe_object_float(mu, "trackIso", 99.0f), std::max(pt, 1.0f));
    if (pt > 55.0f && std::abs(mu.eta()) < 2.4f && pass_scouting_muon_id(mu) && rel_iso < 0.15f) {
      mu.set("mass", safe_object_float(mu, "m", 0.105f));
      selected_muons.push_back(mu);
    }
  }
  if (selected_muons.size() != 1U) {
    return false;
  }
  selected_muons = sort_by_pt(std::move(selected_muons));
  event.set("muons", selected_muons);
  event.set("looseLeptons", selected_muons);

  auto fatjets = event.collection(fatjet_name_).objects();
  for (auto &fj : fatjets) {
    fj.set("idx", static_cast<std::int32_t>(fj.index()));
    fj.set("rawPt", fj.pt());
    fj.set("msoftdrop_uncorrected", safe_object_float(fj, "msoftdrop", 0.0f));
    fj.set("is_qualified", true);
    fj.set("subjets", std::vector<ObjectView>{});
    for (const auto &tagger : config_.tagger_names) {
      fj.set(tagger, tagger_score(fj, tagger));
    }
  }
  load_gen_history(event, fatjets);
  event.set("scoutingFatjetsCommon", fatjets);
  return true;
}

JmeEventResult HeavyFlavScoutingMuonSampleProducer::compute_jme_result(Event &event) const {
  (void)event;
  return JmeEventResult{};
}

bool HeavyFlavScoutingMuonSampleProducer::analyze_variation(Event &event, const JmeEventResult &jme_result,
                                                            JmeVariation variation) {
  (void)jme_result;
  if (variation != JmeVariation::Nominal) {
    throw std::runtime_error("scouting_muon currently supports only the nominal variation");
  }

  const auto met_pt = event.scalar<float>("ScoutingMET_pt");
  const auto met_phi = event.scalar<float>("ScoutingMET_phi");
  if (met_pt < 50.0f) {
    return false;
  }
  event.set("met_pt", met_pt);
  event.set("met_phi", met_phi);
  event.set("jetVetoFlag", std::int32_t{-99});

  auto mu = event.get<std::vector<ObjectView>>("muons").front();
  event.set("mu", mu);
  const auto leptonic_w = polar_p4(mu) + met_p4(met_pt, met_phi);
  event.set("leptonicW", leptonic_w);
  if (leptonic_w.Pt() < 100.0f) {
    return false;
  }

  auto ak4jets = sort_by_pt(event.collection("ScoutingPFJetRecluster2").objects());
  std::vector<ObjectView> clean_ak4jets;
  clean_ak4jets.reserve(ak4jets.size());
  for (auto &jet : ak4jets) {
    jet.set("rawPt", jet.pt() * (1.0f - safe_object_float(jet, "rawFactor", 0.0f)));
    if (jet.pt() > 25.0f && std::abs(jet.eta()) < 2.4f && pass_scouting_ak4_id(jet)) {
      clean_ak4jets.push_back(jet);
    }
  }

  float ht = 0.0f;
  for (const auto &jet : clean_ak4jets) {
    ht += jet.pt();
  }
  event.set("ak4jets", clean_ak4jets);
  event.set("ht", ht);

  std::vector<ObjectView> bjets;
  for (auto &jet : clean_ak4jets) {
    const auto btag_value = jet.get<float>(config_.btag_config.branch);
    if (btag_value > config_.btag_config.medium && std::abs(delta_phi(jet, mu)) < 2.0f) {
      bjets.push_back(jet);
    }
  }
  if (bjets.empty()) {
    return false;
  }
  event.set("bjets", bjets);

  auto fatjets = event.get<std::vector<ObjectView>>("scoutingFatjetsCommon");
  fatjets = sort_by_pt(std::move(fatjets));
  std::vector<ObjectView> probe_jets;
  for (auto &fj : fatjets) {
    if (fj.pt() > 200.0f && std::abs(fj.eta()) < 2.4f && pass_scouting_ak8_id(fj) &&
        std::abs(delta_phi(fj, mu)) > 2.0f) {
      probe_jets.push_back(fj);
    }
  }
  if (probe_jets.empty()) {
    return false;
  }
  probe_jets.erase(probe_jets.begin() + 1, probe_jets.end());
  event.set("fatjets", probe_jets);

  fill_base_event_info(event, variation);
  fill_fatjet_info(event, probe_jets);

  const auto rel_iso = safe_div(safe_object_float(mu, "trackIso", 99.0f), std::max(mu.pt(), 1.0f));
  out_.fill("passMuTrig", pass_trigger(event, config_.required_triggers));
  out_.fill("muon_pt", mu.pt());
  out_.fill("muon_eta", mu.eta());
  out_.fill("muon_miniIso", rel_iso);
  out_.fill("leptonicW_pt", static_cast<float>(leptonic_w.Pt()));
  return true;
}

bool HeavyFlavScoutingMuonSampleProducer::analyze(Event &event) {
  if (!analyze_common(event)) {
    return false;
  }
  const auto jme_result = compute_jme_result(event);
  return analyze_variation(event, jme_result, JmeVariation::Nominal);
}

}  // namespace nano
