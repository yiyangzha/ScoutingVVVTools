#include "nano/helpers/PuWeightProducer.h"

#include <correction.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace nano {

namespace {

using correction::CorrectionSet;

float clip(float value, float low, float high) {
  return std::max(low, std::min(high, value));
}

struct CachedCorrectionSet {
  std::unique_ptr<CorrectionSet> correction_set;
};

CachedCorrectionSet &cache_for(const std::string &json_file) {
  static std::unordered_map<std::string, CachedCorrectionSet> cache;
  auto &entry = cache[json_file];
  if (!entry.correction_set) {
    entry.correction_set = CorrectionSet::from_file(json_file);
  }
  return entry;
}

}  // namespace

PuWeightProducer::PuWeightProducer(const ProducerConfig &config) : config_(config) {}

void PuWeightProducer::begin_file(OutputModel &out) const {
  out.branch("puWeight", 1.0f);
  out.branch("puWeightUp", 1.0f);
  out.branch("puWeightDown", 1.0f);
}

void PuWeightProducer::fill(Event &event, OutputModel &out) const {
  if (!event.is_mc()) {
    out.fill("puWeight", 1.0f);
    out.fill("puWeightUp", 1.0f);
    out.fill("puWeightDown", 1.0f);
    return;
  }

  const auto era_it = config_.pu_eras.find(config_.era);
  if (era_it == config_.pu_eras.end()) {
    throw std::runtime_error("Missing PU era config for " + config_.era);
  }

  const auto json_file = config_.pu_payload_dir + "/" + era_it->second.payload_subdir;
  auto &cache = cache_for(json_file);
  const auto corr = cache.correction_set->at(era_it->second.correction_key);
  const auto npu = clip(event.scalar<float>("Pileup_nTrueInt"), 0.0f, 99.0f);

  const auto eval = [&](const char *variation) {
    return clip(static_cast<float>(corr->evaluate({static_cast<double>(npu), std::string(variation)})), 0.0f, 10.0f);
  };

  out.fill("puWeight", eval("nominal"));
  out.fill("puWeightUp", eval("up"));
  out.fill("puWeightDown", eval("down"));
}

}  // namespace nano
