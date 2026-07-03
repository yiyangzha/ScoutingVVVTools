#pragma once

#include "nano/producers/HeavyFlavBaseProducer.h"

namespace nano {

class HeavyFlavScoutingMuonSampleProducer : public HeavyFlavBaseProducer {
public:
  explicit HeavyFlavScoutingMuonSampleProducer(ProducerConfig config);

  void begin_file() override;
  bool analyze(Event &event) override;
  bool analyze_common(Event &event) override;
  JmeEventResult compute_jme_result(Event &event) const override;
  bool analyze_variation(Event &event, const JmeEventResult &jme_result, JmeVariation variation) override;

private:
  float tagger_score(const ObjectView &fj, const std::string &name) const;
};

}  // namespace nano
