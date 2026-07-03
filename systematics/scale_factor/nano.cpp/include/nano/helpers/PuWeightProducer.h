#pragma once

#include "nano/core/Event.h"
#include "nano/core/OutputModel.h"
#include "nano/producers/HeavyFlavBaseProducer.h"

#include <memory>
#include <string>

namespace correction {
class CorrectionSet;
}

namespace nano {

class PuWeightProducer {
public:
  explicit PuWeightProducer(const ProducerConfig &config);

  void begin_file(OutputModel &out) const;
  void fill(Event &event, OutputModel &out) const;

private:
  ProducerConfig config_;
};

}  // namespace nano
