#pragma once

#include "nano/core/Collection.h"

#include <vector>

namespace nano {

class FatjetGenMatching {
public:
  void process(Event &event, std::vector<ObjectView> &fatjets) const;
};

}  // namespace nano
