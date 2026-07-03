#include "nano/core/OutputModel.h"

#include <stdexcept>

namespace nano {

void OutputModel::branch(std::string name, OutputValue default_value) {
  defaults_[name] = default_value;
  values_[name] = default_value;
}

void OutputModel::fill(std::string_view name, OutputValue value) {
  if (!has(name)) {
    throw std::out_of_range("Unknown output branch: " + std::string(name));
  }
  values_[std::string(name)] = std::move(value);
}

bool OutputModel::has(std::string_view name) const {
  return defaults_.count(std::string(name)) > 0U;
}

void OutputModel::reset() {
  values_ = defaults_;
}

}  // namespace nano
