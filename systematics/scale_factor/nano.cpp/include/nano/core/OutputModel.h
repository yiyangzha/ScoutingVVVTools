#pragma once

#include "nano/core/NanoTypes.h"

#include <string>
#include <unordered_map>

namespace nano {

class OutputModel {
public:
  void branch(std::string name, OutputValue default_value = 0.0f);
  void fill(std::string_view name, OutputValue value);
  bool has(std::string_view name) const;
  void reset();

  const std::unordered_map<std::string, OutputValue> &defaults() const { return defaults_; }
  const std::unordered_map<std::string, OutputValue> &values() const { return values_; }

private:
  std::unordered_map<std::string, OutputValue> defaults_;
  std::unordered_map<std::string, OutputValue> values_;
};

}  // namespace nano
