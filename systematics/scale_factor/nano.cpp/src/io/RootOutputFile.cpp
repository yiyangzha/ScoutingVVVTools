#include "nano/io/RootOutputFile.h"

#include <TObject.h>

#include <stdexcept>

namespace nano {

namespace {

void branch_tree(TTree &tree, const std::string &name, RootOutputFile::BranchStorage &storage) {
  std::visit(
      [&](auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, bool>) {
          tree.Branch(name.c_str(), &value, (name + "/O").c_str());
        } else if constexpr (std::is_same_v<T, std::int32_t>) {
          tree.Branch(name.c_str(), &value, (name + "/I").c_str());
        } else if constexpr (std::is_same_v<T, std::uint32_t>) {
          tree.Branch(name.c_str(), &value, (name + "/i").c_str());
        } else if constexpr (std::is_same_v<T, std::uint64_t>) {
          tree.Branch(name.c_str(), &value, (name + "/l").c_str());
        } else if constexpr (std::is_same_v<T, float>) {
          tree.Branch(name.c_str(), &value, (name + "/F").c_str());
        } else if constexpr (std::is_same_v<T, std::vector<float>>) {
          tree.Branch(name.c_str(), &value);
        }
      },
      storage);
}

}  // namespace

RootOutputFile::RootOutputFile(std::string file_name) : file_name_(std::move(file_name)) {
  file_.reset(TFile::Open(file_name_.c_str(), "RECREATE"));
  if (!file_ || file_->IsZombie()) {
    throw std::runtime_error("Failed to create output ROOT file: " + file_name_);
  }
}

RootOutputFile::~RootOutputFile() = default;

void RootOutputFile::book_events(const OutputModel &model, std::string_view tree_name) {
  file_->cd();
  events_tree_ = new TTree(std::string(tree_name).c_str(), std::string(tree_name).c_str());
  for (const auto &[name, value] : model.defaults()) {
    storage_[name] = value;
    branch_tree(*events_tree_, name, storage_.at(name));
  }
}

void RootOutputFile::fill_event(const OutputModel &model) {
  if (!events_tree_) {
    throw std::runtime_error("Events tree has not been booked");
  }
  for (const auto &[name, value] : model.values()) {
    storage_.at(name) = value;
  }
  events_tree_->Fill();
}

void RootOutputFile::write() {
  file_->cd();
  if (events_tree_) {
    events_tree_->Write("", TObject::kOverwrite);
  }
  file_->Write("", TObject::kOverwrite);
  file_->Close();
}

}  // namespace nano
