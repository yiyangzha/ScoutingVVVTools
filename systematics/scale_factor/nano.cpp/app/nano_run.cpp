#include "nano/core/Event.h"
#include "nano/io/NanoReader.h"
#include "nano/io/RootOutputFile.h"
#include "nano/producers/HeavyFlavMinimalProducer.h"
#include "nano/producers/HeavyFlavMuonSampleProducer.h"
#include "nano/producers/HeavyFlavScoutingMuonSampleProducer.h"

#include "runtime_common.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

bool parse_double_strict(const std::string &text, double &value) {
  char *end = nullptr;
  errno = 0;
  const auto parsed = std::strtod(text.c_str(), &end);
  if (errno != 0 || end == text.c_str() || *end != '\0') {
    return false;
  }
  value = parsed;
  return true;
}

nano::ChannelOptions parse_channel_options(const YAML::Node &settings, const std::string &channel) {
  nano::ChannelOptions options;
  const auto channels = settings["channels"];
  const auto node = channels && channels.IsMap() ? channels[channel] : YAML::Node{};
  if (!node) {
    return options;
  }
  if (!node.IsMap()) {
    throw std::runtime_error("channels." + channel + " must be a map");
  }
  for (const auto &item : node) {
    const auto key = item.first.as<std::string>();
    const auto value = item.second;
    if (!value.IsScalar()) {
      throw std::runtime_error("channels." + channel + "." + key + " must be a scalar");
    }
    const auto text = value.as<std::string>();
    options.strings[key] = text;
    if (text == "true" || text == "false") {
      options.bools[key] = value.as<bool>();
    }
    double number_value = 0.0;
    if (parse_double_strict(text, number_value)) {
      options.numbers[key] = number_value;
    }
  }
  return options;
}

std::vector<std::string> yaml_string_list_node(const YAML::Node &node) {
  std::vector<std::string> out;
  if (!node) {
    return out;
  }
  if (!node.IsSequence()) {
    throw std::runtime_error("Expected a YAML string list");
  }
  for (const auto &item : node) {
    out.push_back(item.as<std::string>());
  }
  return out;
}

const char *branch_type_name(nano::BranchType type) {
  switch (type) {
    case nano::BranchType::kBool:
      return "bool";
    case nano::BranchType::kInt32:
      return "int32";
    case nano::BranchType::kUInt32:
      return "uint32";
    case nano::BranchType::kUInt64:
      return "uint64";
    case nano::BranchType::kFloat:
      return "float";
    case nano::BranchType::kVecBool:
      return "vec_bool";
    case nano::BranchType::kVecUInt8:
      return "vec_uint8";
    case nano::BranchType::kVecUInt16:
      return "vec_uint16";
    case nano::BranchType::kVecInt16:
      return "vec_int16";
    case nano::BranchType::kVecInt32:
      return "vec_int32";
    case nano::BranchType::kVecFloat:
      return "vec_float";
  }
  return "unknown";
}

struct CliOptions {
  std::string input_files;
  std::string output_file;
  std::string tree_name = "Events";
  long long num_events = -1;
  std::string channel = "muon";
  std::string config_file;
  std::string variations;
  bool run_data = false;
  std::unordered_map<std::string, std::string> overrides;
};

CliOptions parse_args(int argc, char **argv) {
  CliOptions opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto need_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + name);
      }
      return argv[++i];
    };
    if (arg == "--input-files") {
      opts.input_files = need_value("--input-files");
    } else if (arg == "--output-file") {
      opts.output_file = need_value("--output-file");
    } else if (arg == "--tree-name") {
      opts.tree_name = need_value("--tree-name");
    } else if (arg == "--num-events") {
      opts.num_events = std::stoll(need_value("--num-events"));
    } else if (arg == "--channel") {
      opts.channel = need_value("--channel");
    } else if (arg == "--config") {
      opts.config_file = need_value("--config");
    } else if (arg == "--variations") {
      opts.variations = need_value("--variations");
    } else if (arg == "--run-data") {
      opts.run_data = true;
    } else if (arg == "--set") {
      const auto kv = need_value("--set");
      const auto pos = kv.find('=');
      if (pos == std::string::npos) {
        throw std::runtime_error("--set expects key=value");
      }
      opts.overrides[kv.substr(0, pos)] = kv.substr(pos + 1);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (opts.input_files.empty() || opts.output_file.empty() || opts.config_file.empty()) {
    throw std::runtime_error("Usage: nano_run --input-files <files> --output-file <out.root> --config <card.yaml> [--channel muon|minimal|scouting_muon] [--num-events -1] [--run-data] [--variations nominal,jes_up,...] [--set key=value]. If omitted, --variations defaults to nominal.");
  }
  return opts;
}

std::string normalized_variations_arg(const CliOptions &cli) {
  return cli.variations.empty() ? std::string("nominal") : cli.variations;
}

void validate_data_variations(const CliOptions &cli) {
  const auto variations = normalized_variations_arg(cli);
  if (!cli.run_data || variations == "nominal") {
    return;
  }
  throw std::runtime_error("--run-data does not support JME variations. If --variations is used with --run-data, it must be the single value 'nominal'; otherwise omit --variations.");
}

nano::ProducerConfig make_config(const YAML::Node &settings, const std::string &channel, bool run_data) {
  const auto validate_era_nano_version = [](const std::string &era, const std::string &nano_version) {
    if (nano_version != "v9" && nano_version != "v12" && nano_version != "v15") {
      throw std::runtime_error("nano_version must be one of: v9, v12, v15. Got: " + nano_version);
    }

    const auto run2 = std::set<std::string>{"2016APV", "2016", "2017", "2018"};
    const auto run3_early = std::set<std::string>{"2022", "2022EE", "2023", "2023BPix"};
    const auto run3_late = std::set<std::string>{"2024", "2025", "2026"};
    const auto allowed = (run2.count(era) != 0U && (nano_version == "v9" || nano_version == "v15")) ||
                         (run3_early.count(era) != 0U && (nano_version == "v12" || nano_version == "v15")) ||
                         (run3_late.count(era) != 0U && nano_version == "v15");
    if (!allowed) {
      throw std::runtime_error("Unsupported era/nano_version pair: era=" + era + ", nano_version=" + nano_version);
    }
  };

  const auto parse_branch_type = [](const std::string &type, const std::string &branch_name) {
    if (type == "bool") {
      return nano::BranchType::kBool;
    }
    if (type == "int32" || type == "int16" || type == "int8") {
      return nano::BranchType::kInt32;
    }
    if (type == "uint32" || type == "uint16" || type == "uint8") {
      return nano::BranchType::kUInt32;
    }
    if (type == "uint64") {
      return nano::BranchType::kUInt64;
    }
    if (type == "float") {
      return nano::BranchType::kFloat;
    }
    if (type == "vec_bool") {
      return nano::BranchType::kVecBool;
    }
    if (type == "vec_uint8") {
      return nano::BranchType::kVecUInt8;
    }
    if (type == "vec_uint16") {
      return nano::BranchType::kVecUInt16;
    }
    if (type == "vec_int16") {
      return nano::BranchType::kVecInt16;
    }
    if (type == "vec_int32" || type == "vec_int8") {
      return nano::BranchType::kVecInt32;
    }
    if (type == "vec_float") {
      return nano::BranchType::kVecFloat;
    }
    throw std::runtime_error("Unsupported NanoAOD branch type '" + type + "' for branch " + branch_name);
  };

  nano::ProducerConfig config;
  config.channel = channel;
  config.era = settings["era"].as<std::string>();
  config.nano_version = settings["nano_version"].as<std::string>();
  validate_era_nano_version(config.era, config.nano_version);
  if (!settings["preselection"]) {
    throw std::runtime_error("Missing preselection in config");
  }
  config.preselection = settings["preselection"].as<std::string>();
  config.channel_options = parse_channel_options(settings, channel);
  config.required_triggers = nano::runtime::yaml_string_list(settings, "required_triggers");
  if (settings["input"]) {
    const auto input = settings["input"];
    if (input["strict_read_branches"]) {
      config.strict_read_branches = input["strict_read_branches"].as<bool>();
    }
    config.optional_read_branches = yaml_string_list_node(input["optional_read_branches"]);
  }
  if (settings["strict_read_branches"]) {
    config.strict_read_branches = settings["strict_read_branches"].as<bool>();
  }
  if (settings["optional_read_branches"]) {
    auto extra_optional = yaml_string_list_node(settings["optional_read_branches"]);
    config.optional_read_branches.insert(config.optional_read_branches.end(), extra_optional.begin(), extra_optional.end());
  }

  const auto catalogue = settings["nano_branches"][config.nano_version]["trees"]["Events"]["branches"];
  if (!catalogue) {
    throw std::runtime_error("Missing NanoAOD branch catalogue for " + config.nano_version);
  }
  config.read_branches = nano::runtime::yaml_string_list(settings, "read_branches");
  if (config.read_branches.empty()) {
    throw std::runtime_error("Missing or empty read_branches list in config");
  }
  if (run_data) {
    auto mc_only = yaml_string_list_node(settings["mc_only_branches"]);
    if (settings["input"]) {
      auto nested_mc_only = yaml_string_list_node(settings["input"]["mc_only_branches"]);
      mc_only.insert(mc_only.end(), nested_mc_only.begin(), nested_mc_only.end());
    }
    const std::set<std::string> mc_only_set(mc_only.begin(), mc_only.end());
    if (!mc_only_set.empty()) {
      config.read_branches.erase(
          std::remove_if(config.read_branches.begin(), config.read_branches.end(),
                         [&](const std::string &branch) { return mc_only_set.count(branch) != 0U; }),
          config.read_branches.end());
    }
  }
  if (settings["output"] && settings["output"]["include_lhe_weights"]) {
    config.include_lhe_weights = settings["output"]["include_lhe_weights"].as<bool>();
  }

  // read_branches safety checks:
  // - Runtime cards should explicitly list every physical NanoAOD branch the
  //   channel reads.
  // - required_triggers, stored_tagger_names, and optional LHEScaleWeight can
  //   imply extra physical branches; auto-add them for backward compatibility,
  //   but warn so the card can be made explicit.
  // - If LHEScaleWeight is listed while output.include_lhe_weights is disabled,
  //   remove it so a stale read_branches entry does not silently read an unused
  //   large vector branch.
  if (!config.include_lhe_weights) {
    const auto old_size = config.read_branches.size();
    config.read_branches.erase(std::remove(config.read_branches.begin(), config.read_branches.end(), "LHEScaleWeight"),
                               config.read_branches.end());
    if (config.read_branches.size() != old_size) {
      std::cerr << "Warning: removing LHEScaleWeight from read_branches because output.include_lhe_weights is false.\n";
    }
  }
  if (!settings["stored_tagger_names"]) {
    throw std::runtime_error("Missing stored_tagger_names list in config");
  }
  for (const auto &item : settings["stored_tagger_names"]) {
    config.tagger_names.push_back(item.as<std::string>());
  }
  std::set<std::string> seen(config.read_branches.begin(), config.read_branches.end());
  for (const auto &trigger : config.required_triggers) {
    if (seen.insert(trigger).second) {
      std::cerr << "Info: adding branch " << trigger << " to read_branches from required_triggers.\n";
      config.read_branches.push_back(trigger);
    }
    config.nano_branch_types[trigger] = nano::BranchType::kBool;
  }
  const bool auto_add_tagger_branches =
      !settings["stored_tagger_auto_add_branches"] || settings["stored_tagger_auto_add_branches"].as<bool>();
  if (auto_add_tagger_branches) {
    for (const auto &tagger : config.tagger_names) {
      const auto branch_name = "FatJet_" + tagger;
      if (seen.insert(branch_name).second) {
        std::cerr << "Info: adding branch " << branch_name << " to read_branches from stored_tagger_names.\n";
        config.read_branches.push_back(branch_name);
      }
    }
  }
  if (!run_data && config.include_lhe_weights && seen.insert("LHEScaleWeight").second) {
    std::cerr << "Warning: adding missing branch LHEScaleWeight to read_branches because output.include_lhe_weights is true. "
                 "Please list it explicitly in read_branches.\n";
    config.read_branches.push_back("LHEScaleWeight");
  }
  for (const auto &branch_name : config.read_branches) {
    if (config.nano_branch_types.count(branch_name) != 0U) {
      continue;
    }
    const auto branch_node = catalogue[branch_name];
    if (!branch_node) {
      throw std::runtime_error("Branch " + branch_name + " is not listed in nano_branches for " + config.nano_version);
    }
    config.nano_branch_types[branch_name] = parse_branch_type(branch_node["type"].as<std::string>(), branch_name);
  }
  const auto btag_node = settings["btag"][config.nano_version][config.era];
  if (!btag_node) {
    throw std::runtime_error("Missing btag config for nano version " + config.nano_version + " and era " + config.era);
  }
  config.btag_config.branch = btag_node["branch"].as<std::string>();
  config.btag_config.loose = btag_node["loose"] ? btag_node["loose"].as<float>() : 0.0f;
  config.btag_config.medium = btag_node["medium"] ? btag_node["medium"].as<float>() : 0.0f;
  config.btag_config.tight = btag_node["tight"] ? btag_node["tight"].as<float>() : 0.0f;
  config.btag_config.xtight = btag_node["xtight"] ? btag_node["xtight"].as<float>() : 0.0f;
  config.btag_config.xxtight = btag_node["xxtight"] ? btag_node["xxtight"].as<float>() : 0.0f;
  config.year_value = settings["year_values"][config.era].as<float>();
  config.lumi_weight = settings["lumi_values"][config.era].as<float>();
  config.jme_payload_dir = settings["jec"]["payload_dir"].as<std::string>();
  config.jme_jer_smear_json = settings["jec"]["jer_smear_json"].as<std::string>();
  if (settings["jec"]["systematics"]) {
    const auto syst = settings["jec"]["systematics"];
    config.jme_jes = syst["jes"] ? syst["jes"].as<std::string>() : "";
    config.jme_jer = syst["jer"] ? syst["jer"].as<std::string>() : "nominal";
    config.jme_met_unclustered = syst["met_unclustered"] ? syst["met_unclustered"].as<std::string>() : "";
    config.jme_smear_met = syst["smear_met"] ? syst["smear_met"].as<bool>() : false;
  }
  auto parse_jme_object = [](const YAML::Node &node, std::string default_algo, std::string default_jerc_file) {
    nano::JmeObjectConfig object;
    object.payload_subdir = node["payload_subdir"] ? node["payload_subdir"].as<std::string>() : "";
    object.jerc_file = node["jerc_file"] ? node["jerc_file"].as<std::string>() : std::move(default_jerc_file);
    object.algo = node["algo"] ? node["algo"].as<std::string>() : std::move(default_algo);
    object.jec_tag_mc = node["jec_tag_mc"] ? node["jec_tag_mc"].as<std::string>() : "";
    object.jec_tag_data = node["jec_tag_data"] ? node["jec_tag_data"].as<std::string>() : "";
    object.jer_tag_mc = node["jer_tag_mc"] ? node["jer_tag_mc"].as<std::string>() : "inherit";
    return object;
  };
  const auto campaigns = settings["jec"]["campaigns"] ? settings["jec"]["campaigns"] : settings["jec"]["eras"];
  for (const auto &item : campaigns) {
    nano::JmeEraConfig era_cfg;
    era_cfg.payload_subdir = item.second["payload_subdir"].as<std::string>();
    era_cfg.jet_jerc_file = item.second["jet_jerc_file"] ? item.second["jet_jerc_file"].as<std::string>() : "jet_jerc.json.gz";
    era_cfg.fatjet_jerc_file = item.second["fatjet_jerc_file"] ? item.second["fatjet_jerc_file"].as<std::string>() : "fatJet_jerc.json.gz";
    era_cfg.met_xy_corr_era = item.second["met_xy_corr_era"].as<std::string>();
    if (item.second["jes_uncertainties"]) {
      for (const auto &unc : item.second["jes_uncertainties"]) {
        era_cfg.jes_uncertainties.push_back(unc.as<std::string>());
      }
    }
    era_cfg.jet = parse_jme_object(item.second["jet"], "AK4PFPuppi", era_cfg.jet_jerc_file);
    era_cfg.fatjet = parse_jme_object(item.second["fatjet"], "AK8PFPuppi", era_cfg.fatjet_jerc_file);
    era_cfg.subjet = parse_jme_object(item.second["subjet"], "AK4PFPuppi", era_cfg.jet_jerc_file);
    config.jme_eras[item.first.as<std::string>()] = era_cfg;
  }
  config.pu_payload_dir = settings["pu"]["payload_dir"].as<std::string>();
  for (const auto &item : settings["pu"]["eras"]) {
    config.pu_eras[item.first.as<std::string>()] = {
        item.second["payload_subdir"].as<std::string>(),
        item.second["correction_key"].as<std::string>(),
    };
  }
  if (settings["jet_veto_map"]) {
    const auto node = settings["jet_veto_map"];
    config.jet_veto_map_enabled = node["enabled"] ? node["enabled"].as<bool>() : false;
    config.jet_veto_map_payload_dir = node["payload_dir"] ? node["payload_dir"].as<std::string>() : config.jme_payload_dir;
    config.jet_veto_map_type = node["type"] ? node["type"].as<std::string>() : "jetvetomap";
    if (node["campaigns"]) {
      for (const auto &item : node["campaigns"]) {
        config.jet_veto_map_eras[item.first.as<std::string>()] = {
            item.second["payload_subdir"].as<std::string>(),
            item.second["correction"].as<std::string>(),
        };
      }
    }
  }
  return config;
}

std::unique_ptr<nano::HeavyFlavBaseProducer> make_producer(const nano::ProducerConfig &config) {
  if (config.channel == "muon") {
    return std::make_unique<nano::HeavyFlavMuonSampleProducer>(config);
  }
  if (config.channel == "minimal") {
    return std::make_unique<nano::HeavyFlavMinimalProducer>(config);
  }
  if (config.channel == "scouting_muon") {
    return std::make_unique<nano::HeavyFlavScoutingMuonSampleProducer>(config);
  }
  throw std::runtime_error("Unsupported channel: " + config.channel);
}

std::string variation_output_path(const std::string &output_file, nano::JmeVariation variation) {
  const fs::path path(output_file);
  const auto suffix = "_" + std::string(nano::variation_name(variation));
  const auto extension = path.has_extension() ? path.extension().string() : std::string{};
  const auto stem = path.has_extension() ? path.stem().string() : path.filename().string();
  const auto parent = path.parent_path();
  return (parent / (stem + suffix + extension)).string();
}

std::string data_lumi_mask_path(const YAML::Node &settings, const nano::ProducerConfig &config) {
  const auto masks = settings["data_lumi_masks"];
  if (!masks || !masks[config.era]) {
    throw std::runtime_error("Missing data_lumi_masks entry for era " + config.era + " in config");
  }
  return masks[config.era].as<std::string>();
}

void warn_missing_optional_branches(TTree &tree, const nano::ProducerConfig &config, const std::string &input_file) {
  if (config.optional_read_branches.empty()) {
    return;
  }

  const std::set<std::string> read_set(config.read_branches.begin(), config.read_branches.end());
  std::vector<std::string> missing;
  for (const auto &branch : config.optional_read_branches) {
    if (read_set.count(branch) == 0U) {
      continue;
    }
    if (!tree.GetBranch(branch.c_str())) {
      missing.push_back(branch);
    }
  }
  if (missing.empty()) {
    return;
  }

  std::cerr << "WARNING: missing optional branch";
  if (missing.size() > 1U) {
    std::cerr << "es";
  }
  std::cerr << " in " << input_file << ": ";
  for (std::size_t i = 0; i < missing.size(); ++i) {
    if (i != 0U) {
      std::cerr << ", ";
    }
    std::cerr << missing[i];
  }
  std::cerr << ". They will use default values";
  if (std::any_of(missing.begin(), missing.end(), [](const std::string &name) { return name.rfind("GenPart_", 0) == 0; })) {
    std::cerr << "; GenPart-based matching will fall back to default W/Z matches with GenJet flavour hints where available";
  }
  std::cerr << ".\n";
}

std::unique_ptr<TFile> open_input_file_with_retry(const std::string &input_file) {
  const bool remote_input = nano::runtime::is_remote_path(input_file);
  const int max_retries = remote_input ? 5 : 0;
  const int retry_sleep_seconds = 5;
  for (int retry = 0; retry <= max_retries; ++retry) {
    auto input = std::unique_ptr<TFile>(TFile::Open(input_file.c_str(), "READ"));
    if (input && !input->IsZombie()) {
      return input;
    }
    if (retry < max_retries) {
      std::cerr << "Warning: failed to open remote input file " << input_file
                << "; retry " << (retry + 1) << "/" << max_retries
                << " after " << retry_sleep_seconds << " seconds\n";
      sleep(retry_sleep_seconds);
    }
  }
  throw std::runtime_error("Failed to open input file: " + input_file);
}

std::vector<std::string> process_one_file_variations(const std::string &input_file, const std::string &output_file, const CliOptions &cli,
                                                     const YAML::Node &settings,
                                                     const std::vector<nano::JmeVariation> &variations) {
  auto input = open_input_file_with_retry(input_file);
  auto *tree = dynamic_cast<TTree *>(input->Get(cli.tree_name.c_str()));
  if (!tree) {
    throw std::runtime_error("Missing tree " + cli.tree_name + " in " + input_file);
  }

  const auto config = make_config(settings, cli.channel, cli.run_data);
  std::cerr << "nano_run debug: input opened file=" << input_file << " tree=" << cli.tree_name
            << " entries=" << tree->GetEntries() << " channel=" << cli.channel << " run_data="
            << (cli.run_data ? "true" : "false") << " era=" << config.era << " nano_version=" << config.nano_version
            << " strict_read_branches=" << (config.strict_read_branches ? "true" : "false")
            << " read_branch_count=" << config.read_branches.size() << "\n";
  for (const auto &branch : config.read_branches) {
    const auto type_it = config.nano_branch_types.find(branch);
    std::cerr << "nano_run debug: bind branch=" << branch << " type="
              << (type_it == config.nano_branch_types.end() ? "missing-from-config" : branch_type_name(type_it->second)) << "\n";
  }
  const auto lumi_mask = cli.run_data ? std::make_unique<nano::runtime::LumiMask>(nano::runtime::LumiMask::from_file(data_lumi_mask_path(settings, config)))
                                      : nullptr;
  warn_missing_optional_branches(*tree, config, input_file);
  nano::NanoReader reader(*tree, nano::BranchSchema(nano::HeavyFlavBaseProducer::default_schema(config)));
  std::cerr << "nano_run debug: all declared branches bound successfully\n";
  auto producer_base = make_producer(config);
  auto *producer = producer_base.get();
  std::cerr << "nano_run debug: producer initialized\n";
  producer->begin_file();
  std::cerr << "nano_run debug: output model initialized\n";

  struct VariationOutput {
    nano::JmeVariation variation;
    std::string file_name;
    std::unique_ptr<nano::RootOutputFile> output;
    std::size_t accepted = 0;
    std::set<nano::runtime::RunLumi> selected_lumis;
  };
  std::vector<VariationOutput> outputs;
  outputs.reserve(variations.size());
  for (const auto variation : variations) {
    auto path = variation_output_path(output_file, variation);
    if (const auto parent = fs::path(path).parent_path(); !parent.empty()) {
      fs::create_directories(parent);
    }
    auto output = std::make_unique<nano::RootOutputFile>(path);
    output->book_events(producer->output());
    outputs.push_back({variation, std::move(path), std::move(output), 0U});
  }

  const auto entry_list = nano::runtime::build_entry_list(*tree, config.preselection, cli.num_events, lumi_mask.get());
  std::cerr << "nano_run debug: preselection='" << config.preselection << "' selected_entries=" << entry_list.size() << "\n";

  bool reported_first_common = false;
  bool reported_first_accepted = false;
  for (const auto entry : entry_list) {
    nano::Event event(reader, static_cast<std::size_t>(entry));
    if (!producer->analyze_common(event)) {
      continue;
    }
    if (!reported_first_common) {
      std::cerr << "nano_run debug: first event passing channel-common selection entry=" << entry << "\n";
      reported_first_common = true;
    }
    const auto jme_result = producer->compute_jme_result(event);
    for (auto &item : outputs) {
      if (!producer->analyze_variation(event, jme_result, item.variation)) {
        continue;
      }
      if (!reported_first_accepted) {
        std::cerr << "nano_run debug: first accepted event entry=" << entry
                  << " variation=" << nano::variation_name(item.variation) << "\n";
        reported_first_accepted = true;
      }
      item.output->fill_event(producer->output());
      item.selected_lumis.insert({event.scalar<std::uint32_t>("run"), event.scalar<std::uint32_t>("luminosityBlock")});
      ++item.accepted;
    }
  }

  std::vector<std::string> output_files;
  for (auto &item : outputs) {
    nano::runtime::copy_filtered_runs_tree(*input, item.output->file(), item.selected_lumis);
    nano::runtime::copy_filtered_luminosity_blocks_tree(*input, item.output->file(), item.selected_lumis);
    item.output->write();
    std::cout << "input=" << input_file << " processed=" << entry_list.size() << " accepted=" << item.accepted
              << " variation=" << nano::variation_name(item.variation) << " output=" << item.file_name << "\n";
    output_files.push_back(item.file_name);
  }
  return output_files;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const auto cli = parse_args(argc, argv);
    validate_data_variations(cli);
    auto settings = nano::runtime::load_config_with_extends(cli.config_file);
    for (const auto &[key, value] : cli.overrides) {
      nano::runtime::apply_override(settings, key, value);
    }

    auto inputs = nano::runtime::split_csv(cli.input_files);
    for (auto &input : inputs) {
      input = nano::runtime::normalize_input_path(input);
    }

    const auto variations = nano::parse_jme_variation_list(normalized_variations_arg(cli));
    if (inputs.size() == 1U) {
      process_one_file_variations(inputs.front(), cli.output_file, cli, settings, variations);
      return 0;
    }

    const auto temp_dir = fs::path("run") / ("pieces_" + std::to_string(::getpid()));
    fs::create_directories(temp_dir);
    std::unordered_map<std::string, std::vector<std::string>> piece_outputs;
    for (const auto variation : variations) {
      piece_outputs[std::string(nano::variation_name(variation))] = {};
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto piece_base = (temp_dir / ("piece_" + std::to_string(i) + ".root")).string();
      const auto outputs = process_one_file_variations(inputs[i], piece_base, cli, settings, variations);
      for (std::size_t j = 0; j < variations.size(); ++j) {
        piece_outputs[std::string(nano::variation_name(variations[j]))].push_back(outputs[j]);
      }
    }
    for (const auto variation : variations) {
      const auto name = std::string(nano::variation_name(variation));
      const auto final_output = variation_output_path(cli.output_file, variation);
      nano::runtime::merge_root_files(piece_outputs.at(name), final_output);
      std::cout << "merged=" << final_output << " variation=" << name << " pieces=" << piece_outputs.at(name).size() << "\n";
    }
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "nano_run failed: " << ex.what() << "\n";
    return 1;
  }
}
