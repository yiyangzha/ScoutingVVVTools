#include "runtime_common.h"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

struct CliOptions {
  std::string input_yaml;
  std::string job_dir;
  std::string output_dir;
  std::string merge_output_dir;
  std::string config_file;
  std::string channel = "muon";
  std::string tree_name = "Events";
  std::string variations;
  bool run_data = false;
  bool use_sample_key_nickname = false;
  bool download_remote_inputs = false;
  long long num_events = -1;
  long long request_disk_mb = 50000;
  std::size_t nfiles_per_job = 1;
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
    if (arg == "--input-yaml") {
      opts.input_yaml = need_value("--input-yaml");
    } else if (arg == "--job-dir") {
      opts.job_dir = need_value("--job-dir");
    } else if (arg == "--output-dir") {
      opts.output_dir = need_value("--output-dir");
    } else if (arg == "--merge-output-dir") {
      opts.merge_output_dir = need_value("--merge-output-dir");
    } else if (arg == "--config") {
      opts.config_file = need_value("--config");
    } else if (arg == "--channel") {
      opts.channel = need_value("--channel");
    } else if (arg == "--tree-name") {
      opts.tree_name = need_value("--tree-name");
    } else if (arg == "--variations") {
      opts.variations = need_value("--variations");
    } else if (arg == "--run-data") {
      opts.run_data = true;
    } else if (arg == "--use-sample-key-nickname") {
      opts.use_sample_key_nickname = true;
    } else if (arg == "--download-remote-inputs") {
      opts.download_remote_inputs = true;
    } else if (arg == "--no-download-remote-inputs") {
      opts.download_remote_inputs = false;
    } else if (arg == "--num-events") {
      opts.num_events = std::stoll(need_value("--num-events"));
    } else if (arg == "--request-disk-mb") {
      opts.request_disk_mb = std::stoll(need_value("--request-disk-mb"));
    } else if (arg == "--nfiles-per-job") {
      opts.nfiles_per_job = static_cast<std::size_t>(std::stoul(need_value("--nfiles-per-job")));
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
  if (opts.input_yaml.empty() || opts.job_dir.empty() || opts.output_dir.empty() || opts.config_file.empty()) {
    throw std::runtime_error("Usage: nano_make_condor --input-yaml <samples.yaml> --job-dir <condor-dir> --output-dir <dir> [--merge-output-dir <local-dir>] --config <card.yaml> [--nfiles-per-job 1] [--request-disk-mb 50000] [--variations nominal,jes_up,...] [--use-sample-key-nickname] [--download-remote-inputs|--no-download-remote-inputs]. If omitted, --variations defaults to nominal.");
  }
  if (opts.merge_output_dir.empty()) {
    opts.merge_output_dir = opts.output_dir;
  }
  if (opts.request_disk_mb <= 0) {
    throw std::runtime_error("--request-disk-mb must be positive");
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

std::string write_merged_config(const fs::path &path, const YAML::Node &settings) {
  nano::runtime::dump_yaml_file(settings, path.string());
  return path.string();
}

std::string read_text_file(const fs::path &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to read template: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

void write_text_file(const fs::path &path, const std::string &content) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Failed to write file: " + path.string());
  }
  output << content;
}

void replace_all(std::string &text, const std::string &from, const std::string &to) {
  std::size_t pos = 0;
  while ((pos = text.find(from, pos)) != std::string::npos) {
    text.replace(pos, from.size(), to);
    pos += to.size();
  }
}

std::string render_template(const fs::path &path, const std::map<std::string, std::string> &replacements) {
  auto text = read_text_file(path);
  for (const auto &[key, value] : replacements) {
    replace_all(text, key, value);
  }
  return text;
}

std::string json_escape(const std::string &value) {
  std::string out;
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += ch;
        break;
    }
  }
  return out;
}

std::string nickname_from_dataset(const std::string &dataset, const std::string &fallback) {
  if (!dataset.empty() && dataset.front() == '/') {
    const auto second_slash = dataset.find('/', 1);
    if (second_slash != std::string::npos && second_slash > 1) {
      return dataset.substr(1, second_slash - 1);
    }
  }
  return fallback;
}

std::string safe_file_stem(std::string value) {
  for (auto &ch : value) {
    const auto uch = static_cast<unsigned char>(ch);
    if (!std::isalnum(uch) && ch != '_' && ch != '-' && ch != '.') {
      ch = '_';
    }
  }
  return value;
}

bool is_remote_output(const std::string &path) {
  return nano::runtime::starts_with(path, "root://");
}

std::string trim_trailing_slashes(std::string value) {
  while (!value.empty() && value.back() == '/') {
    value.pop_back();
  }
  return value;
}

std::string join_path_string(const std::string &base, const std::string &child) {
  if (is_remote_output(base)) {
    return trim_trailing_slashes(base) + "/" + child;
  }
  return (fs::path(base) / child).string();
}

struct JobSpec {
  std::size_t index = 0;
  std::string nickname;
  std::size_t nickname_index = 0;
  std::vector<std::string> inputs;
  std::string output_file;
};

void write_job_manifest(const fs::path &path, const std::vector<JobSpec> &jobs) {
  std::ofstream out(path);
  out << "{\n";
  out << "  \"jobs\": [\n";
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    const auto &job = jobs[i];
    out << "    {\n";
    out << "      \"index\": " << job.index << ",\n";
    out << "      \"nickname\": \"" << json_escape(job.nickname) << "\",\n";
    out << "      \"nickname_index\": " << job.nickname_index << ",\n";
    out << "      \"output_file\": \"" << json_escape(job.output_file) << "\",\n";
    out << "      \"inputs\": [\n";
    for (std::size_t j = 0; j < job.inputs.size(); ++j) {
      out << "        \"" << json_escape(job.inputs[j]) << "\"";
      if (j + 1 != job.inputs.size()) {
        out << ",";
      }
      out << "\n";
    }
    out << "      ]\n";
    out << "    }";
    if (i + 1 != jobs.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
}

void write_job_manifest_tsv(const fs::path &path, const std::vector<JobSpec> &jobs) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to write job manifest TSV: " + path.string());
  }
  for (const auto &job : jobs) {
    out << job.index << "\t" << job.output_file << "\t";
    for (std::size_t i = 0; i < job.inputs.size(); ++i) {
      if (i != 0U) {
        out << ",";
      }
      out << job.inputs[i];
    }
    out << "\n";
  }
}

std::string require_env(const char *name) {
  const auto *value = std::getenv(name);
  if (!value || std::string(value).empty()) {
    throw std::runtime_error(std::string("Required environment variable is not set: ") + name);
  }
  return value;
}

std::string env_or_empty(const char *name) {
  const auto *value = std::getenv(name);
  return value ? std::string(value) : std::string();
}

std::string shell_quote(const std::string &value) {
  std::string out = "'";
  for (const char ch : value) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out += ch;
    }
  }
  out += "'";
  return out;
}

void create_worker_runtime_bundle(const fs::path &workdir) {
  const auto prefix = require_env("SCALE_FACTOR_CONDA_PREFIX");
  const auto script = fs::path("tools") / "package_worker_runtime.py";
  const auto build_dir = fs::path("build");
  if (!fs::exists(script)) {
    throw std::runtime_error("Missing worker runtime packager: " + script.string());
  }
  if (!fs::exists(build_dir / "nano_run")) {
    throw std::runtime_error("nano_run is missing from build/. Build nano.cpp before generating Condor jobs.");
  }

  const auto output = workdir / "worker_runtime.tar.gz";
  const auto cmd = std::string("env -u PYTHONHOME -u PYTHONPATH python3 ") + shell_quote(script.string()) +
                   " --prefix " + shell_quote(prefix) +
                   " --build-dir " + shell_quote(build_dir.string()) +
                   " --output " + shell_quote(output.string());
  const auto rc = std::system(cmd.c_str());
  if (rc != 0) {
    throw std::runtime_error("Failed to create worker runtime bundle");
  }
}

void write_executable_template(const fs::path &path, const fs::path &template_path, const std::map<std::string, std::string> &replacements) {
  write_text_file(path, render_template(template_path, replacements));
  fs::permissions(path, fs::perms::owner_exec | fs::perms::owner_read | fs::perms::owner_write | fs::perms::group_exec |
                            fs::perms::group_read | fs::perms::others_exec | fs::perms::others_read,
                  fs::perm_options::add);
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

    const auto workdir = fs::path(cli.job_dir);
    fs::create_directories(workdir);
    const auto template_dir = fs::path("templates") / "condor";
    const auto merged_config = write_merged_config(workdir / "config_snapshot.yaml", settings);
    write_executable_template(workdir / "process.sh", template_dir / "process.sh.in",
                              {
                                  {"@DAS_HOME@", env_or_empty("SCALE_FACTOR_DAS_HOME")},
                              });

    const auto tarball = (workdir / "repo.tar.gz").string();
    const auto tar_cmd = "tar czf " + tarball +
                         " --exclude='./.git'"
                         " --exclude='./run'"
                         " --exclude='./jobs'"
                         " --exclude='./build'"
                         " --exclude='./build-*'"
                         " --exclude='./cmake-build-*'"
                         " --exclude='./tests/data/muon_validation'"
                         " --exclude='./external/CMSJMECalculators/tests'"
                         " .";
    const auto rc = std::system(tar_cmd.c_str());
    if (rc != 0) {
      throw std::runtime_error("Failed to create repository tarball");
    }
    create_worker_runtime_bundle(workdir);

    const bool remote_output = is_remote_output(cli.output_dir);
    const auto output_base = remote_output ? cli.output_dir : fs::absolute(cli.output_dir).string();
    const auto output_root = join_path_string(output_base, "pieces");
    if (!remote_output) {
      fs::create_directories(output_root);
    }

    const auto sample_map = nano::runtime::parse_sample_yaml(cli.input_yaml);
    std::map<std::string, std::vector<std::string>> files_by_nickname;
    for (const auto &[sample, datasets] : sample_map) {
      for (const auto &dataset : datasets) {
        const auto nickname = cli.use_sample_key_nickname ? sample : nickname_from_dataset(dataset, sample);
        std::cout << "Resolving sample=" << sample << " nickname=" << nickname << " dataset=" << dataset << "\n";
        const auto resolved = nano::runtime::resolve_dataset_entry(dataset);
        std::cout << "  files=" << resolved.size() << "\n";
        auto &files = files_by_nickname[nickname];
        files.insert(files.end(), resolved.begin(), resolved.end());
      }
    }

    std::vector<JobSpec> jobs;
    std::size_t job_index = 0;
    for (const auto &[nickname, files] : files_by_nickname) {
      std::cout << "Grouping nickname=" << nickname << " total_files=" << files.size() << "\n";
      std::size_t nickname_index = 0;
      for (std::size_t begin = 0; begin < files.size(); begin += cli.nfiles_per_job, ++nickname_index) {
        const auto end = std::min(files.size(), begin + cli.nfiles_per_job);
        JobSpec job;
        job.index = job_index++;
        job.nickname = nickname;
        job.nickname_index = nickname_index;
        job.inputs.assign(files.begin() + static_cast<std::ptrdiff_t>(begin), files.begin() + static_cast<std::ptrdiff_t>(end));
        job.output_file = join_path_string(output_root, safe_file_stem(nickname) + "_" + std::to_string(nickname_index) + ".root");
        jobs.push_back(std::move(job));
      }
    }

    write_job_manifest(workdir / "job_manifest.json", jobs);
    write_job_manifest_tsv(workdir / "job_manifest.tsv", jobs);
    std::ofstream index_list(workdir / "job_indices.txt");
    for (const auto &job : jobs) {
      index_list << job.index << "\n";
    }

    const auto variations_arg = normalized_variations_arg(cli);
    const auto run_data_arg = cli.run_data ? std::string("true") : std::string("false");
    const auto download_remote_inputs_arg = cli.download_remote_inputs ? std::string("true") : std::string("false");
    const auto submit_replacements = std::map<std::string, std::string>{
        {"@TREE_NAME@", cli.tree_name},
        {"@NUM_EVENTS@", std::to_string(cli.num_events)},
        {"@CHANNEL@", cli.channel},
        {"@VARIATIONS@", variations_arg},
        {"@RUN_DATA@", run_data_arg},
        {"@DOWNLOAD_REMOTE_INPUTS@", download_remote_inputs_arg},
        {"@NUM_JOBS@", std::to_string(jobs.size())},
        {"@REQUEST_DISK_KB@", std::to_string(cli.request_disk_mb * 1024LL)},
    };
    const auto submit_jdl = render_template(
        template_dir / "submit.jdl.in",
        submit_replacements);
    write_text_file(workdir / "submit.jdl", submit_jdl);
    const auto submit_lxplus_jdl = render_template(
        template_dir / "submit_lxplus.jdl.in",
        submit_replacements);
    write_text_file(workdir / "submit_lxplus.jdl", submit_lxplus_jdl);
    write_executable_template(workdir / "submit.sh", template_dir / "submit.sh.in", submit_replacements);

    fs::create_directories(workdir / "logs");
    std::cout << "Created condor workdir: " << workdir << "\n";
    std::cout << "Output base dir: " << output_base << "\n";
    std::cout << "Piece output dir: " << output_root << "\n";
    std::cout << "Condor job logs: " << (workdir / "logs") << "\n";
    std::cout << "  stdout: logs/<Cluster>.<Process>.out\n";
    std::cout << "  stderr: logs/<Cluster>.<Process>.err\n";
    std::cout << "  scheduler: logs/<Cluster>.log\n";
    std::cout << "Job manifest: " << (workdir / "job_manifest.json") << "\n";
    std::cout << "Job manifest TSV: " << (workdir / "job_manifest.tsv") << "\n";
    std::cout << "Job index list: " << (workdir / "job_indices.txt") << "\n";
    std::cout << "Jobs: " << jobs.size() << "\n";
    std::cout << "Next step:\n";
    std::cout << "  cd " << workdir << " && ./submit.sh\n";
    if (remote_output) {
      std::cout << "After jobs finish, merge Tier pieces to local ntuples with:\n";
      std::cout << "  " << fs::current_path() / "build" / "nano_merge" << " " << cli.merge_output_dir
                << " --pieces-dir " << output_root << "\n";
    } else {
      std::cout << "After jobs finish, merge local pieces with:\n";
      std::cout << "  " << fs::current_path() / "build" / "nano_merge" << " " << cli.merge_output_dir << "\n";
    }
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "nano_make_condor failed: " << ex.what() << "\n";
    return 1;
  }
}
