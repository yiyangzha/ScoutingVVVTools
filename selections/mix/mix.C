// Summary: Shuffle all entries of a sample's trees across its ROOT chunk files, preserving the
// per-chunk entry counts so that the output mirrors the input chunk layout. Intended to be run
// after selections/convert so that downstream BDT train/test splits by entry index are drawn from
// a randomised ordering rather than any intrinsic file ordering.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <TChain.h>
#include <TFile.h>
#include <TROOT.h>
#include <TTree.h>

#include "../../src/simple_json.h"

using namespace std;
namespace fs = std::filesystem;
using JsonValue = simple_json::Value;

namespace {

const char* kAppConfigPath = "./config.json";
const char* kAppConfigEnvVar = "MIX_CONFIG_PATH";
const char* kDefaultSampleConfigPath = "../../src/sample.json";

struct SampleMeta {
    string name;
    bool isMC = false;
    bool isSignal = false;
};

struct AppConfig {
    vector<string> treeNames;
    string runSample;
    int maxThreads = 1;
    string inputRoot;
    string inputPattern;
    string outputRoot;
    string outputPattern;
    string sampleConfigPath;
    uint64_t randomState = 42;
    vector<SampleMeta> samples;  // indexed by name via sampleByName
    unordered_map<string, size_t> sampleByName;
};

string timestamp() {
    time_t now = time(nullptr);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));
    return string(buf);
}

void logMessage(const string& msg) {
    cout << "[" << timestamp() << "] " << msg << endl;
}

string resolveConfigPath(const char* preferredPath, const char* envVar) {
    if (envVar != nullptr) {
        const char* envPath = getenv(envVar);
        if (envPath != nullptr && *envPath != '\0') {
            if (fs::exists(envPath)) {
                return envPath;
            }
            throw runtime_error(string("Cannot find config file from environment variable ") + envVar + ": " + envPath);
        }
    }
    if (fs::exists(preferredPath)) {
        return preferredPath;
    }
    throw runtime_error(string("Cannot find config file: ") + preferredPath);
}

string resolveReferencedPath(const string& baseConfigPath, const string& targetPath) {
    if (targetPath.empty()) {
        return targetPath;
    }
    const fs::path path(targetPath);
    if (path.is_absolute()) {
        return targetPath;
    }
    const fs::path baseDir = fs::path(baseConfigPath).parent_path();
    return fs::weakly_canonical(baseDir / path).string();
}

string replaceToken(const string& text, const string& token, const string& value) {
    string out = text;
    const string pattern = string("{") + token + "}";
    size_t pos = 0;
    while ((pos = out.find(pattern, pos)) != string::npos) {
        out.replace(pos, pattern.size(), value);
        pos += value.size();
    }
    return out;
}

string formatInputPattern(const string& pattern, const string& inputRoot,
                          const string& sampleGroup, const string& sampleName) {
    string out = replaceToken(pattern, "input_root", inputRoot);
    out = replaceToken(out, "sample_group", sampleGroup);
    out = replaceToken(out, "sample", sampleName);
    return out;
}

string formatOutputPattern(const string& pattern, const string& outputRoot,
                           const string& sampleGroup, const string& sampleName) {
    string out = replaceToken(pattern, "output_root", outputRoot);
    out = replaceToken(out, "sample_group", sampleGroup);
    out = replaceToken(out, "sample", sampleName);
    return out;
}

AppConfig loadAppConfig() {
    const string appConfigPath = resolveConfigPath(kAppConfigPath, kAppConfigEnvVar);
    const JsonValue payload = simple_json::parseFile(appConfigPath);

    AppConfig config;
    if (payload.contains("tree_name")) {
        const JsonValue& node = payload.at("tree_name");
        if (node.isArray()) {
            for (const auto& item : node.asArray()) {
                config.treeNames.push_back(item.asString());
            }
        } else if (node.isString()) {
            config.treeNames.push_back(node.asString());
        }
    }
    if (config.treeNames.empty()) {
        config.treeNames = {"fat2", "fat3"};
    }

    config.runSample = payload.getStringOr("run_sample", "");
    config.maxThreads = payload.getIntOr("max_threads", 1);
    config.inputRoot = payload.at("input_root").asString();
    config.inputPattern = payload.getStringOr(
        "input_pattern", "{input_root}/{sample_group}/{sample}.root");
    config.outputRoot = payload.getStringOr("output_root", config.inputRoot + "_mixed");
    config.outputPattern = payload.getStringOr(
        "output_pattern", "{output_root}/{sample_group}/{sample}.root");
    config.randomState = static_cast<uint64_t>(
        payload.getNumberOr("random_state", 42.L));

    config.sampleConfigPath = resolveReferencedPath(
        appConfigPath,
        payload.getStringOr("sample_config", kDefaultSampleConfigPath));

    const JsonValue sampleJson = simple_json::parseFile(config.sampleConfigPath);
    if (!sampleJson.contains("sample")) {
        throw runtime_error("sample_config missing top-level 'sample' array: " + config.sampleConfigPath);
    }
    for (const auto& node : sampleJson.at("sample").asArray()) {
        SampleMeta meta;
        meta.name = node.at("name").asString();
        meta.isMC = node.at("is_MC").asBool();
        meta.isSignal = node.at("is_signal").asBool();
        config.sampleByName[meta.name] = config.samples.size();
        config.samples.push_back(std::move(meta));
    }
    return config;
}

const SampleMeta& lookupSample(const AppConfig& config, const string& sampleName) {
    const auto it = config.sampleByName.find(sampleName);
    if (it == config.sampleByName.end()) {
        throw runtime_error("Unknown sample: " + sampleName);
    }
    return config.samples[it->second];
}

string sampleGroupFor(const SampleMeta& meta) {
    return meta.isSignal ? "signal" : "bkg";
}

// List input chunk files for a sample. Matches either `{sample}.root` (single-file output) or
// `{sample}_<digits>.root` (chunked output), sorted ascending by chunk index (single file first if
// present, otherwise chunks 0..N).
vector<string> listInputChunkFiles(const string& basePath) {
    const fs::path base(basePath);
    const fs::path dir = base.parent_path();
    const string stem = base.stem().string();
    const string extension = base.has_extension() ? base.extension().string() : ".root";

    if (!fs::exists(dir)) {
        throw runtime_error("Input directory does not exist: " + dir.string());
    }

    const regex chunkPattern("^" + stem + "_([0-9]+)\\" + extension + "$");

    vector<pair<long long, string>> indexed;  // chunk index (or -1 for single) → path
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const string fname = entry.path().filename().string();
        if (fname == stem + extension) {
            indexed.emplace_back(-1LL, entry.path().string());
            continue;
        }
        smatch m;
        if (regex_match(fname, m, chunkPattern)) {
            indexed.emplace_back(stoll(m[1]), entry.path().string());
        }
    }

    sort(indexed.begin(), indexed.end(),
         [](const auto& a, const auto& b) { return a.first < b.first; });

    vector<string> files;
    files.reserve(indexed.size());
    for (auto& kv : indexed) {
        files.push_back(std::move(kv.second));
    }
    return files;
}

// Mirror each input chunk path to an output path under outputRoot using the provided pattern.
// The filename (basename) is preserved to keep the 1:1 chunk correspondence explicit, while the
// directory is controlled by outputPattern.
vector<string> makeOutputPaths(const vector<string>& inputFiles,
                               const string& outputPattern,
                               const string& outputRoot,
                               const string& sampleGroup,
                               const string& sampleName) {
    const string baseOut = formatOutputPattern(outputPattern, outputRoot, sampleGroup, sampleName);
    const fs::path baseOutPath(baseOut);
    const fs::path outDir = baseOutPath.parent_path();

    vector<string> outPaths;
    outPaths.reserve(inputFiles.size());
    for (const auto& inPath : inputFiles) {
        const string filename = fs::path(inPath).filename().string();
        outPaths.push_back((outDir / filename).string());
    }
    return outPaths;
}

Long64_t entriesInFileTree(const string& filePath, const string& treeName) {
    unique_ptr<TFile> f(TFile::Open(filePath.c_str(), "READ"));
    if (!f || f->IsZombie()) {
        throw runtime_error("Cannot open input file: " + filePath);
    }
    TTree* t = dynamic_cast<TTree*>(f->Get(treeName.c_str()));
    if (t == nullptr) {
        throw runtime_error("Tree '" + treeName + "' not found in " + filePath);
    }
    return t->GetEntries();
}

void shuffleTreeIntoChunks(const vector<string>& inputFiles,
                           const vector<string>& outputFiles,
                           const string& treeName,
                           uint64_t seed,
                           const string& outputMode) {
    const size_t nChunks = inputFiles.size();
    if (outputFiles.size() != nChunks) {
        throw runtime_error("Output chunk count differs from input for tree " + treeName);
    }

    vector<Long64_t> chunkCounts(nChunks, 0);
    Long64_t total = 0;
    for (size_t i = 0; i < nChunks; ++i) {
        chunkCounts[i] = entriesInFileTree(inputFiles[i], treeName);
        total += chunkCounts[i];
    }
    logMessage("tree=" + treeName + ": total_entries=" + to_string(total) +
               " across " + to_string(nChunks) + " chunk(s)");

    // Permutation: perm[out_pos] = src_idx. We iterate output chunks in order and for each output
    // slot read the corresponding source entry from the TChain.
    vector<Long64_t> perm(static_cast<size_t>(total));
    for (Long64_t i = 0; i < total; ++i) perm[i] = i;
    mt19937_64 rng(seed);
    shuffle(perm.begin(), perm.end(), rng);

    // Open TChain once over all input files. ROOT handles branch address re-binding across
    // underlying tree boundaries when we call CopyAddresses on the output tree.
    TChain chain(treeName.c_str());
    for (const auto& p : inputFiles) chain.Add(p.c_str());
    chain.SetCacheSize(static_cast<Long64_t>(256) * 1024 * 1024);
    chain.AddBranchToCache("*", true);

    if (chain.LoadTree(0) < 0) {
        throw runtime_error("Failed to load first tree of chain for " + treeName);
    }

    Long64_t cursor = 0;
    for (size_t i = 0; i < nChunks; ++i) {
        const Long64_t count = chunkCounts[i];
        const fs::path outPath(outputFiles[i]);
        fs::create_directories(outPath.parent_path());

        // Use a structure source (re-opened for this output chunk) because a TChain's current
        // tree may change as we seek across files. Cloning from a stable standalone TTree yields
        // a well-formed empty output skeleton, into which we then route the chain's branches via
        // CopyAddresses — the same pattern used by selections/convert.
        unique_ptr<TFile> structureFile(TFile::Open(inputFiles[i].c_str(), "READ"));
        if (!structureFile || structureFile->IsZombie()) {
            throw runtime_error("Cannot open structure file: " + inputFiles[i]);
        }
        TTree* structureSrc = dynamic_cast<TTree*>(structureFile->Get(treeName.c_str()));
        if (structureSrc == nullptr) {
            throw runtime_error("Tree '" + treeName + "' missing in structure file " + inputFiles[i]);
        }

        unique_ptr<TFile> outFile(TFile::Open(outPath.c_str(), outputMode.c_str()));
        if (!outFile || outFile->IsZombie()) {
            throw runtime_error("Cannot open output file: " + outPath.string());
        }
        outFile->cd();

        TTree* outTree = structureSrc->CloneTree(0);
        outTree->SetDirectory(outFile.get());
        outTree->SetBasketSize("*", 32000);
        outTree->SetAutoFlush(0);
        outTree->SetAutoSave(0);

        // Route the chain's branches into outTree's buffers so chain.GetEntry + outTree->Fill
        // round-trips values without any per-branch bookkeeping.
        outTree->CopyAddresses(&chain);

        for (Long64_t k = 0; k < count; ++k) {
            const Long64_t srcIdx = perm[static_cast<size_t>(cursor + k)];
            if (chain.GetEntry(srcIdx) <= 0) {
                throw runtime_error("Failed to read entry " + to_string(srcIdx) +
                                    " of tree " + treeName);
            }
            outTree->Fill();
        }
        cursor += count;

        outFile->cd();
        outTree->Write("", TObject::kOverwrite);
        chain.ResetBranchAddresses();
        outTree->ResetBranchAddresses();
        outFile->Close();

        logMessage("tree=" + treeName + " chunk " + to_string(i) +
                   " wrote " + to_string(count) + " entries to " + outPath.string());
    }
}

void processSample(const AppConfig& config, const string& sampleName) {
    const SampleMeta& meta = lookupSample(config, sampleName);
    const string sampleGroup = sampleGroupFor(meta);

    const string basePath = formatInputPattern(
        config.inputPattern, config.inputRoot, sampleGroup, sampleName);
    const vector<string> inputFiles = listInputChunkFiles(basePath);
    if (inputFiles.empty()) {
        logMessage("no input files for sample=" + sampleName + " (base=" + basePath + "), skipping");
        return;
    }
    const vector<string> outputFiles = makeOutputPaths(
        inputFiles, config.outputPattern, config.outputRoot, sampleGroup, sampleName);

    logMessage("sample=" + sampleName + " is_MC=" + (meta.isMC ? "true" : "false") +
               " group=" + sampleGroup + " chunks=" + to_string(inputFiles.size()));
    for (size_t i = 0; i < inputFiles.size(); ++i) {
        logMessage("  input[" + to_string(i) + "] = " + inputFiles[i]);
        logMessage("  output[" + to_string(i) + "] = " + outputFiles[i]);
    }

    for (size_t t = 0; t < config.treeNames.size(); ++t) {
        const string& treeName = config.treeNames[t];
        const string mode = (t == 0) ? "RECREATE" : "UPDATE";
        const uint64_t seed = config.randomState + static_cast<uint64_t>(t) * 1315423911ULL;
        shuffleTreeIntoChunks(inputFiles, outputFiles, treeName, seed, mode);
    }

    logMessage("sample=" + sampleName + " done");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const AppConfig config = loadAppConfig();

        if (config.maxThreads > 1) {
            ROOT::EnableImplicitMT(static_cast<unsigned>(config.maxThreads));
        }

        string sampleName;
        if (argc >= 2 && argv[1] != nullptr && *argv[1] != '\0') {
            sampleName = argv[1];
        } else {
            sampleName = config.runSample;
        }
        if (sampleName.empty()) {
            cerr << "mix: no sample specified (pass as argv[1] or set run_sample in config)" << endl;
            return 2;
        }

        logMessage("mix starting: sample=" + sampleName +
                   " random_state=" + to_string(config.randomState) +
                   " max_threads=" + to_string(config.maxThreads));
        processSample(config, sampleName);
        return 0;
    } catch (const std::exception& ex) {
        cerr << "mix error: " << ex.what() << endl;
        return 1;
    }
}
