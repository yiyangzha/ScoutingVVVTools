// Summary: Shuffle all entries of a sample's trees across its ROOT chunk files, preserving the
// per-chunk entry counts so that the output mirrors the input chunk layout. Intended to be run
// after selections/convert so that downstream BDT train/test splits by entry index are drawn from
// a randomised ordering rather than any intrinsic file ordering. The implementation favours
// sequential ROOT reads by shuffling contiguous entry blocks rather than issuing a fully random
// entry-by-entry read pattern.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <TChain.h>
#include <TFile.h>
#include <TTree.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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
    Long64_t minBlockEntries = 32;
    Long64_t maxBlockEntries = 4096;
    vector<SampleMeta> samples;  // indexed by name via sampleByName
    unordered_map<string, size_t> sampleByName;
};

struct ShuffleBlock {
    Long64_t start = 0;
    Long64_t count = 0;
    Long64_t rotation = 0;
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
    config.minBlockEntries = static_cast<Long64_t>(
        payload.getNumberOr("min_block_entries", 32.L));
    config.maxBlockEntries = static_cast<Long64_t>(
        payload.getNumberOr("max_block_entries", 4096.L));
    if (config.minBlockEntries <= 0) {
        throw runtime_error("min_block_entries must be positive");
    }
    if (config.maxBlockEntries <= 0) {
        throw runtime_error("max_block_entries must be positive");
    }
    if (config.maxBlockEntries < config.minBlockEntries) {
        throw runtime_error("max_block_entries must be >= min_block_entries");
    }

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

void scanSingleInputChunkEntries(const string& inputFile,
                                 const vector<string>& treeNames,
                                 vector<Long64_t>& counts) {
    unique_ptr<TFile> f(TFile::Open(inputFile.c_str(), "READ"));
    if (!f || f->IsZombie()) {
        throw runtime_error("Cannot open input file: " + inputFile);
    }
    for (size_t t = 0; t < treeNames.size(); ++t) {
        TTree* tree = dynamic_cast<TTree*>(f->Get(treeNames[t].c_str()));
        if (tree == nullptr) {
            throw runtime_error("Tree '" + treeNames[t] + "' not found in " + inputFile);
        }
        counts[t] = tree->GetEntries();
    }
}

vector<vector<Long64_t>> scanInputChunkEntries(const vector<string>& inputFiles,
                                               const vector<string>& treeNames) {
    vector<vector<Long64_t>> counts(
        inputFiles.size(), vector<Long64_t>(treeNames.size(), 0));
    if (inputFiles.empty()) {
        return counts;
    }

    // Warm up ROOT I/O serially before parallel file scans so that global startup work does not
    // race across threads.
    scanSingleInputChunkEntries(inputFiles.front(), treeNames, counts.front());
    if (inputFiles.size() == 1) {
        return counts;
    }

    exception_ptr firstError = nullptr;
    mutex errorMutex;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(inputFiles.size() > 2)
#endif
    for (int i = 1; i < static_cast<int>(inputFiles.size()); ++i) {
        try {
            scanSingleInputChunkEntries(
                inputFiles[static_cast<size_t>(i)],
                treeNames,
                counts[static_cast<size_t>(i)]);
        } catch (...) {
            lock_guard<mutex> lock(errorMutex);
            if (!firstError) {
                firstError = current_exception();
            }
        }
    }

    if (firstError) {
        rethrow_exception(firstError);
    }
    return counts;
}

Long64_t chooseShuffleBlockEntries(Long64_t totalEntries,
                                   Long64_t minBlockEntries,
                                   Long64_t maxBlockEntries) {
    const Long64_t kTargetBlocks = 512;

    if (totalEntries <= 0) {
        return minBlockEntries;
    }

    Long64_t blockEntries = totalEntries / kTargetBlocks;
    if (blockEntries < minBlockEntries) {
        blockEntries = minBlockEntries;
    }
    if (blockEntries > maxBlockEntries) {
        blockEntries = maxBlockEntries;
    }
    return blockEntries;
}

vector<ShuffleBlock> buildShuffleBlocks(Long64_t totalEntries,
                                        uint64_t seed,
                                        Long64_t minBlockEntries,
                                        Long64_t maxBlockEntries) {
    const Long64_t blockEntries = chooseShuffleBlockEntries(
        totalEntries, minBlockEntries, maxBlockEntries);
    vector<ShuffleBlock> blocks;
    for (Long64_t start = 0; start < totalEntries; start += blockEntries) {
        ShuffleBlock block;
        block.start = start;
        block.count = min(blockEntries, totalEntries - start);
        blocks.push_back(block);
    }

    mt19937_64 rng(seed);
    shuffle(blocks.begin(), blocks.end(), rng);
    for (auto& block : blocks) {
        if (block.count > 1) {
            uniform_int_distribution<Long64_t> dist(0, block.count - 1);
            block.rotation = dist(rng);
        }
    }
    return blocks;
}

template <typename Func>
void forEachRotatedBlockSegment(const ShuffleBlock& block, Func&& func) {
    if (block.count <= 0) {
        return;
    }

    const Long64_t rotation = block.rotation % block.count;
    const Long64_t firstStart = block.start + rotation;
    const Long64_t firstCount = block.count - rotation;
    if (firstCount > 0) {
        func(firstStart, firstCount);
    }
    if (rotation > 0) {
        func(block.start, rotation);
    }
}

unique_ptr<TFile> openStructureFileWithTree(const vector<string>& inputFiles,
                                            const string& treeName,
                                            TTree*& structureTree) {
    for (const auto& inputFile : inputFiles) {
        unique_ptr<TFile> structureFile(TFile::Open(inputFile.c_str(), "READ"));
        if (!structureFile || structureFile->IsZombie()) {
            continue;
        }
        TTree* tree = dynamic_cast<TTree*>(structureFile->Get(treeName.c_str()));
        if (tree != nullptr) {
            structureTree = tree;
            return structureFile;
        }
    }
    throw runtime_error("Failed to find structure tree '" + treeName + "' in input files");
}

void writeChunkTree(TFile& outFile,
                    TTree& structureTree,
                    const string& treeName) {
    outFile.cd();

    TTree* outTree = structureTree.CloneTree(0);
    if (outTree == nullptr) {
        throw runtime_error("Failed to clone output tree structure for " + treeName);
    }
    outTree->SetDirectory(&outFile);
    outTree->SetBasketSize("*", 32000);
    outTree->SetAutoFlush(0);
    outTree->SetAutoSave(0);

    outFile.cd();
    outTree->Write("", TObject::kOverwrite);
    outTree->ResetBranchAddresses();
}

void shuffleTreeIntoChunks(const vector<string>& inputFiles,
                           const vector<string>& outputFiles,
                           const vector<Long64_t>& chunkCounts,
                           const string& treeName,
                           uint64_t seed,
                           const string& outputMode,
                           Long64_t minBlockEntries,
                           Long64_t maxBlockEntries) {
    const size_t nChunks = inputFiles.size();
    if (outputFiles.size() != nChunks || chunkCounts.size() != nChunks) {
        throw runtime_error("Chunk metadata size mismatch for tree " + treeName);
    }

    Long64_t total = 0;
    for (Long64_t count : chunkCounts) {
        total += count;
    }
    logMessage("tree=" + treeName + ": total_entries=" + to_string(total) +
               " across " + to_string(nChunks) + " chunk(s)");

    TTree* structureSrc = nullptr;
    unique_ptr<TFile> structureFile = openStructureFileWithTree(inputFiles, treeName, structureSrc);

    if (total == 0) {
        for (size_t i = 0; i < nChunks; ++i) {
            const fs::path outPath(outputFiles[i]);
            fs::create_directories(outPath.parent_path());
            unique_ptr<TFile> outFile(TFile::Open(outPath.c_str(), outputMode.c_str()));
            if (!outFile || outFile->IsZombie()) {
                throw runtime_error("Cannot open output file: " + outPath.string());
            }
            writeChunkTree(*outFile, *structureSrc, treeName);
            outFile->Close();
            logMessage("tree=" + treeName + " chunk " + to_string(i) +
                       " wrote 0 entries to " + outPath.string());
        }
        return;
    }

    const vector<ShuffleBlock> blocks = buildShuffleBlocks(
        total, seed, minBlockEntries, maxBlockEntries);
    logMessage("tree=" + treeName + ": shuffle_blocks=" + to_string(blocks.size()) +
               " block_entries~" + to_string(chooseShuffleBlockEntries(
                   total, minBlockEntries, maxBlockEntries)) +
               " block_range=[" + to_string(minBlockEntries) +
               "," + to_string(maxBlockEntries) + "]");

    // Open TChain once over all input files. Each non-empty output chunk clones
    // its tree from the chain itself so ROOT keeps the clone's branch addresses
    // synced as the chain advances across input-file boundaries.
    TChain chain(treeName.c_str());
    for (const auto& p : inputFiles) chain.Add(p.c_str());
    chain.SetCacheSize(static_cast<Long64_t>(256) * 1024 * 1024);
    chain.AddBranchToCache("*", true);
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 0, 0)
    chain.SetClusterPrefetch(true);
#endif

    if (chain.LoadTree(0) < 0) {
        throw runtime_error("Failed to load first tree of chain for " + treeName);
    }

    size_t chunkIndex = 0;
    Long64_t writtenInChunk = 0;
    Long64_t totalWritten = 0;
    unique_ptr<TFile> outFile;
    TTree* outTree = nullptr;

    auto closeCurrentChunk = [&]() {
        if (!outFile || outTree == nullptr) {
            return;
        }
        outFile->cd();
        outTree->Write("", TObject::kOverwrite);
        chain.ResetBranchAddresses();
        outTree->ResetBranchAddresses();
        outFile->Close();
        logMessage("tree=" + treeName + " chunk " + to_string(chunkIndex) +
                   " wrote " + to_string(chunkCounts[chunkIndex]) +
                   " entries to " + outputFiles[chunkIndex]);
        outTree = nullptr;
        outFile.reset();
        ++chunkIndex;
        writtenInChunk = 0;
    };

    auto openNextChunk = [&]() {
        if (chunkIndex >= nChunks) {
            throw runtime_error("Tried to open output chunk past the end for tree " + treeName);
        }
        const fs::path outPath(outputFiles[chunkIndex]);
        fs::create_directories(outPath.parent_path());

        outFile.reset(TFile::Open(outPath.c_str(), outputMode.c_str()));
        if (!outFile || outFile->IsZombie()) {
            throw runtime_error("Cannot open output file: " + outPath.string());
        }
        outFile->cd();

        TTree* outTree = chain.CloneTree(0);
        if (outTree == nullptr) {
            throw runtime_error("Failed to clone output tree from chain for " + treeName);
        }
        outTree->SetDirectory(outFile.get());
        outTree->SetBasketSize("*", 32000);
        outTree->SetAutoFlush(0);
        outTree->SetAutoSave(0);
        return outTree;
    };

    auto ensureWritableChunk = [&]() {
        while (chunkIndex < nChunks && chunkCounts[chunkIndex] == 0) {
            const fs::path outPath(outputFiles[chunkIndex]);
            fs::create_directories(outPath.parent_path());
            unique_ptr<TFile> emptyFile(TFile::Open(outPath.c_str(), outputMode.c_str()));
            if (!emptyFile || emptyFile->IsZombie()) {
                throw runtime_error("Cannot open output file: " + outPath.string());
            }
            writeChunkTree(*emptyFile, *structureSrc, treeName);
            emptyFile->Close();
            logMessage("tree=" + treeName + " chunk " + to_string(chunkIndex) +
                       " wrote 0 entries to " + outPath.string());
            ++chunkIndex;
        }
        if (chunkIndex < nChunks && outTree == nullptr) {
            outTree = openNextChunk();
        }
    };

    ensureWritableChunk();
    for (const auto& block : blocks) {
        forEachRotatedBlockSegment(block, [&](Long64_t segmentStart, Long64_t segmentCount) {
            for (Long64_t srcIdx = segmentStart; srcIdx < segmentStart + segmentCount; ++srcIdx) {
                ensureWritableChunk();
                if (chunkIndex >= nChunks || outTree == nullptr) {
                    throw runtime_error("Ran out of output chunks while writing tree " + treeName);
                }
                if (chain.GetEntry(srcIdx) <= 0) {
                    throw runtime_error("Failed to read entry " + to_string(srcIdx) +
                                        " of tree " + treeName);
                }
                outTree->Fill();
                ++writtenInChunk;
                ++totalWritten;
                if (writtenInChunk == chunkCounts[chunkIndex]) {
                    closeCurrentChunk();
                }
            }
        });
    }

    if (outTree != nullptr) {
        if (writtenInChunk != chunkCounts[chunkIndex]) {
            throw runtime_error("Output chunk entry count mismatch for tree " + treeName);
        }
        closeCurrentChunk();
    }
    while (chunkIndex < nChunks) {
        ensureWritableChunk();
        if (chunkIndex < nChunks) {
            throw runtime_error("Unwritten output chunk remains for tree " + treeName);
        }
    }
    if (totalWritten != total) {
        throw runtime_error("Total written entries mismatch for tree " + treeName +
                            ": expected " + to_string(total) +
                            ", got " + to_string(totalWritten));
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
    const vector<vector<Long64_t>> inputChunkEntries =
        scanInputChunkEntries(inputFiles, config.treeNames);

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
        vector<Long64_t> chunkCounts(inputFiles.size(), 0);
        for (size_t i = 0; i < inputFiles.size(); ++i) {
            chunkCounts[i] = inputChunkEntries[i][t];
        }
        shuffleTreeIntoChunks(
            inputFiles,
            outputFiles,
            chunkCounts,
            treeName,
            seed,
            mode,
            config.minBlockEntries,
            config.maxBlockEntries);
    }

    logMessage("sample=" + sampleName + " done");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const AppConfig config = loadAppConfig();

#ifdef _OPENMP
        if (config.maxThreads > 0) {
            omp_set_num_threads(config.maxThreads);
        }
#endif

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
                   " block_range=[" + to_string(config.minBlockEntries) +
                   "," + to_string(config.maxBlockEntries) + "]" +
                   " max_threads=" + to_string(config.maxThreads));
        processSample(config, sampleName);
        return 0;
    } catch (const std::exception& ex) {
        cerr << "mix error: " << ex.what() << endl;
        return 1;
    }
}
