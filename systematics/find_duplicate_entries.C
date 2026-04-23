// Summary: Scan one ROOT file or a directory of ROOT files and report duplicate
// (run, luminosityBlock, event) entries, chaining matching trees together.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "TClass.h"
#include "TChain.h"
#include "TFile.h"
#include "TKey.h"
#include "TTree.h"

using namespace std;
namespace fs = std::filesystem;

namespace {

struct EventKey {
    UInt_t run = 0;
    UInt_t lumi = 0;
    ULong64_t event = 0;

    bool operator==(const EventKey& other) const {
        return run == other.run && lumi == other.lumi && event == other.event;
    }
};

struct EventKeyHash {
    size_t operator()(const EventKey& key) const {
        size_t seed = static_cast<size_t>(key.run);
        seed ^= static_cast<size_t>(key.lumi) + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        seed ^= static_cast<size_t>(key.event) + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

struct SeenInfo {
    Long64_t firstEntry = -1;
    Long64_t lastEntry = -1;
    Long64_t count = 0;
};

struct DuplicateRecord {
    EventKey key;
    SeenInfo info;
};

struct ScanSummary {
    Long64_t totalEntries = 0;
    Long64_t uniqueKeys = 0;
    Long64_t duplicateKeys = 0;
    Long64_t duplicateEntries = 0;
    Long64_t maxMultiplicity = 0;
    vector<DuplicateRecord> records;
};

void printUsage(const char* argv0) {
    cerr << "Usage: " << argv0 << " <file.root|directory> [tree_name] [max_report]" << endl;
    cerr << "  <file.root|directory>  ROOT file to scan, or a directory searched recursively for *.root." << endl;
    cerr << "  [tree_name]   Optional TTree name. If omitted, scan all top-level TTrees." << endl;
    cerr << "  [max_report]  Optional maximum number of duplicate keys to print (default: 20)." << endl;
}

bool hasEventIdBranches(TTree& tree) {
    return tree.GetBranch("run") != nullptr &&
           tree.GetBranch("luminosityBlock") != nullptr &&
           tree.GetBranch("event") != nullptr;
}

vector<string> discoverTreeNames(TFile& file, const string& requestedTree) {
    vector<string> treeNames;

    if (!requestedTree.empty()) {
        TObject* obj = file.Get(requestedTree.c_str());
        if (obj == nullptr || !obj->InheritsFrom(TTree::Class())) {
            throw runtime_error("Requested tree '" + requestedTree + "' was not found in " + file.GetName());
        }
        treeNames.push_back(requestedTree);
        return treeNames;
    }

    TIter next(file.GetListOfKeys());
    while (TKey* key = static_cast<TKey*>(next())) {
        TClass* cls = TClass::GetClass(key->GetClassName());
        if (cls != nullptr && cls->InheritsFrom(TTree::Class())) {
            treeNames.push_back(key->GetName());
        }
    }

    sort(treeNames.begin(), treeNames.end());
    treeNames.erase(unique(treeNames.begin(), treeNames.end()), treeNames.end());
    return treeNames;
}

bool isRootFilePath(const fs::path& path) {
    return path.has_extension() && path.extension() == ".root";
}

vector<string> collectRootFiles(const string& inputPath) {
    const fs::path path(inputPath);
    if (!fs::exists(path)) {
        throw runtime_error("Input path does not exist: " + inputPath);
    }

    vector<string> rootFiles;
    if (fs::is_regular_file(path)) {
        if (!isRootFilePath(path)) {
            throw runtime_error("Input file is not a ROOT file: " + inputPath);
        }
        rootFiles.push_back(fs::absolute(path).lexically_normal().string());
        return rootFiles;
    }

    if (!fs::is_directory(path)) {
        throw runtime_error("Input path is neither a ROOT file nor a directory: " + inputPath);
    }

    for (const auto& entry : fs::recursive_directory_iterator(path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (!isRootFilePath(entry.path())) {
            continue;
        }
        rootFiles.push_back(fs::absolute(entry.path()).lexically_normal().string());
    }

    sort(rootFiles.begin(), rootFiles.end());
    rootFiles.erase(unique(rootFiles.begin(), rootFiles.end()), rootFiles.end());
    if (rootFiles.empty()) {
        throw runtime_error("No ROOT files found under directory: " + inputPath);
    }
    return rootFiles;
}

vector<string> discoverTreeNames(const vector<string>& rootFiles, const string& requestedTree) {
    if (!requestedTree.empty()) {
        return {requestedTree};
    }

    for (const string& rootFile : rootFiles) {
        unique_ptr<TFile> file(TFile::Open(rootFile.c_str(), "READ"));
        if (!file || file->IsZombie()) {
            continue;
        }
        const vector<string> treeNames = discoverTreeNames(*file, "");
        if (!treeNames.empty()) {
            return treeNames;
        }
    }

    throw runtime_error("No top-level TTrees found in the collected ROOT files");
}

unique_ptr<TChain> buildChain(const vector<string>& rootFiles,
                              const string& treeName,
                              size_t& addedFiles) {
    unique_ptr<TChain> chain = make_unique<TChain>(treeName.c_str());
    addedFiles = 0;

    for (const string& rootFile : rootFiles) {
        unique_ptr<TFile> file(TFile::Open(rootFile.c_str(), "READ"));
        if (!file || file->IsZombie()) {
            continue;
        }
        TObject* obj = file->Get(treeName.c_str());
        if (obj == nullptr || !obj->InheritsFrom(TTree::Class())) {
            continue;
        }
        chain->Add(rootFile.c_str());
        ++addedFiles;
    }

    if (addedFiles == 0) {
        throw runtime_error("Tree '" + treeName + "' was not found in any collected ROOT file");
    }

    return chain;
}

ScanSummary scanTree(TTree& tree) {
    const Long64_t totalEntries = tree.GetEntries();
    if (totalEntries > 0) {
        tree.LoadTree(0);
    }

    if (!hasEventIdBranches(tree)) {
        throw runtime_error("Tree '" + string(tree.GetName()) +
                            "' does not contain all of: run, luminosityBlock, event");
    }

    UInt_t run = 0;
    UInt_t lumi = 0;
    ULong64_t event = 0;

    tree.SetBranchStatus("*", 0);
    tree.SetBranchStatus("run", 1);
    tree.SetBranchStatus("luminosityBlock", 1);
    tree.SetBranchStatus("event", 1);
    tree.SetBranchAddress("run", &run);
    tree.SetBranchAddress("luminosityBlock", &lumi);
    tree.SetBranchAddress("event", &event);

    unordered_map<EventKey, SeenInfo, EventKeyHash> seen;
    if (totalEntries > 0 && totalEntries < 50000000LL) {
        seen.reserve(static_cast<size_t>(totalEntries * 1.3));
    }

    for (Long64_t entry = 0; entry < totalEntries; ++entry) {
        tree.GetEntry(entry);
        const EventKey key{run, lumi, event};
        SeenInfo& info = seen[key];
        if (info.count == 0) {
            info.firstEntry = entry;
        }
        info.lastEntry = entry;
        ++info.count;
    }

    tree.ResetBranchAddresses();

    ScanSummary summary;
    summary.totalEntries = totalEntries;
    summary.uniqueKeys = static_cast<Long64_t>(seen.size());

    for (const auto& kv : seen) {
        const SeenInfo& info = kv.second;
        if (info.count <= 1) {
            continue;
        }
        summary.duplicateKeys += 1;
        summary.duplicateEntries += (info.count - 1);
        summary.maxMultiplicity = max(summary.maxMultiplicity, info.count);
        summary.records.push_back(DuplicateRecord{kv.first, info});
    }

    sort(summary.records.begin(), summary.records.end(),
         [](const DuplicateRecord& a, const DuplicateRecord& b) {
             if (a.info.count != b.info.count) {
                 return a.info.count > b.info.count;
             }
             if (a.key.run != b.key.run) {
                 return a.key.run < b.key.run;
             }
             if (a.key.lumi != b.key.lumi) {
                 return a.key.lumi < b.key.lumi;
             }
             return a.key.event < b.key.event;
         });

    return summary;
}

void printSummary(const string& treeName,
                  size_t filesInChain,
                  const ScanSummary& summary,
                  size_t maxReport) {
    cout << "[TREE] " << treeName << endl;
    cout << "  files_in_chain   = " << filesInChain << endl;
    cout << "  total_entries    = " << summary.totalEntries << endl;
    cout << "  unique_keys      = " << summary.uniqueKeys << endl;
    cout << "  duplicate_keys   = " << summary.duplicateKeys << endl;
    cout << "  duplicate_entries= " << summary.duplicateEntries << endl;
    cout << "  max_multiplicity = " << summary.maxMultiplicity << endl;

    if (summary.records.empty()) {
        cout << "  status           = no duplicate (run, luminosityBlock, event) keys found" << endl;
        return;
    }

    cout << "  status           = duplicates found" << endl;
    const size_t nReport = min(maxReport, summary.records.size());
    cout << "  showing_top      = " << nReport << endl;
    for (size_t i = 0; i < nReport; ++i) {
        const DuplicateRecord& rec = summary.records[i];
        cout << "    [" << (i + 1) << "] run=" << rec.key.run
             << " lumi=" << rec.key.lumi
             << " event=" << rec.key.event
             << " count=" << rec.info.count
             << " first_entry=" << rec.info.firstEntry
             << " last_entry=" << rec.info.lastEntry
             << endl;
    }
}

}  // namespace

int find_duplicate_entries() {
    const string filePath = "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/ScoutingVVVTools/dataset/signal/";
    const string requestedTree = "fat2";
    const size_t maxReport = 20U;

    try {
        const vector<string> rootFiles = collectRootFiles(filePath);
        const vector<string> treeNames = discoverTreeNames(rootFiles, requestedTree);
        if (treeNames.empty()) {
            throw runtime_error("No top-level TTrees found in " + filePath);
        }

        cout << "Scanning input path: " << filePath << endl;
        cout << "Collected ROOT files: " << rootFiles.size() << endl;
        cout << "Trees to scan: " << treeNames.size() << endl;

        bool scannedAnyTree = false;
        for (const string& treeName : treeNames) {
            size_t addedFiles = 0;
            unique_ptr<TChain> chain = buildChain(rootFiles, treeName, addedFiles);
            const Long64_t chainEntries = chain->GetEntries();
            if (chainEntries > 0) {
                chain->LoadTree(0);
            }

            if (!hasEventIdBranches(*chain)) {
                if (!requestedTree.empty()) {
                    throw runtime_error("Tree '" + treeName +
                                        "' does not contain all of: run, luminosityBlock, event");
                }
                cout << "[TREE] " << treeName
                     << "\n  files_in_chain   = " << addedFiles
                     << "\n  status           = skipped (missing run/luminosityBlock/event branches)" << endl;
                continue;
            }

            const ScanSummary summary = scanTree(*chain);
            printSummary(treeName, addedFiles, summary, maxReport);
            scannedAnyTree = true;
        }

        if (!scannedAnyTree) {
            throw runtime_error("No scannable TTrees found in " + filePath);
        }
    } catch (const exception& ex) {
        cerr << "find_duplicate_entries error: " << ex.what() << endl;
        return 1;
    }

    return 0;
}
