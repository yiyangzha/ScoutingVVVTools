// Summary: Scan a ROOT file and report duplicate (run, luminosityBlock, event) entries.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "TClass.h"
#include "TFile.h"
#include "TKey.h"
#include "TTree.h"

using namespace std;

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
    cerr << "Usage: " << argv0 << " <file.root> [tree_name] [max_report]" << endl;
    cerr << "  <file.root>   ROOT file to scan." << endl;
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

ScanSummary scanTree(TTree& tree) {
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

    const Long64_t totalEntries = tree.GetEntries();
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

void printSummary(const string& treeName, const ScanSummary& summary, size_t maxReport) {
    cout << "[TREE] " << treeName << endl;
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
    const string filePath = "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/ScoutingVVVTools/dataset/signal/wplush.root";
    const string requestedTree = "fat2";
    const size_t maxReport = 20U;

    try {
        unique_ptr<TFile> file(TFile::Open(filePath.c_str(), "READ"));
        if (!file || file->IsZombie()) {
            throw runtime_error("Cannot open ROOT file: " + filePath);
        }

        const vector<string> treeNames = discoverTreeNames(*file, requestedTree);
        if (treeNames.empty()) {
            throw runtime_error("No top-level TTrees found in " + filePath);
        }

        cout << "Scanning file: " << filePath << endl;
        cout << "Trees to scan: " << treeNames.size() << endl;

        bool scannedAnyTree = false;
        for (const string& treeName : treeNames) {
            TTree* tree = dynamic_cast<TTree*>(file->Get(treeName.c_str()));
            if (tree == nullptr) {
                continue;
            }

            if (!hasEventIdBranches(*tree)) {
                if (!requestedTree.empty()) {
                    throw runtime_error("Tree '" + treeName +
                                        "' does not contain all of: run, luminosityBlock, event");
                }
                cout << "[TREE] " << treeName
                     << "\n  status           = skipped (missing run/luminosityBlock/event branches)" << endl;
                continue;
            }

            const ScanSummary summary = scanTree(*tree);
            printSummary(treeName, summary, maxReport);
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
