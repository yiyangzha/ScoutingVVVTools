// Summary: Skim data ROOT files from an XRootD list and merge outputs.
#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <TTreeFormula.h>
#include <TFileMerger.h>
#include <TH1D.h>

#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>
#include <regex>
#include <iostream>
#include <fstream>
#include <unistd.h>     // for getcwd()
#include <sys/stat.h>   // for struct stat

// Thread-safe printing
std::mutex printMutex;
std::string tempDirName;

// File list with one full XRootD path per line
const std::string listFile    = "samples/2024H.txt";
const std::string outputFile  = "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024H.root";

const std::vector<std::string> branchPatterns = {          // Branch patterns to keep
    "nScouting*",
    "nGenJetAK8",
    "nGenPart",
    "DST_PFScouting_*",
    "ScoutingPFJetRecluster_*",
    "ScoutingFatPFJetRecluster_*",
    "ScoutingElectron_*",
    "ScoutingMuonVtx_*",
    "ScoutingMuonNoVtx_*"
};
const std::string cutString = "nScoutingFatPFJetRecluster > 0";  // Event selection

void printProgressBar(int threadId, int fileIdx, int nFiles, Long64_t entryIdx, Long64_t nEntries) {
    const int barWidth = 50;
    double fraction = double(entryIdx) / double(nEntries);
    int pos = int(barWidth * fraction);
    int percent = int(fraction * 100);
    std::string bar(barWidth, ' ');
    for (int i = 0; i < barWidth; ++i) {
        if      (i < pos)      bar[i] = '=';
        else if (i == pos)     bar[i] = '>';
    }
    std::lock_guard<std::mutex> lk(printMutex);
    std::cout << "[T" << threadId << "] File " << (fileIdx+1) << "/" << nFiles
              << " [" << bar << "] " << percent << "% "
              << "(" << entryIdx << "/" << nEntries << ")\r"
              << std::flush;
}

// Process one file
void processFile(int idx,
                 const std::vector<std::string>& fileList,
                 const std::vector<std::regex>& patterns,
                 const std::string& cutString,
                 std::vector<std::string>& tempFiles)
{
    int threadId = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 1000;
    const std::string& inPath = fileList[idx];
    std::string tempPath = tempDirName + "/temp_" + std::to_string(idx) + ".root";
    tempFiles[idx] = tempPath;

    {
        std::lock_guard<std::mutex> lk(printMutex);
        std::cout << "[T" << threadId << "] Start " << inPath << std::endl;
    }

    try {
        TFile* inF = TFile::Open(inPath.c_str(), "READ");
        if (!inF || inF->IsZombie()) return;
        TTree* tree = (TTree*)inF->Get("Events");
        if (!tree) { inF->Close(); return; }

        // Select branches.
        auto all = tree->GetListOfBranches();
        std::vector<std::string> sel;
        for (auto* obj : *all) {
            std::string b = obj->GetName();
            for (auto& re : patterns) {
                if (std::regex_match(b, re)) {
                    sel.push_back(b);
                    break;
                }
            }
        }
        if (sel.empty()) { inF->Close(); return; }

        tree->SetBranchStatus("*", 0);
        for (auto& b : sel) tree->SetBranchStatus(b.c_str(), 1);

        TFile* outF = TFile::Open(tempPath.c_str(), "RECREATE");
        outF->SetCompressionSettings(
            100 * ROOT::RCompressionSetting::EAlgorithm::kLZMA
            + 4
        );
        TTree* outT = tree->CloneTree(0);

        TTreeFormula formula("cut", cutString.c_str(), tree);
        Long64_t nEntries = tree->GetEntries();
        Long64_t interval = std::max<Long64_t>(1, nEntries/100);

        for (Long64_t i = 1; i <= nEntries; ++i) {
            tree->GetEntry(i-1);
            if (formula.EvalInstance()) outT->Fill();
            // Uncomment below for a progress bar.
            // if (i % interval == 0 || i==nEntries) printProgressBar(threadId, idx, fileList.size(), i, nEntries);
        }
        {
            std::lock_guard<std::mutex> lk(printMutex);
            std::cout << std::endl;
        }

        outF->cd();
        outT->Write();
        outF->Close();
        inF->Close();
    }
    catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(printMutex);
        std::cerr << "[T" << threadId << "] Error processing " << inPath
                  << ": " << e.what() << std::endl;
        return;
    }

    {
        std::lock_guard<std::mutex> lk(printMutex);
        std::cout << "[T" << threadId << "] Done " << inPath << std::endl;
    }
}

int skim_data() {
    // VOMS proxy setup.
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        std::cerr << "Error: cannot get CWD for proxy setup\n";
        return 1;
    }
    std::string proxy = std::string(cwd) + "/x509up";
    gSystem->Setenv("X509_USER_PROXY", proxy.c_str());
    std::cout << "Using VOMS proxy: " << proxy << std::endl;

    // Read the file list.
    std::vector<std::string> fileList;
    std::ifstream fin(listFile);
    if (!fin) {
        std::cerr << "Error: cannot open list file " << listFile << std::endl;
        return 1;
    }
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        fileList.push_back(line);
    }
    fin.close();

    if (fileList.empty()) {
        std::cerr << "No files listed in " << listFile << std::endl;
        return 1;
    }
    int nFiles = fileList.size();
    std::cout << "Found " << nFiles << " ROOT files from " << listFile << "\n";

    // Create temp directory.
    {
        std::string base = outputFile.substr(outputFile.find_last_of("/")+1);
        base = base.substr(0, base.find_last_of("."));
        tempDirName = base;
        gSystem->mkdir(tempDirName.c_str(), kTRUE);
    }

    // Count total raw entries.
    Long64_t totalRawEntries = 0;
    for (const auto& p : fileList) {
        TFile* f = TFile::Open(p.c_str(), "READ");
        if (!f || f->IsZombie()) continue;
        TTree* t = (TTree*)f->Get("Events");
        if (t) totalRawEntries += t->GetEntries();
        f->Close();
    }
    std::cout << "Total raw entries across all files: " << totalRawEntries << "\n";

    // Compile wildcards into regex.
    std::vector<std::regex> patterns;
    patterns.reserve(branchPatterns.size());
    for (auto& p : branchPatterns) {
        std::string r = "^" + std::regex_replace(p, std::regex(R"(\*)"), ".*") + "$";
        patterns.emplace_back(r);
    }

    // Start thread pool.
    std::atomic<int> idx{0};
    unsigned nThreads = std::min<unsigned>(
        std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4,
        nFiles
    );
    // Set to 1 for debugging.
    // nThreads = 1;
    std::vector<std::thread> pool;
    std::vector<std::string> tempFiles(nFiles);
    for (unsigned t = 0; t < nThreads; ++t) {
        pool.emplace_back([&](){
            while (true) {
                int i = idx.fetch_add(1);
                if (i >= nFiles) break;
                processFile(i, fileList, patterns, cutString, tempFiles);
            }
        });
    }
    for (auto& th : pool) th.join();

    // Merge in 50 GB chunks.
    std::cout << "Merging into chunks of up to 50GB...\n";
    const long long maxChunkSize = 50LL * 1024 * 1024 * 1024; // 50 GB
    std::string outDir  = outputFile.substr(0, outputFile.find_last_of("/"));
    std::string outBase = outputFile.substr(outputFile.find_last_of("/")+1);
    std::string baseName = outBase.substr(0, outBase.find_last_of("."));
    std::string ext      = outBase.substr(outBase.find_last_of(".")); // ".root"

    // Build batches.
    std::vector<std::vector<std::string>> batches;
    std::vector<std::string> current;
    long long currentSize = 0;
    for (auto& fpath : tempFiles) {
        struct stat st;
        if (stat(fpath.c_str(), &st) != 0) {
            continue;
        }
        long long fsize = st.st_size;
        if (!current.empty() && currentSize + fsize > maxChunkSize) {
            batches.push_back(current);
            current.clear();
            currentSize = 0;
        }
        current.push_back(fpath);
        currentSize += fsize;
    }
    if (!current.empty()) {
        batches.push_back(current);
    }

    // Merge batches and write cutflow.
    for (size_t bi = 0; bi < batches.size(); ++bi) {
        std::string outPath;
        if (bi == 0) {
            outPath = outputFile;
        } else {
            outPath = outDir + "/" + baseName + "_" + std::to_string(bi) + ext;
        }
        std::cout << "Merging batch " << (bi+1) << "/" << batches.size()
                  << " into " << outPath << " ...\n";

        TFileMerger merger(true);
        for (auto& tf : batches[bi]) {
            merger.AddFile(tf.c_str());
        }
        merger.OutputFile(outPath.c_str());
        merger.Merge();
        std::cout << "Merge of batch " << (bi+1) << " done.\n";

        // Write cutflow histogram.
        TFile* outF = TFile::Open(outPath.c_str(), "UPDATE");
        TTree* outT = (TTree*)outF->Get("Events");
        Long64_t passedEntries = outT ? outT->GetEntries() : 0;

        TH1D* hCutFlow = new TH1D("cutflow","Cut Flow;Step;Entries", 2, 0.5, 2.5);
        hCutFlow->SetBinContent(1, totalRawEntries);
        hCutFlow->SetBinContent(2, passedEntries);
        hCutFlow->GetXaxis()->SetBinLabel(1,"Raw");
        hCutFlow->GetXaxis()->SetBinLabel(2, cutString.c_str());
        outF->cd();
        hCutFlow->Write();
        outF->Close();

        std::cout << "Written cutflow histogram into " << outPath << "\n";
    }

    // Remove temp directory.
    gSystem->Exec(("rm -rf " + tempDirName).c_str());
    std::cout << "All finished. Outputs:";
    for (size_t bi = 0; bi < batches.size(); ++bi) {
        if (bi == 0) {
            std::cout << " " << outputFile;
        } else {
            std::cout << " " << outDir << "/" << baseName << "_" << bi << ext;
        }
    }
    std::cout << std::endl;

    return 0;
}
