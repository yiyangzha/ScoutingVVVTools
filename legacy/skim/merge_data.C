// Summary: Merge skimmed data ROOT files in size-limited batches.
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <algorithm>

#include "TFileMerger.h"  // Kept to minimize changes.
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TH1D.h"

namespace fs = std::filesystem;

// Max batch size: 25 GB
static constexpr long long MAX_CHUNK_SIZE = 25LL * 1024 * 1024 * 1024;

const std::string inputDir  = "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/v2/CMS_VVV_Scouting_2024G_v4/";
const std::string outputFile= "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024G.root";
const std::string cutString = "nScoutingFatPFJetRecluster > 1";

int merge_data() {

    // Parse output path.
    std::string outDir  = outputFile.substr(0, outputFile.find_last_of("/\\"));
    std::string outBase = outputFile.substr(outputFile.find_last_of("/\\")+1);
    std::string baseName= outBase.substr(0, outBase.find_last_of('.'));
    std::string ext     = outBase.substr(outBase.find_last_of('.')); // Includes "."

    // Recursively collect .root files.
    std::vector<std::string> tempFiles;
    const auto opts = fs::directory_options::follow_directory_symlink |
                      fs::directory_options::skip_permission_denied;
    try {
        for (fs::recursive_directory_iterator it(inputDir, opts), end; it != end; ++it) {
            const fs::directory_entry& ent = *it;
            if (!ent.is_regular_file()) continue;
            if (ent.path().extension() == ".root") {
                tempFiles.push_back(ent.path().string());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "递归遍历时发生文件系统错误："
                  << e.what() << "（path=" << e.path1() << "）\n";
    }

    // Sort by file name.
    std::sort(tempFiles.begin(), tempFiles.end());

    // Build batches.
    std::vector<std::vector<std::string>> batches;
    std::vector<std::string> current;
    long long currentSize = 0;
    for (auto& fpath : tempFiles) {
        struct stat st;
        if (stat(fpath.c_str(), &st) != 0) {
            std::cerr << "警告：无法获取文件大小 " << fpath << "\n";
            continue;
        }
        long long fsize = st.st_size;
        if (!current.empty() && currentSize + fsize > MAX_CHUNK_SIZE) {
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

    std::cout << "一共分成 " << batches.size() << " 个 batch，开始逐个合并 …\n";
    Long64_t totalRawEntries = 0;

    // Sum totalRawEntries across all input files.
    for (size_t bi = 0; bi < batches.size(); ++bi) {
        for (auto& inPath : batches[bi]) {
            TFile* inF = TFile::Open(inPath.c_str(), "READ");
            if (!inF || inF->IsZombie()) {
                std::cerr << "  无法打开输入文件 " << inPath << "\n";
                continue;
            }
            TH1D* hIn = dynamic_cast<TH1D*>(inF->Get("cutflow"));
            if (hIn) {
                totalRawEntries += static_cast<Long64_t>(hIn->GetBinContent(1)); // Raw is usually bin 1.
            } else {
                std::cerr << "  找不到 cutflow 直方图于 " << inPath << "\n";
            }
            inF->Close();
        }
    }

    // Write each batch after applying cutString.
    for (size_t bi = 0; bi < batches.size(); ++bi) {
        // Build output file name.
        std::string outPath;
        if (bi == 0) {
            outPath = outputFile;
        } else {
            outPath = outDir + "/" + baseName + "_" + std::to_string(bi) + ext;
        }
        std::cout << "Batch " << (bi+1) << "/" << batches.size()
                  << " 过滤并写入 " << outPath << " \n";

        // Build a TChain for this batch.
        TChain chain("Events");
        for (auto& fp : batches[bi]) {
            int nadd = chain.Add(fp.c_str());
            if (nadd == 0) {
                std::cerr << "  警告：文件中未找到 TTree 'Events'，跳过： " << fp << "\n";
            }
        }

        // Open the output file so CopyTree writes there.
        TFile* outF = TFile::Open(outPath.c_str(), "RECREATE");
        if (!outF || outF->IsZombie()) {
            std::cerr << "  无法创建输出文件：" << outPath << "\n";
            delete outF;
            continue;
        }
        outF->cd();

        // Apply cutString during copy.
        TTree* outT = nullptr;
        if (chain.GetNtrees() > 0) {
            outT = chain.CopyTree(cutString.c_str());  // Apply the cut while merging.
        } else {
            // Write an empty tree if this batch has no valid Events tree.
            outT = new TTree("Events", "Events");
        }

        Long64_t passedEntries = outT ? outT->GetEntries() : 0;

        // Write cutflow; keep Raw as the global input total.
        TH1D* hCutFlow = new TH1D("cutflow", "Cut Flow;Step;Entries",
                                  2, 0.5, 2.5);
        hCutFlow->SetBinContent(1, totalRawEntries);
        hCutFlow->SetBinContent(2, passedEntries);
        hCutFlow->GetXaxis()->SetBinLabel(1, "Raw");
        hCutFlow->GetXaxis()->SetBinLabel(2, cutString.c_str());

        // Persist output.
        if (outT) outT->Write();               // Write filtered Events.
        hCutFlow->Write("cutflow", TObject::kOverwrite);

        outF->Close();
        delete outF;

        std::cout << "  完成 batch " << (bi+1)
                  << "：totalRaw=" << totalRawEntries
                  << ", passed="   << passedEntries << "\n";
    }

    std::cout << "所有 batch 合并完成。\n";
    return 0;
}
