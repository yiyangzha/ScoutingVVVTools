#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <algorithm>

#include "TFileMerger.h"  // 不再使用，但保留以尽量少改动
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TH1D.h"

namespace fs = std::filesystem;

// 最大单个 batch 大小：25 GB
static constexpr long long MAX_CHUNK_SIZE = 25LL * 1024 * 1024 * 1024;

const std::string inputDir  = "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/v2/CMS_VVV_Scouting_2024G_v4/";
const std::string outputFile= "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024G.root";
const std::string cutString = "nScoutingFatPFJetRecluster > 1";

int merge_data() {

    // 解析输出路径
    std::string outDir  = outputFile.substr(0, outputFile.find_last_of("/\\"));
    std::string outBase = outputFile.substr(outputFile.find_last_of("/\\")+1);
    std::string baseName= outBase.substr(0, outBase.find_last_of('.'));
    std::string ext     = outBase.substr(outBase.find_last_of('.')); // 包括 “.”

    // === 递归收集所有 .root 文件（其余逻辑不变） ===
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

    // 按文件名排序（可选）
    std::sort(tempFiles.begin(), tempFiles.end());

    // 分批
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

    // 计算 totalRawEntries：遍历所有原文件（保持原有“全局总数写入每个输出文件”的行为）
    for (size_t bi = 0; bi < batches.size(); ++bi) {
        for (auto& inPath : batches[bi]) {
            TFile* inF = TFile::Open(inPath.c_str(), "READ");
            if (!inF || inF->IsZombie()) {
                std::cerr << "  无法打开输入文件 " << inPath << "\n";
                continue;
            }
            TH1D* hIn = dynamic_cast<TH1D*>(inF->Get("cutflow"));
            if (hIn) {
                totalRawEntries += static_cast<Long64_t>(hIn->GetBinContent(1)); // Raw 一般是 bin 1
            } else {
                std::cerr << "  找不到 cutflow 直方图于 " << inPath << "\n";
            }
            inF->Close();
        }
    }

    // 对每个 batch：按 cutString 过滤后写入输出文件
    for (size_t bi = 30; bi < batches.size(); ++bi) {
        // 构造输出文件名
        std::string outPath;
        if (bi == 0) {
            outPath = outputFile;
        } else {
            outPath = outDir + "/" + baseName + "_" + std::to_string(bi) + ext;
        }
        std::cout << "Batch " << (bi+1) << "/" << batches.size()
                  << " 过滤并写入 " << outPath << " \n";

        // 用 TChain 聚合本 batch 的 Events 树
        TChain chain("Events");
        for (auto& fp : batches[bi]) {
            int nadd = chain.Add(fp.c_str());
            if (nadd == 0) {
                std::cerr << "  警告：文件中未找到 TTree 'Events'，跳过： " << fp << "\n";
            }
        }

        // 打开输出文件并把当前目录切到它，确保 CopyTree 的输出落到该文件
        TFile* outF = TFile::Open(outPath.c_str(), "RECREATE");
        if (!outF || outF->IsZombie()) {
            std::cerr << "  无法创建输出文件：" << outPath << "\n";
            delete outF;
            continue;
        }
        outF->cd();

        // 施加筛选复制：只写入通过 cutString 的条目
        TTree* outT = nullptr;
        if (chain.GetNtrees() > 0) {
            outT = chain.CopyTree(cutString.c_str());  // 关键修改：在合并时应用 cut
        } else {
            // 若本 batch 没有有效的 Events 树，仍写一个空树以保持结构一致
            outT = new TTree("Events", "Events");
        }

        Long64_t passedEntries = outT ? outT->GetEntries() : 0;

        // 写 cutflow（保持原行为：Raw 为全体输入文件之和）
        TH1D* hCutFlow = new TH1D("cutflow", "Cut Flow;Step;Entries",
                                  2, 0.5, 2.5);
        hCutFlow->SetBinContent(1, totalRawEntries);
        hCutFlow->SetBinContent(2, passedEntries);
        hCutFlow->GetXaxis()->SetBinLabel(1, "Raw");
        hCutFlow->GetXaxis()->SetBinLabel(2, cutString.c_str());

        // 持久化写出
        if (outT) outT->Write();               // 写筛选后的 Events
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
