#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <TTreeFormula.h>
#include <TH1D.h>

#include <vector>
#include <string>
#include <regex>
#include <iostream>
#include <unistd.h>

// 要保留的 branch 通配列表
const std::vector<std::string> branchPatterns = {
    "nScouting*",
    "nGenJetAK8",
    "nGenPart",
    "DST_PFScouting_*",
    "ScoutingPFJetRecluster_*",
    "ScoutingFatPFJetRecluster_*",
    "GenJetAK8_*",
    "GenPart_*",
    "ScoutingElectron_pt",
    "ScoutingMuonVtx_pt",
    "ScoutingMuonNoVtx_pt",
    "ScoutingElectron_trackIso",
    "ScoutingMuonVtx_trackIso",
    "ScoutingMuonNoVtx_trackIso",
    "ScoutingElectron_ecalIso",
    "ScoutingMuonVtx_ecalIso",
    "ScoutingMuonNoVtx_ecalIso",
    "ScoutingElectron_hcalIso",
    "ScoutingMuonVtx_hcalIso",
    "ScoutingMuonNoVtx_hcalIso"
};
// 筛选条件
const std::string cutString = "nScoutingFatPFJetRecluster > 0";

void skim_single(const std::string& inPath) {

    // 编译通配为 regex
    std::vector<std::regex> patterns;
    for (auto& p : branchPatterns) {
        std::string r = "^" + std::regex_replace(p, std::regex(R"(\*)"), ".*") + "$";
        patterns.emplace_back(r);
    }

    // 打开输入文件
    TFile* inF = TFile::Open(inPath.c_str(), "READ");
    if (!inF || inF->IsZombie()) {
        std::cerr << "Error: cannot open input file " << inPath << std::endl;
        return;
    }
    TTree* tree = (TTree*)inF->Get("Events");
    if (!tree) {
        std::cerr << "Error: no Events tree in " << inPath << std::endl;
        inF->Close();
        return;
    }

    // 筛选 branch
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
    if (sel.empty()) {
        std::cerr << "Warning: no matching branches in " << inPath << std::endl;
        inF->Close();
        return;
    }
    tree->SetBranchStatus("*", 0);
    for (auto& b : sel) tree->SetBranchStatus(b.c_str(), 1);

    // 创建输出文件 temp.root
    TFile* outF = TFile::Open("temp.root", "RECREATE");
    outF->SetCompressionSettings(
        100 * ROOT::RCompressionSetting::EAlgorithm::kLZMA
        + 4
    );
    TTree* outT = tree->CloneTree(0);

    // 应用 cut
    TTreeFormula formula("cut", cutString.c_str(), tree);
    Long64_t nEntries = tree->GetEntries();

    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        if (formula.EvalInstance()) outT->Fill();
    }

    // 写入筛选后的树
    outF->cd();
    outT->Write();

    // 写入 cutflow 直方图
    Long64_t passedEntries = outT->GetEntries();
    TH1D* hCutFlow = new TH1D("cutflow", "Cut Flow;Step;Entries", 2, 0.5, 2.5);
    hCutFlow->SetBinContent(1, nEntries);
    hCutFlow->SetBinContent(2, passedEntries);
    hCutFlow->GetXaxis()->SetBinLabel(1, "Raw");
    hCutFlow->GetXaxis()->SetBinLabel(2, cutString.c_str());
    hCutFlow->Write();

    // 关闭文件
    outF->Close();
    inF->Close();

    std::cout << "Finished. Output written to temp.root" << std::endl;
}

// ROOT 宏入口：接受一个字符串参数
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <root://.../yourfile.root>" << std::endl;
        return 1;
    }
    std::string inputPath = argv[1];
    skim_single(inputPath);
    return 0;
}
