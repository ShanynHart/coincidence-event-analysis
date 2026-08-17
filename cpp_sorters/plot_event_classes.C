#include <iostream>
#include <string>
#include <vector>
#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"
#include "TKey.h"

static std::vector<std::string> findClassHistNames(TFile* file, const std::string& prefix, int nclasses = 4) {
    std::vector<std::string> names;
    names.reserve(nclasses);
    for (int i = 0; i < nclasses; ++i) {
        const std::string exact = prefix + std::to_string(i);
        if (file->GetListOfKeys()->Contains(exact.c_str())) {
            names.push_back(exact);
            continue;
        }

        const std::string alt = prefix + "_" + std::to_string(i);
        if (file->GetListOfKeys()->Contains(alt.c_str())) {
            names.push_back(alt);
            continue;
        }

        names.push_back("");
    }
    return names;
}

static void draw2D(TCanvas* canvas, TFile* file, const std::vector<std::string>& names, const char* title) {
    canvas->Clear();
    canvas->Divide(2, 2);
    for (int i = 0; i < 4; ++i) {
        canvas->cd(i + 1);
        gPad->SetRightMargin(0.14);
        const std::string& histName = names[i];
        if (histName.empty()) {
            std::cerr << "Missing histogram for class " << i << std::endl;
            continue;
        }
        TH2* hist = dynamic_cast<TH2*>(file->Get(histName.c_str()));
        if (!hist) {
            std::cerr << "Missing histogram: " << histName << std::endl;
            continue;
        }
        hist->SetTitle((std::string(title) + " class " + std::to_string(i)).c_str());
        hist->Draw("COLZ");
    }
    canvas->Update();
}

static void draw1D(TCanvas* canvas, TFile* file, const std::vector<std::string>& names, const char* title) {
    canvas->Clear();
    canvas->Divide(2, 2);
    for (int i = 0; i < 4; ++i) {
        canvas->cd(i + 1);
        const std::string& histName = names[i];
        if (histName.empty()) {
            std::cerr << "Missing histogram for class " << i << std::endl;
            continue;
        }
        TH1* hist = dynamic_cast<TH1*>(file->Get(histName.c_str()));
        if (!hist) {
            std::cerr << "Missing histogram: " << histName << std::endl;
            continue;
        }
        hist->SetTitle((std::string(title) + " class " + std::to_string(i)).c_str());
        hist->Draw();
    }
    canvas->Update();
}

void plot_event_classes(const char* inputFile = "ComptonPairs.root", const char* outputPrefix = "event_classes") {
    TFile* file = TFile::Open(inputFile);
    if (!file || file->IsZombie()) {
        std::cerr << "Failed to open " << inputFile << std::endl;
        return;
    }

    gStyle->SetOptStat(1110);

    const std::vector<std::string> esEaNames = findClassHistNames(file, "hEsEa_class");
    const std::vector<std::string> dtNames = findClassHistNames(file, "hDT_class");
    const std::vector<std::string> etotNames = findClassHistNames(file, "hEtot_class");

    TCanvas* cEsEa = new TCanvas("cEsEa", "Es vs Ea by event class", 1400, 1000);
    draw2D(cEsEa, file, esEaNames, "Es vs Ea");
    cEsEa->SaveAs((std::string(outputPrefix) + "_EsEa.png").c_str());

    TCanvas* cDT = new TCanvas("cDT", "dt by event class", 1400, 1000);
    draw1D(cDT, file, dtNames, "dt");
    cDT->SaveAs((std::string(outputPrefix) + "_dt.png").c_str());

    TCanvas* cEtot = new TCanvas("cEtot", "Etot by event class", 1400, 1000);
    draw1D(cEtot, file, etotNames, "Etot");
    cEtot->SaveAs((std::string(outputPrefix) + "_Etot.png").c_str());

    file->Close();
}
