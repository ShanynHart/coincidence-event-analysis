#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <cmath>
#include <vector>
#include <iostream>

const double MeCsq = 510.9989461; // keV

bool physical_energy_ordering_double(double E1, double E2) {
    double E0 = E1 + E2;
    return std::abs(1.0 + MeCsq * (1.0/E0 - 1.0/(E0-E1))) < 1.0;
}

int main(int argc, char** argv) {

    const char* inputFilename = argv[1];

    Double_t energySlow[8], timeSlow[8], timeFast[8];
    Double_t xPos[8], yPos[8], zPos[8];

    TFile *inFile = TFile::Open(inputFilename);
    if (!inFile || inFile->IsZombie()) {
        std::cerr << "Error opening file: " << inputFilename << std::endl;
        return 1;
    }

    TTree *LaBrData = (TTree*)inFile->Get("LaBrData");
    if (!LaBrData) {
        std::cerr << "Tree not found!" << std::endl;
        return 1;
    }

    for (int i=0;i<8;i++) {
        LaBrData->SetBranchAddress(Form("slowEL%d",i), &energySlow[i]);
        LaBrData->SetBranchAddress(Form("timeSL%d",i), &timeSlow[i]);
        LaBrData->SetBranchAddress(Form("timeFL%d",i), &timeFast[i]);
        LaBrData->SetBranchAddress(Form("xPosL%d",i), &xPos[i]);
        LaBrData->SetBranchAddress(Form("yPosL%d",i), &yPos[i]);
        LaBrData->SetBranchAddress(Form("zPosL%d",i), &zPos[i]);
    }

    // Histograms
    TH2D *hScatterVsAbsorb = new TH2D("hScatterVsAbsorb", "Scatterer vs Absorber Energy;E_scatterer [keV];E_absorber [keV]", 700, 0, 700, 700, 0, 700);
    TH2D *hAngleVsScatter = new TH2D("hAngleVsScatter", "Compton Angle vs Scatterer Energy;Theta [rad];E_scatterer [keV]", 180, 0, 180, 700, 0, 700);
    TH2D *hAngleVsAbsorber = new TH2D("hAngleVsAbsorber", "Compton Angle vs Absorber Energy;Theta [rad];E_absorber [keV]", 180, 0, 180, 700, 0, 700);
    TH1D* htimediff = new TH1D("htimediff","htimediff (ns)",1000,0,1000);
    TH1D* hscatterenergy = new TH1D("hscatterenergy","hscatterenergy (keV)",700,0,700);
    TH1D* habsorberenergy = new TH1D("habsorberenergy","habsorberenergy (keV)",700,0,700);

    Long64_t nEntries = LaBrData->GetEntries();
    for (Long64_t entry=0; entry<nEntries; entry++) {
        LaBrData->GetEntry(entry);

        for (int s=4; s<8; s++) {
            if (energySlow[s] <= 0) continue;

            for (int a=0; a<4; a++) {
                if (energySlow[a] <= 0) continue;

                if (timeFast[s] >= timeFast[a]) continue;

                //std::cout << "Scat E: " << energySlow[s] << " keV, " << "Abs E: " << energySlow[a] << " keV, " << "TD: " << (timeFast[a] - timeFast[s]) << " ns" << std::endl;

                double deltaT = timeFast[a] - timeFast[s];
                double E0 = energySlow[s] + energySlow[a];

                if ((E0 < 662 - 0.1*662) || (E0 > 662 + 0.1*662)) continue;

                if (!physical_energy_ordering_double(energySlow[s], energySlow[a])) continue;

                double arg = 1.0 + MeCsq * (1.0/E0 - 1.0/(E0-energySlow[s]));
                if (arg < -1.0 || arg > 1.0) continue; // remove unphysical theta

                double theta = acos(arg);
                theta = theta * 180.0 / M_PI;
                //printf("theta: %.2f degrees\n", theta);

                hScatterVsAbsorb->Fill(energySlow[s], energySlow[a]);
                hAngleVsScatter->Fill(theta, energySlow[s]);
                hAngleVsAbsorber->Fill(theta, energySlow[a]);
                htimediff->Fill(deltaT);
                hscatterenergy->Fill(energySlow[s]);
                habsorberenergy->Fill(energySlow[a]);
            }
        }
    }

    TFile *outFile = new TFile("ComptonPlots.root","RECREATE");
    hScatterVsAbsorb->Write();
    hAngleVsScatter->Write();
    hAngleVsAbsorber->Write();
    htimediff->Write();
    hscatterenergy->Write();
    habsorberenergy->Write();

    outFile->Close();
    inFile->Close();

    std::cout << "Compton filtering and plotting complete!" << std::endl;

    return 0;
}