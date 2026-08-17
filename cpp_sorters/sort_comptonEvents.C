#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TCutG.h>
#include <TROOT.h>
#include <TSystem.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <TCanvas.h>
#include <TStyle.h>

TCutG *comptonGate = nullptr;

// Klein-Nishina helper: returns differential cross-section (relative) for unpolarized photons
static double klein_nishina_rel(double E0_keV, double theta_rad) {
    const double mec2 = 511.0; // keV
    const double r0 = 2.8179403227e-13; // cm (classical electron radius)
    double cosT = std::cos(theta_rad);
    double sinT = std::sin(theta_rad);
    double alpha = E0_keV / mec2;
    // k = E'/E0
    double k = 1.0 / (1.0 + alpha * (1.0 - cosT));
    double k2 = k*k;
    double term = k + 1.0 / k - sinT*sinT;
    double val = 0.5 * r0 * r0 * k2 * term; // absolute units cm^2/sr, but we use relatively
    return val;
}

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: ./exeCC input.csv [srcX srcY srcZ] [tol_deg]" << std::endl;
        std::cerr << "  srcX/srcY/srcZ : optional source coordinates (default 0,0,0)" << std::endl;
        std::cerr << "  tol_deg        : optional angular tolerance in degrees (default 30)" << std::endl;
        return 1;
    }

    const char* inputFilename = argv[1];

    // Optional: source coordinates and tolerance
        // Optional: source coordinates and tolerance
        // Note: input CSV coordinates are in cm; we convert everything to mm internally (1 cm = 10 mm)
        double srcX = 0.0, srcY = 0.0, srcZ = 10.0; // default in cm
    double tolDeg = 45.0; // default tolerance in degrees
    if (argc >= 5) {
        try {
            srcX = std::stod(argv[2]);
            srcY = std::stod(argv[3]);
            srcZ = std::stod(argv[4]);
        } catch (...) {
            std::cerr << "Warning: could not parse source coordinates, using default (0,0,0)" << std::endl;
            srcX = srcY = srcZ = 0.0;
        }
    }
    if (argc >= 6) {
        try { tolDeg = std::stod(argv[5]); } catch (...) { /* ignore */ }
    }

    // KN weighting toggle: argv[6] optional. Set to "1" to enable computing KN weight and L; "0" (default) to disable.
    bool knEnabled = true;
    if (argc >= 7) {
        std::string s = argv[6];
        if (s == "1" || s == "true") knEnabled = true;
    }
    // Optional KN threshold (argv[7]) - if provided and >0, events with L < threshold will be rejected
    double knThreshold = 0.2;
    if (argc >= 8) {
        try { knThreshold = std::stod(argv[7]); } catch (...) { knThreshold = 0.0; }
    }

    // convert source coords from cm to mm for internal geometry calculations
    const double coordFactor = 10.0; // cm -> mm
    srcX *= coordFactor; srcY *= coordFactor; srcZ *= coordFactor;

    // Determine output directory from input CSV parent path
    namespace fs = std::filesystem;
    fs::path inputPath(inputFilename);
    fs::path outDir = inputPath.parent_path();
    if (outDir.empty()) outDir = fs::current_path();
    std::string outRootPath = (outDir / "comptonAnalysis.root").string();
    std::string outCsvName = (outDir / "sorted_comptonKept.csv").string();

    // Load the graphical cut
    std::vector<Double_t> cutg_vect0{ 629.5131114381977, 666.591763676106, 93.93257911285626, 60.97377712360448, 633.6329616868542, 643.9325873084954, 639.8127370598389, 635.6928868111825, 629.5131114381977 };
   std::vector<Double_t> cutg_vect1{ -1.927972910452558, 113.7499949149785, 699.851698563829, 611.1652565643319, 9.639823872090545, 52.05507874141526, 9.639823872090545, 13.49575613293825, -1.927972910452558 };

    // construct TCutG with the actual number of points. If the exported polygon
    // repeats the first point at the end (closed polygon), drop the duplicate.
    int nPoints = (int)std::min(cutg_vect0.size(), cutg_vect1.size());
    if (nPoints > 1) {
        if (cutg_vect0.front() == cutg_vect0.back() && cutg_vect1.front() == cutg_vect1.back()) {
            --nPoints; // drop duplicated closing point
        }
    }
    comptonGate = new TCutG("CUTG", nPoints, cutg_vect0.data(), cutg_vect1.data());

    // Read CSV input file (argv[1]) with columns: eventID,scatterDet,scatterE_keV,scatterT_ns,scatterX,scatterY,scatterZ,
    // absorberDet,absorberE_keV,absorberT_ns,absorberX,absorberY,absorberZ

    const char* inputCsvFilename = argv[1];
    // small helper: trim whitespace from both ends
    auto trim = [](std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
    };

    // Histograms
    TH2D *hRaw = new TH2D("hRaw", "Scatter vs Absorber (raw);Scatter;Absorber",
                          200,0,2000, 200,0,2000);

    TH2D *hGated = new TH2D("hGated", "Scatter vs Absorber (Compton-gated)",
                            200,0,2000, 200,0,2000);

    TH1D *hAbsorberProj = new TH1D("hAbs", "Absorber Energy (gated)", 
                                  200,0,2000);
    
    TH1D *hScattererProj = new TH1D("hScatter", "Scatter Energy (gated)", 
                                   200,0,2000);                               
    
    // Time-difference histogram (absorber time - scatter time), in ns
    TH1D *hTimeDiff = new TH1D("hTimeDiff", "Absorber - Scatter Time Difference;#Delta t (ns);Counts",
                               400, -200.0, 200.0);

    // Open CSV
    std::ifstream infile(inputCsvFilename);
    if (!infile.is_open()) {
        std::cerr << "ERROR: Could not open input file '" << inputCsvFilename << "'" << std::endl;
        return 1;
    }

    // Open output CSV for kept (sorted) Compton coincidence events
    std::ofstream csvOut(outCsvName);
    if (!csvOut.is_open()) {
        std::cerr << "WARNING: Could not open output CSV '" << outCsvName << "' for writing. Proceeding without CSV output." << std::endl;
    } else {
        // no header requested — write rows only
    }

    std::string line;
    // Read first line and detect whether it is a header. If it's data, rewind to start.
    if (!std::getline(infile, line)) {
        std::cerr << "ERROR: Input file appears empty: " << inputCsvFilename << std::endl;
        return 1;
    }

    auto looks_like_header = [](const std::string &s) {
        for (char c : s) if (std::isalpha((unsigned char)c)) return true;
        return false;
    };

    if (!looks_like_header(line)) {
        infile.clear();
        infile.seekg(0);
    }

    long long lineNo = 0;
    long long totalLines = 0;
    long long gatedTotal = 0;     // number inside graphical gate before any rejection
    long long gatedKept = 0;      // number kept after all checks
    long long gatedRejectedKine = 0;  // number rejected by kinematic check
    long long gatedRejectedGeom = 0;  // number rejected by geometric/position check
    long long gatedRejectedKN = 0;    // number rejected by KN threshold
    const double mec2 = 511.0; // electron rest mass energy in keV
    const double tolRad = tolDeg * M_PI / 180.0;
    while (std::getline(infile, line)) {
        ++lineNo;
        ++totalLines;
        if (line.empty()) continue;

        std::vector<std::string> cols;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            trim(item);
            cols.push_back(item);
        }

    if (cols.size() < 13) continue; // skip malformed lines

        double scatterE = 0.0;
        double absorberE = 0.0;
        double scatterT = 0.0;
        double absorberT = 0.0;
        try {
            scatterE = std::stod(cols[2]);
            scatterT = std::stod(cols[3]);
            absorberE = std::stod(cols[8]);
            absorberT = std::stod(cols[9]);
        } catch (...) {
            continue;
        }

        hRaw->Fill(scatterE, absorberE);
        if (comptonGate->IsInside(scatterE, absorberE)) 
        {
            // inside graphical gate
            ++gatedTotal;

            // Kinematic check using Compton formula:
            // E0 = scatterE + absorberE
            // cos(theta) = 1 - mec2*(1/Ea - 1/E0)
            double E0 = scatterE + absorberE;
            bool kinematic_ok = true;
            if (absorberE <= 0.0 || E0 <= 0.0) kinematic_ok = false;
            double cosTheta = 0.0;
            if (kinematic_ok) {
                cosTheta = 1.0 - mec2 * (1.0 / absorberE - 1.0 / E0);
                // allow small numerical tolerance
                if (cosTheta < -1.000001 || cosTheta > 1.000001) kinematic_ok = false;
            }

            if (!kinematic_ok) {
                ++gatedRejectedKine;
                continue; // skip filling "kept" histograms
            }

            // passed kinematic check -> now do geometric (quadrant) consistency check
            // Parse positions from CSV columns (cols[4..6] and cols[10..12])
            double sx = 0.0, sy = 0.0, sz = 0.0;
            double ax = 0.0, ay = 0.0, az = 0.0;
            try 
            {
                // input CSV positions are in cm; convert to mm
                sx = std::stod(cols[4]) * coordFactor; sy = std::stod(cols[5]) * coordFactor; sz = std::stod(cols[6]) * coordFactor;
                ax = std::stod(cols[10]) * coordFactor; ay = std::stod(cols[11]) * coordFactor; az = std::stod(cols[12]) * coordFactor;
            } catch (...) 
            {
                // if positions cannot be parsed, treat as geometric fail-safe: accept event
                sx = sy = sz = ax = ay = az = 0.0;
            }

            // geometry: vector of scattered photon (from scatter point to absorber point)
            double vx = ax - sx;
            double vy = ay - sy;
            double vz = az - sz;
            double vnorm = std::sqrt(vx*vx + vy*vy + vz*vz);
            bool geom_ok = true;
            if (vnorm <= 0.0) geom_ok = false;
            // incident direction: vector from source to scatter
            double ux = sx - srcX;
            double uy = sy - srcY;
            double uz = sz - srcZ;
            double unorm = std::sqrt(ux*ux + uy*uy + uz*uz);
            if (unorm <= 0.0) geom_ok = false;
            double cosThetaGeom = 0.0;
            if (geom_ok) {
                cosThetaGeom = (vx*ux + vy*uy + vz*uz) / (vnorm * unorm);
                if (cosThetaGeom < -1.0) cosThetaGeom = -1.0;
                if (cosThetaGeom > 1.0) cosThetaGeom = 1.0;
            }

            double thetaEnergy = 0.0;
            if (cosTheta < -1.0) cosTheta = -1.0;
            if (cosTheta > 1.0) cosTheta = 1.0;
            thetaEnergy = std::acos(cosTheta);

            if (geom_ok) {
                double thetaGeom = std::acos(cosThetaGeom);
                // compare; if mismatch too large reject
                if (std::fabs(thetaGeom - thetaEnergy) > tolRad) {
                    ++gatedRejectedGeom;
                    continue; // reject event
                }

                // If KN weighting is enabled, compute the KN weight and a consistency weight
                double kn_norm = 0.0;
                double L = 1.0;
                if (knEnabled) {
                    double E0 = scatterE + absorberE; // keV
                    double kn_val = klein_nishina_rel(E0, thetaGeom);
                    double kn_max = klein_nishina_rel(E0, 0.0);
                    if (kn_max > 0) kn_norm = kn_val / kn_max; else kn_norm = 0.0;

                    // use tolRad as a (conservative) sigma for the consistency Gaussian
                    double sigma_rad = std::max(tolRad, 0.1); // at least ~0.1 rad if tol very small
                    double delta = thetaGeom - thetaEnergy;
                    double p_cons = std::exp(-0.5 * (delta*delta) / (sigma_rad*sigma_rad));
                    L = kn_norm * p_cons;
                }

                // If KN threshold set and enabled, reject events with low L
                if (knEnabled && knThreshold > 0.0) {
                    if (L < knThreshold) {
                        ++gatedRejectedKN;
                        continue;
                    }
                }

                // passed all checks -> keep
                ++gatedKept;
            }
            hGated->Fill(scatterE, absorberE);
            hAbsorberProj->Fill(absorberE);
            hScattererProj->Fill(scatterE);
            double dt = absorberT - scatterT; // absorber - scatter in ns
            hTimeDiff->Fill(dt);

            // write to CSV if available: energies in MeV
            if (csvOut.is_open()) {
                double scatterE_MeV = scatterE / 1000.0;
                double absorberE_MeV = absorberE / 1000.0;
                // Always output two trailing columns: kn_norm and L.
                // If KN weighting is disabled, emit neutral values (1.0) so columns are present.
                double kn_norm_out = 1.0;
                double Lout = 1.0;
                if (knEnabled) {
                    double E0tmp = scatterE + absorberE; // keV
                    // compute geometric angle safely
                    double dot = (ax - sx) * (sx - srcX) + (ay - sy) * (sy - srcY) + (az - sz) * (sz - srcZ);
                    double mag1 = std::sqrt((ax - sx)*(ax - sx) + (ay - sy)*(ay - sy) + (az - sz)*(az - sz));
                    double mag2 = std::sqrt((sx - srcX)*(sx - srcX) + (sy - srcY)*(sy - srcY) + (sz - srcZ)*(sz - srcZ));
                    double cos_g = 0.0;
                    if (mag1 > 0.0 && mag2 > 0.0) cos_g = dot / (mag1 * mag2);
                    cos_g = std::max(-1.0, std::min(1.0, cos_g));
                    double thetaGeom_out = std::acos(cos_g);

                    double kn_val_out = klein_nishina_rel(E0tmp, thetaGeom_out);
                    double kn_max_out = klein_nishina_rel(E0tmp, 0.0);
                    if (kn_max_out > 0.0) kn_norm_out = kn_val_out / kn_max_out; else kn_norm_out = 0.0;

                    double thetaEnergy_out = thetaEnergy;
                    double sigma_rad_out = std::max(tolRad, 0.1);
                    double delta_out = thetaGeom_out - thetaEnergy_out;
                    double p_cons_out = std::exp(-0.5 * (delta_out*delta_out) / (sigma_rad_out*sigma_rad_out));
                    Lout = kn_norm_out * p_cons_out;
                }

                csvOut << scatterE_MeV << "," << sx << "," << sy << "," << sz << ","
                       << absorberE_MeV << "," << ax << "," << ay << "," << az << //"\n";
                        "," << kn_norm_out << "," << Lout << "\n";
            }
        }
    }

    infile.close();
    if (csvOut.is_open()) csvOut.close();

    TFile out(outRootPath.c_str(), "RECREATE");
    hRaw->Write();
    hGated->Write();
    hAbsorberProj->Write();
    hScattererProj->Write();
    hTimeDiff->Write();
    out.Close();

    // Diagnostics: print gate points and histogram axis ranges
    if (comptonGate) {
        std::cout << "Compton gate has " << comptonGate->GetN() << " points. First points (x,y):\n";
        for (int i=0; i<comptonGate->GetN() && i<20; ++i) {
            double gx = comptonGate->GetX()[i];
            double gy = comptonGate->GetY()[i];
            std::cout << "  " << i << ": (" << gx << ", " << gy << ")\n";
        }
    }
    std::cout << "Raw histogram X range: [" << hRaw->GetXaxis()->GetXmin() << ", " << hRaw->GetXaxis()->GetXmax() << "]\n";
    std::cout << "Raw histogram Y range: [" << hRaw->GetYaxis()->GetXmin() << ", " << hRaw->GetYaxis()->GetXmax() << "]\n";

    // Create and save quick plots
    gROOT->SetBatch(kTRUE); // don't open GUI windows

    // Time-difference
    TCanvas *cTime = new TCanvas("cTime","Time Difference",800,600);
    hTimeDiff->SetLineColor(kBlue);
    hTimeDiff->Draw();
    cTime->SaveAs((outDir / "hTimeDiff.png").string().c_str());
    delete cTime;

    // Gated 2D (overlay gate for diagnostics)
    TCanvas *cG = new TCanvas("cG","Gated Scatter vs Absorber",800,600);
    hGated->Draw("COLZ");
    if (comptonGate) {
        comptonGate->SetLineColor(kRed);
        comptonGate->SetLineWidth(2);
        comptonGate->Draw("same");
    }
    cG->SaveAs((outDir / "hGated.png").string().c_str());
    delete cG;

    // Raw 2D (overlay gate for diagnostics)
    TCanvas *cR = new TCanvas("cR","Raw Scatter vs Absorber",800,600);
    hRaw->Draw("COLZ");
    if (comptonGate) {
        comptonGate->SetLineColor(kRed);
        comptonGate->SetLineWidth(2);
        comptonGate->Draw("same");
    }
    cR->SaveAs((outDir / "hRaw.png").string().c_str());
    delete cR;

    // Projections
    TCanvas *cA = new TCanvas("cA","Absorber Projection",800,600);
    hAbsorberProj->Draw();
    cA->SaveAs((outDir / "hAbsorberProj.png").string().c_str());
    delete cA;

    TCanvas *cS = new TCanvas("cS","Scatter Projection",800,600);
    hScattererProj->Draw();
    cS->SaveAs((outDir / "hScattererProj.png").string().c_str());
    delete cS;

    std::cout << "Done. Output written to: " << outRootPath << std::endl;

    // Report counts
    std::cout << "Total CSV lines processed: " << totalLines << std::endl;
    std::cout << "Number inside graphical gate (before rejections): " << gatedTotal << std::endl;
    std::cout << "Number rejected by kinematic check: " << gatedRejectedKine << std::endl;
    std::cout << "Number rejected by geometric (position) check: " << gatedRejectedGeom << std::endl;
    if (knEnabled) std::cout << "Number rejected by KN threshold: " << gatedRejectedKN << std::endl;
    std::cout << "Number kept after all rejections: " << gatedKept << std::endl;

    return 0;
}