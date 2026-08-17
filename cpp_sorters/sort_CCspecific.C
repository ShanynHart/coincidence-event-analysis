/* Compton Pair Builder for LaBr3 2x2 + 2x2 system */
/* Upgraded with unbiased timing extraction */

#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <inttypes.h> 
#include <time.h>

#include "TTree.h"
#include "TFile.h"
#include "TH1D.h"
#include "TRandom3.h"
#include "TString.h"

#define SIZE 16384

int32_t buffer[SIZE];
uint64_t TS=0, TSfirst=0;
uint64_t counter=0;

struct DetEvent {
    std::vector<double> slowT;
    std::vector<double> fastT;
    std::vector<double> slowE;
    std::vector<double> fastE;
    double x=0, y=0, z=0;
};

DetEvent det[8];

int main (int argc, char** argv)
{
    FILE *f;
    int i;
    char file_in[50], file_out[50];
    int Run=0, RunEnd=999999;
    uint64_t eventCounter = 0;

    if ((argc < 2) || (argc > 3)) {
        printf("Usage: %s data [last_subrun]\n", argv[0]);
        return 1;
    }

    sprintf(file_out, "%s.root", argv[1]);
    if (argc > 2) sscanf(argv[2], "%d", &RunEnd);

    TFile *g = new TFile(file_out,"RECREATE");

    // 🔥 UNBIASED TIMING HISTOGRAM
    TH1D* hDT = new TH1D("hDT","Fast time difference (all pairs)",400,-200,200);

    // 🔥 Rolling fast-time memory (NEW)
    double lastFastT[8] = {0};
    bool hasFast[8] = {false};

    // ================= ROOT TREE =================
    TTree *ComptonPairs = new TTree("ComptonPairs","ComptonPairs");

    double Es, Ea, xs, ys, zs, xa, ya, za;
    double ts, ta, dt, theta, costheta, Etot;
    int scatterID, absorberID;

    ComptonPairs->Branch("Es",&Es,"Es/D");
    ComptonPairs->Branch("Ea",&Ea,"Ea/D");
    ComptonPairs->Branch("xs",&xs,"xs/D");
    ComptonPairs->Branch("ys",&ys,"ys/D");
    ComptonPairs->Branch("zs",&zs,"zs/D");
    ComptonPairs->Branch("xa",&xa,"xa/D");
    ComptonPairs->Branch("ya",&ya,"ya/D");
    ComptonPairs->Branch("za",&za,"za/D");
    ComptonPairs->Branch("ts",&ts,"ts/D");
    ComptonPairs->Branch("ta",&ta,"ta/D");
    ComptonPairs->Branch("dt",&dt,"dt/D");
    ComptonPairs->Branch("theta",&theta,"theta/D");
    ComptonPairs->Branch("costheta",&costheta,"costheta/D");
    ComptonPairs->Branch("Etotal",&Etot,"Etotal/D");
    ComptonPairs->Branch("scatterID",&scatterID,"scatterID/I");
    ComptonPairs->Branch("absorberID",&absorberID,"absorberID/I");

    // ================= GEOMETRY =================
    double xCoord[8] = { 4.25, -4.25, -4.25,  4.25,  4.25,  4.25, -4.25, -4.25 };
    double yCoord[8] = {-4.25, -4.25,  4.25,  4.25, -4.25,  4.25,  4.25, -4.25 };
    double zCoord[8] = { 61,    61,    61,    61,    28,    28,    28,    28 };

    // ================= ENERGY CALIBRATION =================
    const double E2 = 661.657;
    std::vector<double> ch2slow = {8866,8097,8719,8766,9326,7908,7582,8576};
    std::vector<double> a1slow(8), b1slow(8);

    for (int i=0;i<8;i++) {
        a1slow[i] = E2 / ch2slow[i];
        b1slow[i] = 0;
    }

    TRandom3 randy;

    // ================= DATA LOOP =================
    while(Run <= RunEnd)
    {
        sprintf(file_in,"%s_%d",argv[1],Run);
        f = fopen(file_in,"rb");
        if (!f) break;

        uint64_t eventStartTS = 0;

        while (!feof(f))
        {
            fread(buffer,sizeof(buffer[0]),SIZE,f);

            for (i=6;i<SIZE;i+=2)
            {
                uint64_t data = buffer[i+1];
                uint64_t TSbot = buffer[i];

                if ((data & 0xc0000000) == 0xc0000000)
                {
                    int ident = (data & 0x0fff0000) >> 16;
                    int adcdata = (data & 0xffff);

                    TS = TSbot;
                    if (counter==0){ TSfirst=TS; counter=1; }

                    if (eventStartTS==0) eventStartTS=TS;

                    int detID;

                    // ===== SLOW ENERGY =====
                    if (ident>=64 && ident<72) {
                        detID = ident-64;

                        double E = a1slow[detID]*(adcdata + randy.Gaus(0,16));
                        det[detID].slowE.push_back(E);
                        det[detID].slowT.push_back(TS*100.0);

                        det[detID].x = xCoord[detID];
                        det[detID].y = yCoord[detID];
                        det[detID].z = zCoord[detID];
                    }

                    // ===== FAST TIME =====
                    else if (ident>=88 && ident<96) {
                        detID = ident-88;

                        int tick = adcdata & 0x1fff;
                        int tock = (adcdata>>13)&0x7;

                        double t = TS*100.0 + tock*20.0 + (tick/8192.0)*20.0;

                        det[detID].fastT.push_back(t);

                        // UNBIASED Δt BUILDING
                        for (int d=0; d<8; d++) {
                            if (d == detID) continue;
                            if (!hasFast[d]) continue;

                            double dt_all = t - lastFastT[d];
                            hDT->Fill(dt_all);
                        }

                        lastFastT[detID] = t;
                        hasFast[detID] = true;
                    }

                    // ===== EVENT BUILDER =====
                    if ((TS - eventStartTS) >=9)
                    {
                        eventCounter++;

                        //printf("\n================ EVENT %llu ================\n", eventCounter);

                        std::vector<int> S={0,1,2,3};
                        std::vector<int> A={4,5,6,7};

                        for (int s : S)
                        for (int a : A)
                        {
                            if (det[s].slowE.empty() || det[a].slowE.empty()) continue;
                            if (det[s].fastT.empty() || det[a].fastT.empty()) continue;

                            double tsf = det[s].fastT[0];
                            double taf = det[a].fastT[0];

                            // 🔥 USE THIS LATER AFTER YOU PICK WINDOW
                            // if (fabs(taf-tsf) > 20.0) continue;

                            Es = det[s].slowE[0];
                            Ea = det[a].slowE[0];
                            Etot = Es + Ea;

                            if (Etot<400 || Etot>800) continue;

                            costheta = 1.0 - 511.0*(1.0/Ea - 1.0/Etot);
                            if (costheta<-1 || costheta>1) continue;

                            theta = acos(costheta);

                            xs = det[s].x; ys = det[s].y; zs = det[s].z;
                            xa = det[a].x; ya = det[a].y; za = det[a].z;

                            ts = det[s].slowT[0]/10.0;
                            ta = det[a].slowT[0]/10.0;
                            dt = ta-ts;

                            scatterID=s;
                            absorberID=a;

                            ComptonPairs->Fill();
                        }

                        // clear event
                        for (int d=0;d<8;d++) {
                            det[d].slowE.clear();
                            det[d].slowT.clear();
                            det[d].fastT.clear();
                        }

                        eventStartTS = TS;
                    }
                }
            }
        }
        fclose(f);
        Run++;
    }

    // ================= WRITE =================
    hDT->Write();
    ComptonPairs->Write();
    g->Write();
    g->Close();

    printf("Done.\n");
}