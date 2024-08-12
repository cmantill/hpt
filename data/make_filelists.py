from __future__ import annotations

import json
import os
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

os.environ["RUCIO_HOME"] = "/cvmfs/cms.cern.ch/rucio/x86_64/rhel7/py3/current"

qcd_ht_bins = [
    # "40to70",
    "70to100",
    "40to100",
    "100to200",
    "200to400",
    "400to600",
    "600to800",
    "800to1000",
    "1000to1200",
    "1200to1500",
    "1500to2000",
    "2000",
]

def get_v12():
    return {
        "2022EE": {
            "ParkingSingleMuon": {
                "ParkingSingleMuon_Run2022F": [
                    "/ParkingSingleMuon0/Run2022F-22Sep2023-v1/NANOAOD",
                    "/ParkingSingleMuon1/Run2022F-22Sep2023-v1/NANOAOD",
                    "/ParkingSingleMuon2/Run2022F-22Sep2023-v1/NANOAOD",
                ],
                "ParkingSingleMuon_Run2022G": [
                    "/ParkingSingleMuon0/Run2022G-22Sep2023-v2/NANOAOD",
                    "/ParkingSingleMuon1/Run2022G-22Sep2023-v1/NANOAOD",
                    "/ParkingSingleMuon2/Run2022G-22Sep2023-v1/NANOAOD",
                ],
            },
            "ParkingDoubleMuonLowMass": {
                "ParkingDoubleMuonLowMass_Run2022F": [
                    "/ParkingDoubleMuonLowMass0/Run2022F-22Sep2023-v1/NANOAOD",
                    "/ParkingDoubleMuonLowMass1/Run2022F-22Sep2023-v1/NANOAOD",
                    "/ParkingDoubleMuonLowMass2/Run2022F-22Sep2023-v1/NANOAOD",
                    "/ParkingDoubleMuonLowMass3/Run2022F-22Sep2023-v1/NANOAOD",
                    "/ParkingDoubleMuonLowMass4/Run2022F-22Sep2023-v1/NANOAOD",
                    "/ParkingDoubleMuonLowMass5/Run2022F-22Sep2023-v1/NANOAOD",
                    "/ParkingDoubleMuonLowMass6/Run2022F-22Sep2023-v1/NANOAOD",
                    "/ParkingDoubleMuonLowMass7/Run2022F-22Sep2023-v1/NANOAOD",
                ],
                "ParkingDoubleMuonLowMass_Run2022G": [
                    "/ParkingDoubleMuonLowMass0/Run2022G-22Sep2023-v1/NANOAOD",
                    "/ParkingDoubleMuonLowMass1/Run2022G-22Sep2023-v1/NANOAOD",
                    "/ParkingDoubleMuonLowMass2/Run2022G-22Sep2023-v1/NANOAOD",
                    "/ParkingDoubleMuonLowMass3/Run2022G-22Sep2023-v1/NANOAOD",
                    "/ParkingDoubleMuonLowMass4/Run2022G-22Sep2023-v2/NANOAOD",
                    "/ParkingDoubleMuonLowMass5/Run2022G-22Sep2023-v1/NANOAOD",
                    "/ParkingDoubleMuonLowMass6/Run2022G-22Sep2023-v1/NANOAOD",
                    "/ParkingDoubleMuonLowMass7/Run2022G-22Sep2023-v1/NANOAOD"
                ],
            },
            "HH": {
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM",
            },
            "Hbb": {
                "GluGluHto2B_PT-200_M-125": [
                    "/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
                    "/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM",
                ],
            },
        },
        "2023": {
            "ParkingHH": {
                "ParkingHH_Run2023Cv1": [
                    "/ParkingHH/Run2023D-22Sep2023_v1-v1/NANOAOD",
                ],
                "ParkingHH_Run2023Cv2": [
                    "/ParkingHH/Run2023D-22Sep2023_v2-v1/NANOAOD",
                ],
                "ParkingHH_Run2023Cv3": [
                    "/ParkingHH/Run2023C-22Sep2023_v3-v1/NANOAOD",
                ],
                "ParkingHH_Run2023Cv4": [
                    "/ParkingHH/Run2023C-22Sep2023_v4-v1/NANOAOD",
                ],
            },
            "ParkingVBF": {
                # "ParkingVBF_Run2023Cv1": [
                #     "/ParkingVBF0/Run2023C-22Sep2023_v1-v1/NANOAOD",
                #     "/ParkingVBF1/Run2023C-22Sep2023_v1-v1/NANOAOD",
                #     "/ParkingVBF2/Run2023C-22Sep2023_v1-v1/NANOAOD",
                #     "/ParkingVBF3/Run2023C-22Sep2023_v1-v1/NANOAOD",
                #     "/ParkingVBF4/Run2023C-22Sep2023_v1-v1/NANOAOD",
                #     "/ParkingVBF5/Run2023C-22Sep2023_v1-v1/NANOAOD",
                #     "/ParkingVBF6/Run2023C-22Sep2023_v1-v1/NANOAOD",
                #     "/ParkingVBF7/Run2023C-22Sep2023_v1-v1/NANOAOD",
                # ],
                # "ParkingVBF_Run2023Cv2": [
                #     "/ParkingVBF0/Run2023C-22Sep2023_v2-v1/NANOAOD",
                #     "/ParkingVBF1/Run2023C-22Sep2023_v2-v1/NANOAOD",
                #     "/ParkingVBF2/Run2023C-22Sep2023_v2-v1/NANOAOD",
                #     "/ParkingVBF3/Run2023C-22Sep2023_v2-v1/NANOAOD",
                #     "/ParkingVBF4/Run2023C-22Sep2023_v2-v1/NANOAOD",
                #     "/ParkingVBF5/Run2023C-22Sep2023_v2-v1/NANOAOD",
                #     "/ParkingVBF6/Run2023C-22Sep2023_v2-v1/NANOAOD",
                #     "/ParkingVBF7/Run2023C-22Sep2023_v2-v1/NANOAOD",
                # ],
                # "ParkingVBF_Run2023Cv3": [
                #     "/ParkingVBF0/Run2023C-22Sep2023_v3-v1/NANOAOD",
                #     "/ParkingVBF1/Run2023C-22Sep2023_v3-v1/NANOAOD",
                #     "/ParkingVBF2/Run2023C-22Sep2023_v3-v1/NANOAOD",
                #     "/ParkingVBF3/Run2023C-22Sep2023_v3-v1/NANOAOD",
                #     "/ParkingVBF4/Run2023C-22Sep2023_v3-v1/NANOAOD",
                #     "/ParkingVBF5/Run2023C-22Sep2023_v3-v1/NANOAOD",
                #     "/ParkingVBF6/Run2023C-22Sep2023_v3-v1/NANOAOD",
                #     "/ParkingVBF7/Run2023C-22Sep2023_v3-v1/NANOAOD",
                # ],
                # "ParkingVBF_Run2023Cv4": [
                #     "/ParkingVBF0/Run2023C-22Sep2023_v4-v1/NANOAOD",
                #     "/ParkingVBF1/Run2023C-22Sep2023_v4-v1/NANOAOD",
                #     "/ParkingVBF2/Run2023C-22Sep2023_v4-v1/NANOAOD",
                #     "/ParkingVBF3/Run2023C-22Sep2023_v4-v1/NANOAOD",
                #     "/ParkingVBF4/Run2023C-22Sep2023_v4-v1/NANOAOD",
                #     "/ParkingVBF5/Run2023C-22Sep2023_v4-v1/NANOAOD",
                #     "/ParkingVBF6/Run2023C-22Sep2023_v4-v1/NANOAOD",
                #     "/ParkingVBF7/Run2023C-22Sep2023_v4-v1/NANOAOD",
                # ],
                "ParkingVBF_Run2023Cv3": [
                    "/ParkingVBF0/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
                    "/ParkingVBF1/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
                    "/ParkingVBF2/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
                    "/ParkingVBF3/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
                    "/ParkingVBF4/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
                    "/ParkingVBF5/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
                    "/ParkingVBF6/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
                    "/ParkingVBF7/Run2023C-PromptNanoAODv12_v3-v1/NANOAOD",
                ],
                "ParkingVBF_Run2023Cv4": [
                    "/ParkingVBF0/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
                    "/ParkingVBF1/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
                    "/ParkingVBF2/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
                    "/ParkingVBF3/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
                    "/ParkingVBF4/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
                    "/ParkingVBF5/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
                    "/ParkingVBF6/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
                    "/ParkingVBF7/Run2023C-PromptNanoAODv12_v4-v1/NANOAOD",
                ],
            },
            "JetMET": {
                "JetMET_Run2023Cv1": [
                    "/JetMET0/Run2023C-22Sep2023_v1-v1/NANOAOD",
                    "/JetMET1/Run2023C-22Sep2023_v1-v1/NANOAOD",
                ],
                "JetMET_Run2023Cv2": [
                    "/JetMET0/Run2023C-22Sep2023_v2-v1/NANOAOD",
                    "/JetMET1/Run2023C-22Sep2023_v2-v1/NANOAOD",
                ],
                "JetMET_Run2023Cv3": [
                    "/JetMET0/Run2023C-22Sep2023_v3-v1/NANOAOD",
                    "/JetMET1/Run2023C-22Sep2023_v3-v1/NANOAOD",
                ],
                "JetMET_Run2023Cv4": [
                    "/JetMET0/Run2023C-22Sep2023_v4-v1/NANOAOD",
                    "/JetMET1/Run2023C-22Sep2023_v4-v1/NANOAOD",
                ],
            },
            "Muon": {
                "Muon_Run2023Cv1": [
                    "/Muon0/Run2023C-22Sep2023_v1-v1/NANOAOD",
                    "/Muon1/Run2023C-22Sep2023_v1-v1/NANOAOD",
                ],
                "Muon_Run2023Cv2": [
                    "/Muon0/Run2023C-22Sep2023_v2-v1/NANOAOD",
                    "/Muon1/Run2023C-22Sep2023_v2-v1/NANOAOD",
                ],
                "Muon_Run2023Cv3": [
                    "/Muon0/Run2023C-22Sep2023_v3-v1/NANOAOD",
                    "/Muon1/Run2023C-22Sep2023_v3-v1/NANOAOD",
                ],
                "Muon_Run2023Cv4": [
                    "/Muon0/Run2023C-22Sep2023_v4-v1/NANOAOD",
                    "/Muon1/Run2023C-22Sep2023_v4-v1/NANOAOD",
                ],
            },
            "QCD": {
                "QCD_HT-1000to1200": "/QCD-4Jets_HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-100to200": "/QCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-1200to1500": "/QCD-4Jets_HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-1500to2000": "/QCD-4Jets_HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "QCD_HT-2000": "/QCD-4Jets_HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-200to400": "/QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-400to600": "/QCD-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "QCD_HT-40to70": "/QCD-4Jets_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-600to800": "/QCD-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-70to100": "/QCD-4Jets_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "QCD_HT-800to1000": "/QCD-4Jets_HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
            },
            "HH": {
                "GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV": "/GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv13-133X_mcRun3_2023_realistic_ForNanov13_v1-v3/NANOAODSIM",
            },
            "Hbb": {
                "GluGluHto2B_PT-200_M-125": "/GluGluHto2B_PT-200_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
            },
            "VJets": {
                # LO
                "Wto2Q-3Jets_HT-200to400": "/Wto2Q-3Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-400to600": "/Wto2Q-3Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-600to800": "/Wto2Q-3Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "Wto2Q-3Jets_HT-800": "/Wto2Q-3Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                # NLO
                "Wto2Q-2Jets_PTQQ-100to200_1J": "/Wto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-100to200_2J": "/Wto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-200to400_1J": "/Wto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-200to400_2J": "/Wto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-400to600_1J": "/Wto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-400to600_2J": "/Wto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-600_1J": "/Wto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Wto2Q-2Jets_PTQQ-600_2J": "/Wto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                # NLO
                "Zto2Q-2Jets_PTQQ-100to200_1J": "/Zto2Q-2Jets_PTQQ-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-100to200_2J": "/Zto2Q-2Jets_PTQQ-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-200to400_1J": "/Zto2Q-2Jets_PTQQ-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-200to400_2J": "/Zto2Q-2Jets_PTQQ-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-400to600_1J": "/Zto2Q-2Jets_PTQQ-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-400to600_2J": "/Zto2Q-2Jets_PTQQ-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-600_1J": "/Zto2Q-2Jets_PTQQ-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Q-2Jets_PTQQ-600_2J": "/Zto2Q-2Jets_PTQQ-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                # LO
                "Zto2Q-4Jets_HT-200to400": "/Zto2Q-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-400to600": "/Zto2Q-4Jets_HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-600to800": "/Zto2Q-4Jets_HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM",
                "Zto2Q-4Jets_HT-800": "/Zto2Q-4Jets_HT-800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM",
                # WJetsToLnu NLO
                "WtoLNu-2Jets_0J": "/WtoLNu-2Jets_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-2Jets_1J": "/WtoLNu-2Jets_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "WtoLNu-2Jets_2J": "/WtoLNu-2Jets_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "WtoLNu-2Jets_PTLNu-100to200_1J": "/WtoLNu-2Jets_PTLNu-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-2Jets_PTLNu-100to200_2J": "/WtoLNu-2Jets_PTLNu-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-2Jets_PTLNu-200to400_1J": "/WtoLNu-2Jets_PTLNu-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-2Jets_PTLNu-200to400_2J": "/WtoLNu-2Jets_PTLNu-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-2Jets_PTLNu-400to600_1J": "/WtoLNu-2Jets_PTLNu-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-2Jets_PTLNu-400to600_2J": "/WtoLNu-2Jets_PTLNu-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-2Jets_PTLNu-40to100_1J": "/WtoLNu-2Jets_PTLNu-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-2Jets_PTLNu-40to100_2J": "/WtoLNu-2Jets_PTLNu-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-2Jets_PTLNu-600_1J": "/WtoLNu-2Jets_PTLNu-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-2Jets_PTLNu-600_2J": "/WtoLNu-2Jets_PTLNu-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                # WJetsToLnu LO
                "WtoLNu-4Jets": "/WtoLNu-4Jets_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-4Jets_1J": "/WtoLNu-4Jets_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-4Jets_2J": "/WtoLNu-4Jets_2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-4Jets_3J": "/WtoLNu-4Jets_3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-4Jets_4J": "/WtoLNu-4Jets_4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-4Jets_MLNu-0to120_HT-40to100": "/WtoLNu-4Jets_MLNu-0to120_HT-40to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "WtoLNu-4Jets_MLNu-0to120_HT-100to400": "/WtoLNu-4Jets_MLNu-0to120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "WtoLNu-4Jets_MLNu-0to120_HT-400to800": "/WtoLNu-4Jets_MLNu-0to120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "WtoLNu-4Jets_MLNu-120to200": "/WtoLNu-4Jets_MLNu-120to200_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v5/NANOAODSIM",
                "WtoLNu-4Jets_MLNu-200to400": "/WtoLNu-4Jets_MLNu-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v5/NANOAODSIM",
                "WtoLNu-4Jets_MLNu-400to800": "/WtoLNu-4Jets_MLNu-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM",
                "WtoLNu-4Jets_MLNu-800to1500": "/WtoLNu-4Jets_MLNu-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM",
                "WtoLNu-4Jets_MLNu-1500to2500": "/WtoLNu-4Jets_MLNu-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "WtoLNu-4Jets_MLNu-2500to4000": "/WtoLNu-4Jets_MLNu-2500to4000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v5/NANOAODSIM",
                "WtoLNu-4Jets_MLNu-4000to6000": "/WtoLNu-4Jets_MLNu-4000to6000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v5/NANOAODSIM",
                "WtoLNu-4Jets_MLNu-6000": "/WtoLNu-4Jets_MLNu-6000_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM",
                # DYJetsTonunu NLO
                "Zto2Nu-2Jets_PTNuNu-100to200_1J": "/Zto2Nu-2Jets_PTNuNu-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Nu-2Jets_PTNuNu-100to200_2J": "/Zto2Nu-2Jets_PTNuNu-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Nu-2Jets_PTNuNu-200to400_1J": "/Zto2Nu-2Jets_PTNuNu-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Nu-2Jets_PTNuNu-200to400_2J": "/Zto2Nu-2Jets_PTNuNu-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Nu-2Jets_PTNuNu-400to600_1J": "/Zto2Nu-2Jets_PTNuNu-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Nu-2Jets_PTNuNu-400to600_2J": "/Zto2Nu-2Jets_PTNuNu-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Nu-2Jets_PTNuNu-40to100_1J": "/Zto2Nu-2Jets_PTNuNu-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Nu-2Jets_PTNuNu-40to100_2J": "/Zto2Nu-2Jets_PTNuNu-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Nu-2Jets_PTNuNu-600_1J": "/Zto2Nu-2Jets_PTNuNu-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                "Zto2Nu-2Jets_PTNuNu-600_2J": "/Zto2Nu-2Jets_PTNuNu-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
                # DYJetsTonunu LO
                "Zto2Nu-4Jets_HT-100to200": "/Zto2Nu-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "Zto2Nu-4Jets_HT-200to400": "/Zto2Nu-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "Zto2Nu-4Jets_HT-400to800": "/Zto2Nu-4Jets_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "Zto2Nu-4Jets_HT-800to1500": "/Zto2Nu-4Jets_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "Zto2Nu-4Jets_HT-1500to2500": "/Zto2Nu-4Jets_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM",
                "Zto2Nu-4Jets_HT-2500": "/Zto2Nu-4Jets_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM",
            },
        },
    }


def eos_rec_search(startdir, suffix, dirs):
    print(f"EOS Recursive search in {startdir}.")
    eosbase = "root://cmseos.fnal.gov/"
    try:
        dirlook = (
            subprocess.check_output(f"eos {eosbase} ls {startdir}", shell=True)
            .decode("utf-8")
            .split("\n")[:-1]
        )
    except:
        print(f"No files found for {startdir}")
        return dirs

    donedirs = [[] for d in dirlook]
    for di, d in enumerate(dirlook):
        # print(f"Looking in {dirlook}.")
        if d.endswith(suffix):
            donedirs[di].append(startdir + "/" + d)
        elif d == "log":
            continue
        else:
            donedirs[di] = donedirs[di] + eos_rec_search(
                startdir + "/" + d, suffix, dirs + donedirs[di]
            )
    donedir = [d for da in donedirs for d in da]
    return dirs + donedir


def get_files(dataset, version):
    if "private" in version:
        files = eos_rec_search(dataset, ".root", [])
        return [f"root://cmseos.fnal.gov/{f}" for f in files]
    else:
        import requests
        from rucio_utils import get_dataset_files, get_proxy_path

        proxy = get_proxy_path()
        if "USER" in dataset:
            link = f"https://cmsweb.cern.ch:8443/dbs/prod/phys03/DBSReader/files?dataset={dataset}&detail=True"
        else:
            link = f"https://cmsweb.cern.ch:8443/dbs/prod/global/DBSReader/files?dataset={dataset}&detail=True"
        r = requests.get(
            link,
            cert=proxy,
            verify=False,
        )
        filesjson = r.json()
        files = []
        not_valid = []
        for fj in filesjson:
            if "is_file_valid" in fj:
                if fj["is_file_valid"] == 0:
                    # print(f"ERROR: File not valid on DAS: {fj['logical_file_name']}")
                    not_valid.append(fj["logical_file_name"])
                else:
                    files.append(fj["logical_file_name"])
            else:
                continue

        if "USER" in dataset:
            files_valid = [f"root://cmseos.fnal.gov/{f}" for f in files]
            return files_valid

        if len(files) == 0:
            print(f"Found 0 files for sample {dataset}!")
            return []

        # Now query rucio to get the concrete dataset passing the sites filtering options
        sites_cfg = {
            "whitelist_sites": [],
            "blacklist_sites": [
                "T2_FR_IPHC" "T2_US_MIT",
                "T2_US_Vanderbilt",
                "T2_UK_London_Brunel",
                "T2_UK_SGrid_RALPP",
                "T1_UK_RAL_Disk",
                "T2_PT_NCG_Lisbon",
            ],
            "regex_sites": None,
        }
        if version == "v12" or version == "v11":
            sites_cfg["whitelist_sites"] = ["T1_US_FNAL_Disk"]

        files_rucio, sites = get_dataset_files(dataset, **sites_cfg, output="first")

        # print(dataset, sites)

        # Get rid of invalid files
        files_valid = []
        for f in files_rucio:
            invalid = False
            for nf in not_valid:
                if nf in f:
                    invalid = True
                    break
            if not invalid:
                files_valid.append(f)

        return files_valid


for version in ["v12"]:
    datasets = globals()[f"get_{version}"]()
    index = datasets.copy()
    for year, ydict in datasets.items():
        print(year)
        for sample, sdict in ydict.items():
            print(sample)
            for sname, dataset in sdict.items():
                if isinstance(dataset, list):
                    files = []
                    for d in dataset:
                        files.extend(get_files(d, version))
                    index[year][sample][sname] = files
                else:
                    index[year][sample][sname] = get_files(dataset, version)

    with Path(f"nanoindex_{version}.json").open("w") as f:
        json.dump(index, f, indent=4)
