import sys
import os
import json
import ROOT  # Import ROOT for histogram handling
import argparse
from hpt import utils
import numpy as np

# Argument parser for command-line input
parser = argparse.ArgumentParser(description="Process histograms for a given year.")
parser.add_argument("year", type=str, help="Year for data processing")
args = parser.parse_args()

# Use the provided year
year = args.year  # Example: "2023"

MAIN_DIR = "/Users/gbibim/Here/genZ/data"
dir_name = "PNetchildren"
path_to_dir = f"{MAIN_DIR}/{dir_name}/"

# Load samples from JSON file
args.samples_file = "samples.json"
with open(args.samples_file, "r") as f:
    samples = json.load(f)

dirs = {path_to_dir: samples}

# Define columns to load
load_columns = [
    ("weight", 1),
    ("ak8FatJetPt", 1),
    ("ak8FatJetmsoftdrop", 1),
    ('ak8FatJetParTPQCD1HF', 1),
    ('ak8FatJetParTPQCD2HF', 1),
    ('ak8FatJetParTPQCD0HF', 1),
    ('ak8FatJetParTPXbb', 1),
    ("AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35", 1), # for 2022 and a small fraction of 2023
    ("AK8PFJet230_SoftDropMass40_PNetBB0p06", 1), #new for 2023
]

load_columns_V = load_columns + [
    ("GenVPt", 1),
    ("GenVis_bb", 1),
    ("GenVis_cc", 1),
    ("GenVis_cs", 1),
]

# Define pt bins
ptbins = np.array([450, 500, 550, 600, 675, 800, 1200])
msd_nbins = 24  # Number of bins
msd_min, msd_max = 40, 201  # Range of mSD

# Initialize a dictionary to store histograms
histograms = {}

for category in samples.keys():
    histograms[category] = {
        "pass": {bin_edge: ROOT.TH1F(f"{category}_pass_{bin_edge}", f"{category} Pass {bin_edge}",
                                     msd_nbins, msd_min, msd_max) for bin_edge in ptbins[:-1]},
        "fail": {bin_edge: ROOT.TH1F(f"{category}_fail_{bin_edge}", f"{category} Fail {bin_edge}",
                                     msd_nbins, msd_min, msd_max) for bin_edge in ptbins[:-1]},
    }

# Function to find the pt bin for each event
def get_ptbin(pt):
    bins = np.digitize(pt, ptbins) - 1
    bins[bins < 0] = 0  # Assign lowest bin if pt is too small
    bins[bins >= len(ptbins) - 1] = len(ptbins) - 2  # Assign highest valid bin
    return ptbins[bins]  # Return corresponding bin lower edge

# Function to fill ROOT histograms
def fill_mass(events, category, sample):
    for key, data in events.items():
        weight = data["finalWeight"]
        msd = data["ak8FatJetmsoftdrop"][0]
        pt = data["ak8FatJetPt"][0]
        Pxbb = data["ak8FatJetParTPXbb"][0]
        PQCD = (
            data["ak8FatJetParTPQCD1HF"][0]
            + data["ak8FatJetParTPQCD2HF"][0]
            + data["ak8FatJetParTPQCD0HF"][0]
        )

        # Compute discriminator
        Txbb = Pxbb / (Pxbb + PQCD)

        # Apply selection criteria
        HLTs = (data["AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35"][0] |
                data["AK8PFJet230_SoftDropMass40_PNetBB0p06"][0])
        
        selection = (msd > 40) & HLTs & (Txbb > 0.95) & (pt > 450)
        fail = (Txbb < 0.95) & (msd > 40) & HLTs & (pt > 450)

        # Get corresponding pt bins
        ptbins_selected = get_ptbin(pt[selection])
        ptbins_fail = get_ptbin(pt[fail])

        # Fill histograms
        for pbin, msd_val, weight_val in zip(ptbins_selected, msd[selection], weight[selection]):
            histograms[category]["pass"][pbin].Fill(msd_val, weight_val)

        for pbin, msd_val, weight_val in zip(ptbins_fail, msd[fail], weight[fail]):
            histograms[category]["fail"][pbin].Fill(msd_val, weight_val)

        # Clear intermediate arrays
        del weight, msd, Pxbb, PQCD, selection, HLTs, fail, ptbins_selected, ptbins_fail

# Process samples
for category, sample_list in samples.items():
    for input_dir, dirs_samples in dirs.items():
        for sample in sample_list:
            try:
                # Load only one sample at a time
                events = utils.load_samples(
                    input_dir,
                    category,
                    [sample],  # List containing a single sample
                    year,
                    columns=utils.format_columns(
                        load_columns_V if category in {"Zto2Q", "Wto2Q"} else load_columns
                    ),
                )

                # Fill histograms with the loaded sample
                fill_mass(events, category, sample)

            except KeyError as e:
                print(f"Warning: Missing key {e} in sample {sample}. Skipping.")

            # Ensure the sample is deleted from memory after use
            del events

# Save histograms to a ROOT file
output_file = f"histograms_{year}.root"
root_file = ROOT.TFile(output_file, "RECREATE")

# Write histograms to the ROOT file
for category in histograms.keys():
    for ptbin, histo in histograms[category]["pass"].items():
        histo.Write()  # Save pass histograms
    for ptbin, histo in histograms[category]["fail"].items():
        histo.Write()  # Save fail histograms

# Close ROOT file
root_file.Close()
print(f"Histograms saved to {output_file}")