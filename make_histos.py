import sys
sys.path.append('/Users/gbibim/Here/hpt')

# List all modules and packages available at this path
import os
print(os.listdir('/Users/gbibim/Here/hpt'))


from hpt import utils
import pickle

#import mplhep as hep
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplhep as hep
from pathlib import Path


import hist

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
plt.rcParams.update({"font.size": 12})
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["grid.color"] = "#CCCCCC"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["figure.edgecolor"] = "none"

pt_axis = hist.axis.Regular(50, 200, 1000, name="pt", label="Jet $p_T$ [GeV]")
trig_axis = hist.axis.Regular(100, 0, 1, name="trigger", label="Trigger")
msd_axis = hist.axis.Regular(32, 40, 200, name="msd", label="mSD [GeV]")                # Soft Drop Mass
xbb_axis = hist.axis.Regular(80, 0, 1, name="xjj", label="Xbb")                         # Discriminator

sample_axis = hist.axis.StrCategory([], name="name", growth=True)                       #samples
proc_axis = hist.axis.StrCategory([], name="process", growth=True)                      #processes: Zto2Q, QCD, TT, Diboson
order_axis = hist.axis.StrCategory([], name="order", growth=True)                       #order: NLO, LO

MAIN_DIR = "/Users/gbibim/Here/genZ/data"
#dir_name = "children" #"new"  # data for older samples new for the files with lhe variables
dir_name = "PNetchildren" 
path_to_dir = f"{MAIN_DIR}/{dir_name}/"
year = "2023"  

# Define the samples and the directories where they are stored
samples = {
    "Zto2Q": [
        "Zto2Q-2Jets_PTQQ-100to200_1J",
        "Zto2Q-2Jets_PTQQ-100to200_2J",
        "Zto2Q-2Jets_PTQQ-200to400_1J",
        "Zto2Q-2Jets_PTQQ-200to400_2J",
        "Zto2Q-2Jets_PTQQ-400to600_1J",
        "Zto2Q-2Jets_PTQQ-400to600_2J",
        "Zto2Q-2Jets_PTQQ-600_1J",
        "Zto2Q-2Jets_PTQQ-600_2J",
        ],
    
    
    "Wto2Q":[
        "Wto2Q-2Jets_PTQQ-100to200_1J",
        "Wto2Q-2Jets_PTQQ-100to200_2J",
        "Wto2Q-2Jets_PTQQ-200to400_1J",
        "Wto2Q-2Jets_PTQQ-200to400_2J",
        "Wto2Q-2Jets_PTQQ-400to600_1J",
        "Wto2Q-2Jets_PTQQ-400to600_2J",
        "Wto2Q-2Jets_PTQQ-600_1J",
        "Wto2Q-2Jets_PTQQ-600_2J",
        ],
        
    "Diboson": {
        "ZZ",
        "WZ",
        "WW",
        "WWto4Q",       
    },

    "TT": {
        "TTto4Q",
        "TTto2L2Nu",
        "TTtoLNu2Q",
    },

    "QCD": {
        "QCD_HT-40to70",
        "QCD_HT-70to100",
        "QCD_HT-100to200",
        "QCD_HT-200to400",
        "QCD_HT-400to600",
        "QCD_HT-600to800",
        "QCD_HT-800to1000",
        "QCD_HT-1000to1200",
        "QCD_HT-1200to1500",
        "QCD_HT-1500to2000",
    },

    "data": {
        "JetMET_Run2023Cv1",
        "JetMET_Run2023Cv2",
        "JetMET_Run2023Cv3",
        "JetMET_Run2023Cv4",
    },
    "ggH": {
        "GluGluHto2B_M-125",
    },
    "WH": {
        "WminusH_Hto2B_Wto2Q_M-125",
        "WminusH_Hto2B_WtoLNu_M-125",
        "WplusH_Hto2B_Wto2Q_M-125",
        "WplusH_Hto2B_WtoLNu_M-125",
    },
    "ZH": {
        "ZH_Hto2B_Zto2Q_M-125",
        "ZH_Hto2B_Zto2L_M-125",
        "ZH_Hto2B_Zto2Nu_M-125",
        "ggZH_Hto2B_Zto2L_M-125",
        "ggZH_Hto2B_Zto2Q_M-125",
        "ggZH_Hto2B_Zto2Nu_M-125",
    },
    "ttH": {    
        "ttHto2B_M-125",
    },

    "VBF": {
        "VBFHto2B_M-125",
    },
  
}

dirs = {path_to_dir: samples}

load_columns = [
    ("weight", 1),
    #("GenVPt", 1),
    ("ak8FatJetPt", 1),
    ("ak8FatJetmsoftdrop", 1),
    ("ak8FatJetPNetMass", 1),
    ("ak8FatJetMass_legacy", 2),
    ("ak8FatJetParTmassRes", 1),
    ("ak8FatJetParTmassVis", 1),
    #("ak8FatJetPNetTXbb", 2),
    #("ak8FatJetPNetTXjj", 2),
    #("ak8FatJetPNetTXcc", 2),
    #("ak8FatJetPNetQCD", 2),
    #("ak8FatJetPNetTXgg", 2),
    ('ak8FatJetParTPQCD1HF', 1),
    ('ak8FatJetParTPQCD2HF', 1),
    ('ak8FatJetParTPQCD0HF', 1),
    ('ak8FatJetParTPXbb', 1),
    #('ak8FatJetParTPXcc', 1),
    #('ak8FatJetParTPXcs', 1),
    #('ak8FatJetParTPXgg', 1),
    #('ak8FatJetParTPXqq', 1),
    ("AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35", 1), # for 2022 and a small fraction of 2023
    ("AK8PFJet230_SoftDropMass40_PNetBB0p06", 1), #new for 2023
    #("AK8PFJet400_SoftDropMass40", 1),
    #("AK8PFJet425_SoftDropMass40", 1),
]

load_columns_V = load_columns + [
    ("GenVPt", 1),
    ("GenVis_bb", 1),
    ("GenVis_cc", 1),
    ("GenVis_cs", 1),
]
    
# Initialize histograms once (these stay in memory)

#h_mass = hist.Hist(msd_axis, mreg_axis, mleg_axis, proc_axis, sample_axis)
#f_mass = hist.Hist(msd_axis, mreg_axis, mleg_axis, proc_axis, sample_axis)


histograms = {}

for category in samples.keys():
    histograms[category] = {
        "pass": hist.Hist(msd_axis, proc_axis, sample_axis),
        "fail": hist.Hist(msd_axis, proc_axis, sample_axis)
    }

# FILL HIGGS
# Define a function to handle the histogram filling logic

def fill_mass(events, zto, sample):
    for key, data in events.items():
        weight = data["finalWeight"]
        msd = data["ak8FatJetmsoftdrop"][0]
        pt = data["ak8FatJetPt"][0]

        #Pxqq = data["ak8FatJetParTPXqq"][0]
        Pxbb = data["ak8FatJetParTPXbb"][0]
        #Pxcc = data["ak8FatJetParTPXcc"][0]
        #Pxgg = data["ak8FatJetParTPXgg"][0]
        #Pxcs = data["ak8FatJetParTPXcs"][0]
        PQCD = (
            data["ak8FatJetParTPQCD1HF"][0]
            + data["ak8FatJetParTPQCD2HF"][0]
            + data["ak8FatJetParTPQCD0HF"][0]
        )

        # Compute discriminators
        #Txqq = (Pxqq + Pxcc) / (Pxqq + Pxcc + PQCD)
        Txbb = Pxbb / (Pxbb + PQCD)
        #Txcc = Pxcc / (Pxcc + PQCD)
        #Txgg = Pxgg / (Pxgg + PQCD)
        #Txcs = Pxcs / (Pxcs + PQCD)

        HLTs = ( data["AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35"][0] |
            data["AK8PFJet230_SoftDropMass40_PNetBB0p06"][0] #|  
            #data["AK8PFJet400_SoftDropMass40"][0] | 
            #data["AK8PFJet425_SoftDropMass40"][0] 
        )


        selection = (msd > 40) & (HLTs) & (Txbb>0.95) & (pt>300) 
        fail = (Txbb<0.95) & (msd > 40) & (HLTs) & (pt>300) 

        # Fill histograms
        histograms[category]["pass"].fill(msd[selection], category, sample, weight=weight[selection])
        histograms[category]["fail"].fill(msd[fail], category, sample, weight=weight[fail])

        # Clear intermediate arrays to save memory
        del weight, msd, Pxbb, PQCD,  selection, HLTs, fail # reg, leg, Txqq, Txbb, Txcc, Txgg, Txcs, TW, TZ, TV,

# Loop through Zto2Q, Wto2Q, QCD... processes
for category, sample_list in samples.items():
    for input_dir, dirs_samples in dirs.items():
        # Loop through each sample individually to avoid loading everything at once
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
                fill_mass(events, category, sample)  # See function definition below
                #fill_discriminator(events, zto, sample)  # See function definition

            except KeyError as e:
                print(f"Warning: Missing key {e} in sample {sample}. Skipping.")

            # Ensure the sample is deleted from memory after use
            del events


# Define the output file
output_file = "histograms.pkl"

# Save histograms to a pickle file
with open(output_file, "wb") as f:
    pickle.dump(histograms, f)

print(f"Histograms saved to {output_file}")
