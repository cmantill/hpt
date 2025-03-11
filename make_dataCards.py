import pickle
import numpy as np
import rhalphalib as rl
import scipy.stats
import ROOT
import os
import sys
from pathlib import Path

# Initialize rhalphalib
rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = False



# Define output directory
output_dir = Path("rhalphabet_datacards")
output_dir.mkdir(exist_ok=True)

# Function to load and structure templates like get_templates
def get_templates(years):
    templates_summed = {}

    for year in years:
        # Load histograms
        histogram_file = f"histograms_{year}.pkl"
        with open(histogram_file, "rb") as f:
            histograms = pickle.load(f)

        for process, hist_dict in histograms.items():
            # Ensure process is initialized in templates_summed
            if process not in templates_summed:
                templates_summed[process] = {"pass": {}, "fail": {}}  # Initialize regions

            for region in ["pass", "fail"]:
                if region not in hist_dict:
                    print(f"Warning: '{region}' missing in {process} for {year}. Skipping.")
                    continue  # Skip if region is missing

                for ptbin, hist in hist_dict[region].items():
                    if ptbin not in templates_summed[process][region]:
                        templates_summed[process][region][ptbin] = hist.copy()  # Initialize
                    else:
                        templates_summed[process][region][ptbin] += hist  # Sum across years

    return templates_summed

def get_hist(process, region, ptbin, histograms):
    """
    Retrieve the histogram for a given process, region, and ptbin.

    Args:
        process (str): Name of the process (e.g., "QCD", "Wto2Q", etc.).
        region (str): Either "pass" or "fail".
        ptbin (int): The pt bin key to retrieve from the histogram.
        histograms (dict): Dictionary containing all histograms.

    Returns:
        tuple: (sumw, binning, obs_name, sumw2) where:
            - sumw: The histogram bin values.
            - binning: The bin edges.
            - obs_name: The observable name.
            - sumw2: The bin errors squared (if available).
    """
    if process not in histograms:
        raise KeyError(f"Process '{process}' not found in histograms.")

    if region not in histograms[process]:
        raise KeyError(f"Region '{region}' not found in histograms for process '{process}'.")

    if ptbin not in histograms[process][region]:
        raise KeyError(f"Ptbin '{ptbin}' not found in histograms for process '{process}', region '{region}'.")

    hist = histograms[process][region][ptbin]
    print(f"hist variances: {hist.variances()}")  # Print variances
    sumw2 = hist.variances(flow=False) if hist.variances() is not None else np.zeros_like(hist)  # Variances
    print(f"sumw2: {sumw2}")

    binning = hist.axes[0].edges  # Get bin edges
    obs_name = hist.axes[0].name  # Observable name (e.g., "msd")


    return (np.array(hist), binning, obs_name, np.array(sumw2))

import ROOT
import numpy as np

def get_histogram_from_root(year, category, pass_fail, ptbin, obs):
    """
    Retrieve histograms from a ROOT file and return it as a numpy-compatible format.

    Parameters:
    - year (str): Year of the dataset.
    - category (str): Sample category (e.g., "QCD").
    - pass_fail (str): "pass" or "fail" selection.
    - ptbin (int): The pt bin value.
    - obs (object): Observable with binning information.

    Returns:
    - tuple: (sumw, binning, obs_name, sumw2) where:
      - sumw is the bin content (numpy array)
      - binning is the bin edges from the observable
      - obs_name is the observable name
      - sumw2 is the sum of squared weights (errors)
    """


    # Open ROOT file
    root_filename = f"histograms_{year}.root"
    f = ROOT.TFile.Open(root_filename, "READ")

    if not f or f.IsZombie():
        raise FileNotFoundError(f"Could not open ROOT file: {root_filename}")

    # Construct histogram name
    hist_name = f"{category}_{pass_fail}_{ptbin}"
    print(f"Extracting histogram: {hist_name}")

    # Get the histogram
    h = f.Get(hist_name)
    if not h:
        raise ValueError(f"Histogram '{hist_name}' not found in {root_filename}")

    # Extract bin edges directly from ROOT histogram
    binning = np.array([h.GetBinLowEdge(i) for i in range(1, h.GetNbinsX() + 2)])

    # Prepare sumw and sumw2 arrays
    sumw = np.array([h.GetBinContent(i) for i in range(1, h.GetNbinsX() + 1)])
    sumw2 = np.array([h.GetBinError(i)**2 for i in range(1, h.GetNbinsX() + 1)])

    # Close the ROOT file
    f.Close()

    return (sumw, binning, obs.name, sumw2)

def model_rhalphabet():

    ######## SETTING IT UP ########
    years = ["2023"]
    # Extract structured templates
    #histograms = get_templates(years)
    #print(histograms)

    jec = rl.NuisanceParameter("CMS_jec", "lnN")
    massScale = rl.NuisanceParameter("CMS_msdScale", "shape")
    lumi = rl.NuisanceParameter("CMS_lumi", "lnN")
    tqqeffSF = rl.IndependentParameter("tqqeffSF", 1.0, 0, 10)
    tqqnormSF = rl.IndependentParameter("tqqnormSF", 1.0, 0, 10)

    # Define the pt bins
    ptbins = np.array([450, 500, 550, 600, 675, 800, 1200]) #  500, 550, 600, 675, 800,
    npt = len(ptbins) - 1
    # Extract binning information
    msdbins = np.linspace(40, 201, 25)
    msd = rl.Observable("msd", msdbins)

    # FOR PICKLES
    #msd_axis = histograms["QCD"]["pass"][ptbins[0]].axes[0]  # First pt bin for binning info
    #msd_binning = msd_axis.edges


    # here we derive these all at once with 2D array
    ptpts, msdpts = np.meshgrid(ptbins[:-1] + 0.3 * np.diff(ptbins), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing="ij")
    rhopts = 2 * np.log(msdpts / ptpts)
    ptscaled = (ptpts - 450.0) / (1200.0 - 450.0)
    rhoscaled = (rhopts - (-6)) / ((-2.1) - (-6))
    validbins = (rhoscaled >= 0) & (rhoscaled <= 1)
    rhoscaled[~validbins] = 1  # we will mask these out later

    ######## BUILDING THE MODEL ########
    # Build the QCD MC pass+fail model and fit to polynomial
    qcdmodel = rl.Model("qcdmodel")
    qcdpass, qcdfail = 0.0, 0.0

    for ptbin in ptbins[:-1]:
        failCh = rl.Channel(f"ptbin{ptbin}fail")
        passCh = rl.Channel(f"ptbin{ptbin}pass")
        qcdmodel.addChannel(failCh)
        qcdmodel.addChannel(passCh)

        # FOR PICKLES
        #failTempl = get_hist("QCD", "fail", ptbin, histograms)
        #passTempl = get_hist("QCD", "pass", ptbin, histograms)
        
        # FOR ROOT
        failTempl = get_histogram_from_root("2023", "QCD", "fail", ptbin, msd)
        passTempl = get_histogram_from_root("2023", "QCD", "pass", ptbin, msd)

        failCh.setObservation(failTempl, read_sumw2=True)
        passCh.setObservation(passTempl, read_sumw2=True)

        qcdfail += sum([val for val in failCh.getObservation()[0]])
        qcdpass += sum([val for val in passCh.getObservation()[0]])

    # Compute QCD efficiency
    qcdeff = qcdpass / qcdfail
    #tf_MCtempl = rl.BernsteinPoly("tf_MCtempl", (2, 2), ["pt", "rho"], limits=(0, 10))

    tf_MCtempl = rl.BasisPoly("tf_MCtempl", (2,2), ["pt", "rho"], basis='Bernstein', limits=(0, 10))

    tf_MCtempl_params = qcdeff * tf_MCtempl(ptscaled, rhoscaled)

    #print(f"Available qcdmodel channels: {qcdmodel.channels}")
    # Apply transfer function to QCD
    for ptbin in ptbins[:-1]:
        failCh = qcdmodel[f"ptbin{ptbin}fail"]
        passCh = qcdmodel[f"ptbin{ptbin}pass"]
        failObs = failCh.getObservation()
        failObs = failObs[0]  # Extract sumw (bin counts)
        qcdparams = np.array([
            rl.IndependentParameter(f"qcdparam_ptbin{ptbin}_msdbin{i}", 0)
            for i in range(msd.nbins)
        ])
        sigmascale = 10.0
        scaledparams = failObs * (1 + sigmascale / np.maximum(1.0, np.sqrt(failObs))) ** qcdparams

        #fail_qcd = rl.ParametericSample(f"{failCh.name}qcd", rl.Sample.BACKGROUND, msd, scaledparams)
        fail_qcd = rl.ParametericSample(f"{failCh.name}_qcd".replace(" ", ""), rl.Sample.BACKGROUND, msd, scaledparams)
        print(f"failCh.name: '{failCh.name}'")
        print(f"fail_qcd.name: '{fail_qcd.name}'")
        print(f"Does it match? {fail_qcd.name.startswith(failCh.name)}")

        failCh.addSample(fail_qcd)
        print(f"ptscaled: {ptscaled}")
        #pass_qcd = rl.TransferFactorSample(f"ptbin{ptbin}pass", rl.Sample.BACKGROUND, tf_MCtempl_params[ptbin], fail_qcd)
        # Convert ptbin (which is a bin edge) to an index
        ptbin_index = np.where(ptbins[:-1] == ptbin)[0][0]  # Find the index

        # Debugging print
        print(f"ptbin: {ptbin}, converted to index: {ptbin_index}")

        # Use the index instead of the bin lower edge
        pass_qcd = rl.TransferFactorSample(f"{passCh.name}_qcd".replace(" ", ""), rl.Sample.BACKGROUND, tf_MCtempl_params[ptbin_index], fail_qcd)
        #pass_qcd = rl.TransferFactorSample(f"{passCh.name}_qcd".replace(" ", ""), rl.Sample.BACKGROUND, tf_MCtempl_params[], fail_qcd)
        passCh.addSample(pass_qcd)

    # Fit QCD Model with RooFit
    qcdfit_ws = ROOT.RooWorkspace("qcdfit_ws")
    simpdf, obs = qcdmodel.renderRoofit(qcdfit_ws)
    qcdfit = simpdf.fitTo(
        obs,
        ROOT.RooFit.Extended(True),
        ROOT.RooFit.SumW2Error(True),
        ROOT.RooFit.Strategy(2),
        ROOT.RooFit.Save(),
        ROOT.RooFit.Minimizer("Minuit2", "migrad"),
        ROOT.RooFit.PrintLevel(-1),
    )
    qcdfit_ws.add(qcdfit)
    if "pytest" not in sys.modules:
        qcdfit_ws.writeToFile(str(output_dir / "qcdfit.root"))
    if qcdfit.status() != 0:
        raise RuntimeError("Could not fit QCD")
    
    param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
    decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(tf_MCtempl.name + "_deco", qcdfit, param_names)
    tf_MCtempl.parameters = decoVector.correlated_params.reshape(tf_MCtempl.parameters.shape)
    tf_MCtempl_params_final = tf_MCtempl(ptscaled, rhoscaled)
    tf_dataResidual = rl.BernsteinPoly("tf_dataResidual", (2, 2), ["pt", "rho"], limits=(0, 10))
    tf_dataResidual_params = tf_dataResidual(ptscaled, rhoscaled)
    tf_params = qcdeff * tf_MCtempl_params_final * tf_dataResidual_params


    ######## BUILDING THE ACTUAL FIT MODEL ########
    # Build the signal model   
    model = rl.Model("testModel")
    sigs = ['WH', 'ZH', 'VBF', 'ttH', 'ggH', 'data']
    for ptbin in range(npt):
        for region in ["pass", "fail"]:
            ch = rl.Channel(f"ptbin{ptbin}_{region}")
            model.addChannel(ch)

            for process, templ in histograms.items():
                if process == "QCD":
                    continue

                templ = templ[region]#[ptbin]
                stype = rl.Sample.SIGNAL if process in sigs else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(f"ptbin{ptbin}_{region}_{process}", stype, templ)

                # mock systematics
                jecup_ratio = np.random.normal(loc=1, scale=0.05, size=msd.nbins)
                msdUp = np.linspace(0.9, 1.1, msd.nbins)
                msdDn = np.linspace(1.2, 0.8, msd.nbins)

                # for jec we set lnN prior, shape will automatically be converted to norm systematic
                sample.setParamEffect(jec, jecup_ratio)
                sample.setParamEffect(massScale, msdUp, msdDn)
                sample.setParamEffect(lumi, 1.027)

                ch.addSample(sample)

            # Set observed data
            data_obs = templates_summed[region][ptbin]
            ch.setObservation(data_obs)

    # Define QCD model
    for ptbin in range(npt):
        failCh = model[f"ptbin{ptbin}_fail"]
        passCh = model[f"ptbin{ptbin}_pass"]
        failObs = failCh.getObservation()
        initial_qcd = failObs.astype(float)
        for sample in failCh:
            initial_qcd -= sample.getExpectation(nominal=True)
        if np.sum(initial_qcd) == 0:
            raise ValueError("Initial QCD prediction is zero. This is likely a problem.")
        qcdparams = np.array([
            rl.IndependentParameter(f"qcdparam_ptbin{ptbin}_msdbin{i}", 0)
            for i in range(msd.nbins)
        ])
        sigmascale = 10.0
        scaledparams = failObs * (1 + sigmascale / np.maximum(1.0, np.sqrt(failObs))) ** qcdparams
        fail_qcd = rl.ParametericSample(f"ptbin{ptbin}_fail_qcd", rl.Sample.BACKGROUND, msd, scaledparams)
        failCh.addSample(fail_qcd)
        pass_qcd = rl.TransferFactorSample(f"ptbin{ptbin}_pass_qcd", rl.Sample.BACKGROUND, tf_params[ptbin], fail_qcd)
        passCh.addSample(pass_qcd)

        tqqpass = passCh["tqq"]
        tqqfail = failCh["tqq"]
        tqqPF = tqqpass.getExpectation(nominal=True).sum() / tqqfail.getExpectation(nominal=True).sum()
        tqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
        tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
        tqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
        tqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)

    # Save workspace and model
    ws_path = output_dir / "testModel.json"
    pkl_path = output_dir / "testModel.pkl"

    with open(pkl_path, "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine(ws_path)
    print(f"Workspace saved: {ws_path}")
    print(f"Model pickle saved: {pkl_path}")



if __name__ == "__main__":
    model_rhalphabet()


############################################################################################################

