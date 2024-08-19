"""
Common functions for processors.

Author(s): Raghav Kansal
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from coffea.analysis_tools import PackedSelection

P4 = {
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "pt": "Pt",
}


PAD_VAL = -99999


def pad_val(
    arr: ak.Array,
    target: int,
    value: float = PAD_VAL,
    axis: int = 0,
    to_numpy: bool = True,
    clip: bool = True,
):
    """
    pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
    optionally converts to numpy array
    """
    ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=axis)
    return ret.to_numpy() if to_numpy else ret


def add_selection(
    name: str,
    sel: np.ndarray,
    selection: PackedSelection,
    cutflow: dict,
    isData: bool,
    genWeights: ak.Array = None,
):
    """adds selection to PackedSelection object and the cutflow dictionary"""
    if isinstance(sel, ak.Array):
        sel = sel.to_numpy()

    selection.add(name, sel.astype(bool))
    cutflow[name] = (
        np.sum(selection.all(*selection.names))
        if isData
        # add up genWeights for MC
        else np.sum(genWeights[selection.all(*selection.names)])
    )


def add_selection_no_cutflow(
    name: str,
    sel: np.ndarray,
    selection: PackedSelection,
):
    """adds selection to PackedSelection object"""
    selection.add(name, ak.fill_none(sel, False))


def concatenate_dicts(dicts_list: list[dict[str, np.ndarray]]):
    """given a list of dicts of numpy arrays, concatenates the numpy arrays across the lists"""
    if len(dicts_list) > 1:
        return {
            key: np.concatenate(
                [
                    dicts_list[i][key].reshape(dicts_list[i][key].shape[0], -1)
                    for i in range(len(dicts_list))
                ],
                axis=1,
            )
            for key in dicts_list[0]
        }

    return dicts_list[0]


def select_dicts(dicts_list: list[dict[str, np.ndarray]], sel: np.ndarray):
    """given a list of dicts of numpy arrays, select the entries per array across the lists according to ``sel``"""
    return {
        key: np.stack(
            [
                dicts_list[i][key].reshape(dicts_list[i][key].shape[0], -1)
                for i in range(len(dicts_list))
            ],
            axis=1,
        )[sel]
        for key in dicts_list[0]
    }


def remove_variation_suffix(var: str):
    """removes the variation suffix from the variable name"""
    if var.endswith("Down"):
        return var.split("Down")[0]
    elif var.endswith("Up"):
        return var.split("Up")[0]
    return var


def get_pickles(pickles_path, year, sample_name):
    """Accumulates all pickles in ``pickles_path`` directory"""
    from coffea.processor.accumulator import accumulate

    out_pickles = [f for f in listdir(pickles_path) if f != ".DS_Store"]

    file_name = out_pickles[0]
    with Path(f"{pickles_path}/{file_name}").open("rb") as file:
        # out = pickle.load(file)[year][sample_name]  # TODO: uncomment and delete below
        out = pickle.load(file)[year]
        sample_name = next(iter(out.keys()))
        out = out[sample_name]

    for file_name in out_pickles[1:]:
        try:
            with Path(f"{pickles_path}/{file_name}").open("rb") as file:
                out_dict = pickle.load(file)[year][sample_name]
                out = accumulate([out, out_dict])
        except:
            warnings.warn(f"Not able to open file {pickles_path}/{file_name}", stacklevel=1)
    return out


def _normalize_weights(
    events: pd.DataFrame,
    year: str,
    totals: dict,
    sample: str,
    isData: bool,
    variations: bool = True,
    weight_shifts: dict[str, Syst] = None,
):
    """Normalize weights and all the variations"""
    # don't need any reweighting for data
    if isData:
        events["finalWeight"] = events["weight"]
        return

    # check weights are scaled
    if "weight_noxsec" in events and np.all(events["weight"] == events["weight_noxsec"]):
        # print(sample)
        if sample == "VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8":
            warnings.warn(
                f"Temporarily scaling {sample} by its xsec and lumi - remember to remove after fixing in the processor!",
                stacklevel=0,
            )
            events["weight"] = (
                events["weight"]
                * xsecs["VBFHHto4B_CV_1_C2V_0_C3_1_TuneCP5_13p6TeV_madgraph-pythia8"]
                * LUMI[year]
            )
        else:
            raise ValueError(f"{sample} has not been scaled by its xsec and lumi!")

    events["finalWeight"] = events["weight"] / totals["np_nominal"]

    if not variations:
        return

    if weight_shifts is None:
        raise ValueError(
            "Variations requested but no weight shifts given! Please use ``variations=False`` or provide the systematics to be normalized."
        )

    # normalize all the variations
    for wvar in weight_shifts:
        if f"weight_{wvar}Up" not in events:
            continue

        for shift in ["Up", "Down"]:
            wlabel = wvar + shift
            if wvar in norm_preserving_weights:
                # normalize by their totals
                events[f"weight_{wlabel}"] /= totals[f"np_{wlabel}"]
            else:
                # normalize by the nominal
                events[f"weight_{wlabel}"] /= totals["np_nominal"]

    # normalize scale and PDF weights
    for wkey in ["scale_weights", "pdf_weights"]:
        if wkey in events:
            # .to_numpy() makes it way faster
            events[wkey] = events[wkey].to_numpy() / totals[f"np_{wkey}"]
