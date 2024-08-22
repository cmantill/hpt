"""
Common functions for processors.

Author(s): Raghav Kansal
"""

from __future__ import annotations

from __future__ import annotations

import contextlib
import pickle
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from os import listdir
from pathlib import Path

import hist
import numpy as np
import pandas as pd
import vector
from hist import Hist

import awkward as ak
import numpy as np
from coffea.analysis_tools import PackedSelection

from .common_vars import (
    LUMI,
    data_key,
)

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


def load_samples(
    data_dir: Path,
    samples: dict[str, str],
    year: str,
    filters: list = None,
    columns: list = None,
    variations: bool = True,
    weight_shifts: dict[str, Syst] = None,
    reorder_txbb: bool = True,  # temporary fix for sorting by given Txbb
    txbb: str = "bbFatJetPNetTXbbLegacy",
    # select_testing: bool = False,
    load_weight_noxsec: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Loads events with an optional filter.
    Divides MC samples by the total pre-skimming, to take the acceptance into account.

    Args:
        data_dir (str): path to data directory.
        samples (Dict[str, str]): dictionary of samples and selectors to load.
        year (str): year.
        filters (List): Optional filters when loading data.
        columns (List): Optional columns to load.
        variations (bool): Normalize variations as well (saves time to not do so). Defaults to True.
        weight_shifts (Dict[str, Syst]): dictionary of weight shifts to consider.

    Returns:
        Dict[str, pd.DataFrame]: ``events_dict`` dictionary of events dataframe for each sample.

    """
    data_dir = Path(data_dir) / year
    full_samples_list = listdir(data_dir)  # get all directories in data_dir
    events_dict = {}

    # label - key of sample in events_dict
    # selector - string used to select directories to load in for this sample
    for label, selector in samples.items():
        # important to check that samples have been normalized properly
        load_columns = columns
        if label != "data" and load_weight_noxsec:
            load_columns = columns + format_columns([("weight_noxsec", 1)])

        events_dict[label] = []  # list of directories we load in for this sample
        for sample in full_samples_list:
            # check if this directory passes our selector string
            if not check_selector(sample, selector):
                continue

            sample_path = data_dir / sample
            parquet_path, pickles_path = sample_path / "parquet", sample_path / "pickles"

            # no parquet directory?
            if not parquet_path.exists():
                warnings.warn(f"No parquet directory for {sample}!", stacklevel=1)
                continue

            print(f"Loading {sample}")
            events = pd.read_parquet(parquet_path, filters=filters, columns=load_columns)

            # no events?
            if not len(events):
                warnings.warn(f"No events for {sample}!", stacklevel=1)
                continue

            if reorder_txbb:
                _reorder_txbb(events, txbb)

            # normalize by total events
            pickles = get_pickles(pickles_path, year, sample)
            if "totals" in pickles:
                totals = pickles["totals"]
                _normalize_weights(
                    events,
                    year,
                    totals,
                    sample,
                    isData=label == data_key,
                    variations=variations,
                    weight_shifts=weight_shifts,
                )
            else:
                if label == data_key:
                    events["finalWeight"] = events["weight"]
                else:
                    n_events = get_nevents(pickles_path, year, sample)
                    events["weight_nonorm"] = events["weight"]
                    events["finalWeight"] = events["weight"] / n_events

            events_dict[label].append(events)
            print(f"Loaded {sample: <50}: {len(events)} entries")

        if len(events_dict[label]):
            events_dict[label] = pd.concat(events_dict[label])
        else:
            del events_dict[label]

    return events_dict




def format_columns(columns: list):
    """
    Reformat input of (`column name`, `num columns`) into (`column name`, `idx`) format for
    reading multiindex columns
    """
    ret_columns = []
    for key, num_columns in columns:
        for i in range(num_columns):
            ret_columns.append(f"('{key}', '{i}')")
    return ret_columns


def check_selector(sample: str, selector: str | list[str]):
    if not isinstance(selector, (list, tuple)):
        selector = [selector]

    for s in selector:
        if s.endswith("?"):
            if s[:-1] == sample:
                return True
        elif s.startswith("*"):
            if s[1:] in sample:
                return True
        else:
            if sample.startswith(s):
                return True

    return False



def get_nevents(pickles_path, year, sample_name):
    """Adds up nevents over all pickles in ``pickles_path`` directory"""
    try:
        out_pickles = listdir(pickles_path)
    except:
        return None

    file_name = out_pickles[0]
    with Path(f"{pickles_path}/{file_name}").open("rb") as file:
        try:
            out_dict = pickle.load(file)
        except EOFError:
            print(f"Problem opening {pickles_path}/{file_name}")
        nevents = out_dict[year][sample_name]["nevents"]  # index by year, then sample name

    for file_name in out_pickles[1:]:
        with Path(f"{pickles_path}/{file_name}").open("rb") as file:
            try:
                out_dict = pickle.load(file)
            except EOFError:
                print(f"Problem opening {pickles_path}/{file_name}")
            nevents += out_dict[year][sample_name]["nevents"]

    return nevents
