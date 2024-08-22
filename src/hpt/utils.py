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


def load_samples(
    data_dir: Path,
    samples: dict[str, str],
    year: str,
    filters: list = None,
    columns: list = None,
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

            # normalize by total events
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
