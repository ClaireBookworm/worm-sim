"""
Derive recorded neuron indices (obs_idx) from Ca_traces_cell_name.txt.

The Ca imaging dataset (from Uzel/Kato/Zimmer 2022) labels neurons that were
recorded. We map those names to jaxley_worm network indices for extracting
voltages for the muscle readout.

Uses the same cell ordering as neuron/jaxley_worm/network.py load_cb2022()
but does NOT import jaxley to avoid heavy dependencies / segfaults.

Options for muscle readout:
- "all": Use all 58 named neurons (recommended - let readout learn which matter)
- "motor": Use only the ~27 motor neurons from the Ca dataset (biologically constrained)
"""

# Motor neurons from the 58 Ca dataset (WormAtlas/MoW/Connectome Toolbox)
# Ventral cord: AS10, DA*, DB*, VA*, VB*, VD*; Head: RME*, SMD*, SAB*; PDA; RIVR; PVNL/R
MOTOR_NEURON_NAMES = frozenset([
    "AS10", "DA01", "DA07", "DA09", "DB01", "DB02", "DB07",
    "VA01", "VA11", "VA12", "VB01", "VB02", "VB11", "VD13",
    "RMEL", "RMED", "RMEV", "SMDVR", "SMDDL", "SMDDR",
    "SABVL", "SABVR", "SABD", "PDA", "RIVR", "PVNL", "PVNR",
])

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent

# Default paths to Ca_traces_cell_name.txt (neuron names from Ca imaging dataset)
# Try NEURON components cb2022_data first, then BAAIWorm
CA_CELL_NAMES_CANDIDATES = [
    PROJECT_ROOT.parent.parent.parent / "davy" / "NEURON" / "components" / "cb2022_data" / "Ca_traces_cell_name.txt",
    PROJECT_ROOT / "BAAIWorm" / "eworm_learn" / "components" / "cb2022_data" / "Ca_traces_cell_name.txt",
]


def _default_ca_cell_names_path() -> Optional[Path]:
    """First existing path from candidates."""
    for p in CA_CELL_NAMES_CANDIDATES:
        if p.exists():
            return p
    return CA_CELL_NAMES_CANDIDATES[-1]  # For clearer error message

# cells_id_sim and cell_name_dict from network.py load_cb2022() - kept in sync
CELLS_ID_SIM = [
    "24", "25", "36", "37", "38", "39", "48", "49", "60", "64", "71", "72",
    "87", "88", "89", "90", "111", "112", "113", "114", "117", "118", "119", "120",
    "130", "131", "138", "139", "141", "142", "143", "148", "149", "152", "153", "154", "155",
    "158", "169", "170", "171", "172", "175", "176", "177", "178", "179", "180", "181",
    "188", "189", "190", "191", "192", "193", "194", "195", "196", "197", "198", "199",
    "202", "203", "204", "205", "206", "207", "208", "209", "210", "211", "212",
    "217", "218", "219", "220", "221", "222", "223", "224", "225", "226", "227",
    "228", "229", "230", "231", "232", "233", "234", "235", "236", "237",
    "247", "250", "251", "252", "253", "254", "255", "256", "257", "258", "259",
    "260", "261", "262", "263", "264", "265", "266", "267", "268", "269", "270",
    "271", "272", "273", "274", "275", "276", "277", "278", "279", "280", "281",
    "282", "283", "284", "285", "286", "287", "288", "289", "290", "291",
]

# Subset of cell_name_dict for cells_id_sim (from network.py)
CELL_NAME_DICT = {
    "24": "AWAL", "25": "AWAR", "36": "AWCL", "37": "AWCR", "38": "ASKL", "39": "ASKR",
    "48": "ALNL", "49": "ALNR", "60": "PLML", "64": "DVA", "71": "PHAL", "72": "PHAR",
    "87": "URYDL", "88": "URYDR", "89": "URYVL", "90": "URYVR",
    "111": "AIYL", "112": "AIYR", "113": "AIAL", "114": "AIAR",
    "117": "AIZL", "118": "AIZR", "119": "RIS", "120": "ALA",
    "130": "AVFL", "131": "AVFR", "138": "PVNL", "139": "PVNR", "141": "DVB", "142": "RIBL",
    "143": "RIBR", "148": "AIBL", "149": "AIBR", "152": "SAADL", "153": "SAADR",
    "154": "SAAVL", "155": "SAAVR", "158": "DVC", "169": "RIML", "170": "RIMR",
    "171": "AVEL", "172": "AVER", "175": "RID", "176": "AVBL", "177": "AVBR",
    "178": "AVAL", "179": "AVAR", "180": "PVCL", "181": "PVCR",
    "188": "RMEL", "189": "RMER", "190": "RMED", "191": "RMEV",
    "192": "RMDDL", "193": "RMDDR", "194": "RMDL", "195": "RMDR",
    "196": "RMDVL", "197": "RMDVR", "198": "RIVL", "199": "RIVR",
    "202": "SABD", "203": "SABVL", "204": "SABVR", "205": "SMDDL", "206": "SMDDR",
    "207": "SMDVL", "208": "SMDVR", "209": "SMBDL", "210": "SMBDR",
    "211": "SMBVL", "212": "SMBVR", "217": "SIADL", "218": "SIADR",
    "219": "SIAVL", "220": "SIAVR", "221": "DA01", "222": "DA02", "223": "DA03",
    "224": "DA04", "225": "DA05", "226": "DA06", "227": "DA07", "228": "DA08",
    "229": "DA09", "230": "PDA", "231": "DB01", "232": "DB02", "233": "DB03",
    "234": "DB04", "235": "DB05", "236": "DB06", "237": "DB07",
    "247": "AS10", "250": "DD01", "251": "DD02", "252": "DD03", "253": "DD04",
    "254": "DD05", "255": "DD06", "256": "VA01", "257": "VA02", "258": "VA03",
    "259": "VA04", "260": "VA05", "261": "VA06", "262": "VA07", "263": "VA08",
    "264": "VA09", "265": "VA10", "266": "VA11", "267": "VA12",
    "268": "VB01", "269": "VB02", "270": "VB03", "271": "VB04", "272": "VB05",
    "273": "VB06", "274": "VB07", "275": "VB08", "276": "VB09", "277": "VB10",
    "278": "VB11", "279": "VD01", "280": "VD02", "281": "VD03", "282": "VD04",
    "283": "VD05", "284": "VD06", "285": "VD07", "286": "VD08", "287": "VD09",
    "288": "VD10", "289": "VD11", "290": "VD12", "291": "VD13",
}


def _get_remapped_network_dict() -> dict:
    """Index -> neuron name for jaxley_worm (136 cells)."""
    total = {k: CELL_NAME_DICT[k] for k in CELLS_ID_SIM if k in CELL_NAME_DICT}
    # Use consistent order: cells_id_sim
    ordered_names = [total[k] for k in CELLS_ID_SIM if k in total]
    return {i: name for i, name in enumerate(ordered_names)}


def load_recorded_neuron_indices(
    ca_cell_names_path: Optional[str] = None,
    subset: str = "all",
) -> Tuple[np.ndarray, list]:
    """
    Load obs_idx from Ca_traces_cell_name.txt.

    The file has one tab-separated row. Non-numeric entries are neuron names
    (those with Ca imaging data). We map them to jaxley_worm network indices.

    Args:
        ca_cell_names_path: Path to Ca_traces_cell_name.txt (default: BAAIWorm/...)
        subset: "all" = all 58 named neurons; "motor" = only motor neurons (~28)
                for muscle readout (WormAtlas/MoW motor classification)

    Returns:
        obs_idx: (n_recorded,) int64 array of network indices
        labels_sub: list of neuron names (same order as obs_idx)
    """
    path = Path(ca_cell_names_path) if ca_cell_names_path else _default_ca_cell_names_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Ca_traces_cell_name.txt not found at {path}. "
            "Provide --ca-cell-names or ensure BAAIWorm/eworm_learn/components/cb2022_data/ exists."
        )

    with open(path, "r") as f:
        content = f.read()
    labels = content.strip().split("\t")
    is_numeric = [str(x).strip().isdigit() for x in labels]
    labels_sub = [labels[i] for i in range(len(labels)) if not is_numeric[i]]

    if subset == "motor":
        labels_sub = [n for n in labels_sub if n in MOTOR_NEURON_NAMES]
        if not labels_sub:
            raise ValueError(
                f"No motor neurons found in Ca file. MOTOR_NEURON_NAMES={MOTOR_NEURON_NAMES}"
            )

    remapped_network_dict = _get_remapped_network_dict()
    name_to_netidx = {name: idx for idx, name in remapped_network_dict.items()}

    # Only include names that exist in the network
    valid_labels = [n for n in labels_sub if n in name_to_netidx]
    missing = [n for n in labels_sub if n not in name_to_netidx]
    if missing:
        print(f"[recorded_neurons] {len(missing)} names not in jaxley_worm network: {missing[:10]}...")

    obs_idx = np.array([name_to_netidx[n] for n in valid_labels], dtype=np.int64)
    return obs_idx, valid_labels
