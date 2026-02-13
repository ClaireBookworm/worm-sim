from utils import *
from synapse import *
from scipy import signal, interpolate
from jaxley.connect import connect
import jaxley.optimize.transforms as jt
import numpy as np

def normalize(data, eps=1e-6):
    data = np.asarray(data, dtype=np.float64)
    mu = data.mean()
    sd = data.std()
    return (data - mu) / (sd + eps)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def Ca2V(ca_data):
    """
    ca_data could be shape (N_output, tstep)
    """
    n_output, tstep = ca_data.shape
    smooth_window1 = 20
    smooth_window2 = 8
    conv_window = 200
    dt = 0.25 # (s)
    ts = np.arange(0, conv_window*dt, dt)
    flt = np.exp(-ts/1) # (s/s)
    voltage_data = np.zeros((n_output, tstep-conv_window+1-smooth_window2), dtype=np.float64)
    for i in range(n_output):
        smooth_ca_data = smooth(normalize(ca_data[i,:]), smooth_window1)/2
        deconv_smooth_ca_data, _ = signal.deconvolve(smooth_ca_data, flt)
        smooth_deconv_smooth_ca_data = smooth(deconv_smooth_ca_data, smooth_window2)
        voltage_data[i,:] = smooth_deconv_smooth_ca_data[smooth_window2:]

    return voltage_data

def interpolate_data(src_time, src_data, dst_tstop, dst_dt):
    assert src_time[-1] >= dst_tstop and len(src_data)
    f = interpolate.interp1d(src_time[:src_data.shape[1]], src_data, kind="quadratic")
    dst_time = np.arange(0, dst_tstop+dst_dt, dst_dt)
    interpolated_data = f(dst_time)
    return interpolated_data

def map_data(data, map_range):
    ret = (data-np.min(data))/(np.max(data)-np.min(data))*(map_range[1] - map_range[0]) + map_range[0]
    return ret



def _conductance_key(chan_inst):
    keys = list(chan_inst.channel_params.keys())
    # Prefer explicit 'gb*' keys (your channels use these)
    for k in keys:
        if k.startswith("gb"):
            return k
    # Fallbacks if you ever add channels with different naming
    for k in keys:
        if k.lower() in ("gbar", "g"):  # common alternates
            return k
    for k in keys:
        if k.lower().startswith("g"):
            return k
    # Last resort: first key
    return keys[0]

def setup_biophysics(cell, param, channel_mechanism, soma_branch_index: int = 0):
    """
    • Adds a Leak channel to the entire cell and sets passive params separately
      for soma vs. neurite.
    • Inserts active channels only into the soma branch.
    """

    # 2) Passive parameters for every branch
    for idx, br in enumerate(cell.branches):
        if idx == soma_branch_index:
            pars   = param["soma"]
            seg = cell.branch(idx)
            seg.set("axial_resistivity", pars["Ra"])
            seg.set("capacitance",        pars["cm"])

        else:
            pars   = param["neurite"]
            seg = cell.branch(idx)
            seg.set("axial_resistivity", pars["Ra"])
            seg.set("capacitance",        pars["cm"])


    # 3) Active channels to only soma
    soma = cell.branch(soma_branch_index)
    for key, gbar in param["soma"].items():
        if not key.startswith("gb"):
            continue
        mech_name = key[2:]
        if mech_name not in channel_mechanism:
            print(f"[setup_biophysics] Unknown mechanism: {mech_name}")
            continue

        template = channel_mechanism[mech_name]  
        ChanCls  = template.__class__
        chan     = ChanCls(name=template._name)        

        g_key = _conductance_key(chan)                 
        chan.channel_params[g_key] = float(gbar) 
        soma.insert(chan)

def load_cb2022():

    cells_id_sim = [
            "24",
            "25",
            "36",
            "37",
            "38",
            "39",
            "48",
            "49",
            "60",
            "64",
            "71",
            "72",
            "87",
            "88",
            "89",
            "90",
            "111",
            "112",
            "113",
            "114",
            "117",
            "118",
            "119",
            "120",
            "130",
            "131",
            "138",
            "139",
            "141",
            "142",
            "143",
            "148",
            "149",
            "152",
            "153",
            "154",
            "155",
            "158",
            "169",
            "170",
            "171",
            "172",
            "175",
            "176",
            "177",
            "178",
            "179",
            "180",
            "181",
            "188",
            "189",
            "190",
            "191",
            "192",
            "193",
            "194",
            "195",
            "196",
            "197",
            "198",
            "199",
            "202",
            "203",
            "204",
            "205",
            "206",
            "207",
            "208",
            "209",
            "210",
            "211",
            "212",
            "217",
            "218",
            "219",
            "220",
            "221",
            "222",
            "223",
            "224",
            "225",
            "226",
            "227",
            "228",
            "229",
            "230",
            "231",
            "232",
            "233",
            "234",
            "235",
            "236",
            "237",
            "247",
            "250",
            "251",
            "252",
            "253",
            "254",
            "255",
            "256",
            "257",
            "258",
            "259",
            "260",
            "261",
            "262",
            "263",
            "264",
            "265",
            "266",
            "267",
            "268",
            "269",
            "270",
            "271",
            "272",
            "273",
            "274",
            "275",
            "276",
            "277",
            "278",
            "279",
            "280",
            "281",
            "282",
            "283",
            "284",
            "285",
            "286",
            "287",
            "288",
            "289",
            "290",
            "291"
        ]
    cell_name_dict = {
            "0": "I1L",
            "1": "I1R",
            "2": "I2L",
            "3": "I2R",
            "4": "I3",
            "5": "I4",
            "6": "I5",
            "7": "I6",
            "8": "M1",
            "9": "M2L",
            "10": "M2R",
            "11": "M3L",
            "12": "M3R",
            "13": "M4",
            "14": "M5",
            "15": "MCL",
            "16": "MCR",
            "17": "MI",
            "18": "NSML",
            "19": "NSMR",
            "20": "ASIL",
            "21": "ASIR",
            "22": "ASJL",
            "23": "ASJR",
            "24": "AWAL",
            "25": "AWAR",
            "26": "ASGL",
            "27": "ASGR",
            "28": "AWBL",
            "29": "AWBR",
            "30": "ASEL",
            "31": "ASER",
            "32": "ADFL",
            "33": "ADFR",
            "34": "AFDL",
            "35": "AFDR",
            "36": "AWCL",
            "37": "AWCR",
            "38": "ASKL",
            "39": "ASKR",
            "40": "ASHL",
            "41": "ASHR",
            "42": "ADLL",
            "43": "ADLR",
            "44": "BAGL",
            "45": "BAGR",
            "46": "URXL",
            "47": "URXR",
            "48": "ALNL",
            "49": "ALNR",
            "50": "PLNL",
            "51": "PLNR",
            "52": "SDQL",
            "53": "SDQR",
            "54": "AQR",
            "55": "PQR",
            "56": "ALML",
            "57": "ALMR",
            "58": "AVM",
            "59": "PVM",
            "60": "PLML",
            "61": "PLMR",
            "62": "FLPL",
            "63": "FLPR",
            "64": "DVA",
            "65": "PVDL",
            "66": "PVDR",
            "67": "ADEL",
            "68": "ADER",
            "69": "PDEL",
            "70": "PDER",
            "71": "PHAL",
            "72": "PHAR",
            "73": "PHBL",
            "74": "PHBR",
            "75": "PHCL",
            "76": "PHCR",
            "77": "IL2DL",
            "78": "IL2DR",
            "79": "IL2L",
            "80": "IL2R",
            "81": "IL2VL",
            "82": "IL2VR",
            "83": "CEPDL",
            "84": "CEPDR",
            "85": "CEPVL",
            "86": "CEPVR",
            "87": "URYDL",
            "88": "URYDR",
            "89": "URYVL",
            "90": "URYVR",
            "91": "OLLL",
            "92": "OLLR",
            "93": "OLQDL",
            "94": "OLQDR",
            "95": "OLQVL",
            "96": "OLQVR",
            "97": "IL1DL",
            "98": "IL1DR",
            "99": "IL1L",
            "100": "IL1R",
            "101": "IL1VL",
            "102": "IL1VR",
            "103": "AINL",
            "104": "AINR",
            "105": "AIML",
            "106": "AIMR",
            "107": "RIH",
            "108": "URBL",
            "109": "URBR",
            "110": "RIR",
            "111": "AIYL",
            "112": "AIYR",
            "113": "AIAL",
            "114": "AIAR",
            "115": "AUAL",
            "116": "AUAR",
            "117": "AIZL",
            "118": "AIZR",
            "119": "RIS",
            "120": "ALA",
            "121": "PVQL",
            "122": "PVQR",
            "123": "ADAL",
            "124": "ADAR",
            "125": "RIFL",
            "126": "RIFR",
            "127": "BDUL",
            "128": "BDUR",
            "129": "PVR",
            "130": "AVFL",
            "131": "AVFR",
            "132": "AVHL",
            "133": "AVHR",
            "134": "PVPL",
            "135": "PVPR",
            "136": "LUAL",
            "137": "LUAR",
            "138": "PVNL",
            "139": "PVNR",
            "140": "AVG",
            "141": "DVB",
            "142": "RIBL",
            "143": "RIBR",
            "144": "RIGL",
            "145": "RIGR",
            "146": "RMGL",
            "147": "RMGR",
            "148": "AIBL",
            "149": "AIBR",
            "150": "RICL",
            "151": "RICR",
            "152": "SAADL",
            "153": "SAADR",
            "154": "SAAVL",
            "155": "SAAVR",
            "156": "AVKL",
            "157": "AVKR",
            "158": "DVC",
            "159": "AVJL",
            "160": "AVJR",
            "161": "PVT",
            "162": "AVDL",
            "163": "AVDR",
            "164": "AVL",
            "165": "PVWL",
            "166": "PVWR",
            "167": "RIAL",
            "168": "RIAR",
            "169": "RIML",
            "170": "RIMR",
            "171": "AVEL",
            "172": "AVER",
            "173": "RMFL",
            "174": "RMFR",
            "175": "RID",
            "176": "AVBL",
            "177": "AVBR",
            "178": "AVAL",
            "179": "AVAR",
            "180": "PVCL",
            "181": "PVCR",
            "182": "RIPL",
            "183": "RIPR",
            "184": "URADL",
            "185": "URADR",
            "186": "URAVL",
            "187": "URAVR",
            "188": "RMEL",
            "189": "RMER",
            "190": "RMED",
            "191": "RMEV",
            "192": "RMDDL",
            "193": "RMDDR",
            "194": "RMDL",
            "195": "RMDR",
            "196": "RMDVL",
            "197": "RMDVR",
            "198": "RIVL",
            "199": "RIVR",
            "200": "RMHL",
            "201": "RMHR",
            "202": "SABD",
            "203": "SABVL",
            "204": "SABVR",
            "205": "SMDDL",
            "206": "SMDDR",
            "207": "SMDVL",
            "208": "SMDVR",
            "209": "SMBDL",
            "210": "SMBDR",
            "211": "SMBVL",
            "212": "SMBVR",
            "213": "SIBDL",
            "214": "SIBDR",
            "215": "SIBVL",
            "216": "SIBVR",
            "217": "SIADL",
            "218": "SIADR",
            "219": "SIAVL",
            "220": "SIAVR",
            "221": "DA01",
            "222": "DA02",
            "223": "DA03",
            "224": "DA04",
            "225": "DA05",
            "226": "DA06",
            "227": "DA07",
            "228": "DA08",
            "229": "DA09",
            "230": "PDA",
            "231": "DB01",
            "232": "DB02",
            "233": "DB03",
            "234": "DB04",
            "235": "DB05",
            "236": "DB06",
            "237": "DB07",
            "238": "AS01",
            "239": "AS02",
            "240": "AS03",
            "241": "AS04",
            "242": "AS05",
            "243": "AS06",
            "244": "AS07",
            "245": "AS08",
            "246": "AS09",
            "247": "AS10",
            "248": "AS11",
            "249": "PDB",
            "250": "DD01",
            "251": "DD02",
            "252": "DD03",
            "253": "DD04",
            "254": "DD05",
            "255": "DD06",
            "256": "VA01",
            "257": "VA02",
            "258": "VA03",
            "259": "VA04",
            "260": "VA05",
            "261": "VA06",
            "262": "VA07",
            "263": "VA08",
            "264": "VA09",
            "265": "VA10",
            "266": "VA11",
            "267": "VA12",
            "268": "VB01",
            "269": "VB02",
            "270": "VB03",
            "271": "VB04",
            "272": "VB05",
            "273": "VB06",
            "274": "VB07",
            "275": "VB08",
            "276": "VB09",
            "277": "VB10",
            "278": "VB11",
            "279": "VD01",
            "280": "VD02",
            "281": "VD03",
            "282": "VD04",
            "283": "VD05",
            "284": "VD06",
            "285": "VD07",
            "286": "VD08",
            "287": "VD09",
            "288": "VD10",
            "289": "VD11",
            "290": "VD12",
            "291": "VD13",
            "292": "CANL",
            "293": "CANR",
            "294": "HSNL",
            "295": "HSNR",
            "296": "VC01",
            "297": "VC02",
            "298": "VC03",
            "299": "VC04",
            "300": "VC05",
            "301": "VC06"
        }


    total_network_dict = {k: v for k, v in cell_name_dict.items() if k in cells_id_sim}

    return total_network_dict


def get_connection_site_always_soma(net, cell_idx, point_id, pre_id=None, post_id=None):
    """
    Resolve correct neurite/axon/dendrite segment for a given point_id.
    We use cell.nodes DataFrame to find branch and segment mapping.
    local_branch_index: branch ID (0 for soma, >=1 for neurites)
    local_comp_index: segment index within that branch
    """
    cell = net.cell(cell_idx)
    nodes_df = cell.nodes
    neurite_nodes = nodes_df
    neurite_nodes_sorted = neurite_nodes.sort_values([
        "local_branch_index", "local_comp_index"
    ]).reset_index(drop=True)

    target_row = neurite_nodes_sorted.iloc[0]
    branch_idx = int(target_row.local_branch_index)
    seg_idx = int(target_row.local_comp_index)

    return cell.branch(branch_idx)[seg_idx]

def _apply_edge_weight_to_synapse(syn, weight):
    prefix = syn._name
    w = float(weight)

    # Chemical: either set w or scale gS
    if isinstance(syn, (ElectricalSynapse, Excitatory_chemical_synapse, Inhibitory_chemical_synapse)):
        syn.synapse_params[f"{prefix}_w"] = w
    else:
        raise ValueError(f"Unknown synapse type.")
    
def build_connections_always_soma(net, connection_dict, cell_name_dict):
    id_to_idx = {cid: idx for idx, cid in enumerate(cell_name_dict.keys())}

    for post_id, conn_list in connection_dict.items():
        if post_id not in id_to_idx:
            continue
        post_idx = id_to_idx[post_id]

        for conn in conn_list:
            pre_id, id_pre_point, id_post_point, conn_type, weight = conn
            if pre_id not in id_to_idx:
                continue
            pre_idx = id_to_idx[pre_id]

            # Choose synapse type
            if conn_type == 0:
                syn = ElectricalSynapse()
            elif conn_type == 1:
                syn = Excitatory_chemical_synapse()
            elif conn_type == 2:
                syn = Inhibitory_chemical_synapse()
            else:
                raise ValueError(f"Unknown conn_type {conn_type}")

            _apply_edge_weight_to_synapse(syn, weight)

            # Find correct neurite/axon/dendrite sites for pre and post
            pre_site = get_connection_site_always_soma(net, pre_idx, id_pre_point, pre_id=pre_id, post_id=post_id)
            post_site = get_connection_site_always_soma(net, post_idx, id_post_point, pre_id=pre_id, post_id=post_id)

            # Connect pre -> post at resolved sites
            connect(pre_site, post_site, syn)



def create_transform(name):
    if name.endswith("_w"):
        return jt.SigmoidTransform(0, 10)
    elif name.endswith("_g_gap") or name == "g_gap" or name.endswith("synapse_gS"):
        return jt.SigmoidTransform(0.01e-4, 10.0e-4)
    else:
        assert False

