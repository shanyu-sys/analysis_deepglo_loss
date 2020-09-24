import argparse
import pandas as pd
import os

EP = "Epoch"

# for xseq and yseq
TL = "Train Loss"
VL = "Validation Loss"

# for factors
FLF = "Factorization Loss F"
FLX = "Factorization Loss X"
TLX = "Temporal Loss X"
VLG = "Validation Loss (Global)"


def extract_num(line):
    return float(line.split(":")[-1])


parser = argparse.ArgumentParser()
parser.add_argument("-filename", help="log name", default="/home/shan/01_KPI4_default.log")
parser.add_argument("-output_path", help="output folder")
args = parser.parse_args()

filename = args.filename
output_path = args.output_path
if output_path is None:
    output_path = args.filename[:-4]
if not os.path.exists(output_path):
    os.mkdir(output_path)

with open(filename, "r") as f:

    iter_num = -1
    losses = []
    df_list = []

    for line in f:
        if "Training Factors. Iter#:" in line or "Initializing Factors" in line:
            iter_num += 1
            losses.append({
                FLF: [],
                FLX: [],
                TLX: [],
                VLG: []
            })
        elif FLF in line:
            losses[iter_num][FLF].append(extract_num(line))
        elif FLX in line:
            losses[iter_num][FLX].append(extract_num(line))
        elif TLX in line:
            losses[iter_num][TLX].append(extract_num(line))
        elif VLG in line:
            losses[iter_num][VLG].append(extract_num(line))
        elif "FINISHED" in line:
            break

    for i, loss in enumerate(losses):
        df = pd.DataFrame(loss)
        df.index.name = EP
        df.to_csv(os.path.join(output_path, f"factor_iter{i}.csv"))

with open(args.filename, "r") as f:

    iter_num = -1
    losses = []

    for line in f:
        if "Training Xseq Model. Iter#:" in line or "training Yseq" in line:
            iter_num += 1
            losses.append({
                TL: [],
                VL: [],
            })
        elif TL in line:
            losses[iter_num][TL].append(extract_num(line))
        elif VL in line and VLG not in line:
            losses[iter_num][VL].append(extract_num(line))
        elif "FINISHED" in line:
            break

    for i, loss in enumerate(losses):
        df = pd.DataFrame(loss)
        df.index.name = EP
        if i == len(losses) - 1:
            df.to_csv(os.path.join(output_path, f"yseq.csv"))
        else:
            df.to_csv(os.path.join(output_path, f"xseq_iter{i}.csv"))
