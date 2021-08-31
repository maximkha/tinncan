import numpy as np
from typing import List
from numpy.core.numeric import indices
from sklearn.neural_network import MLPRegressor
from enum import Enum

def mlpreg_tibasic(reg: MLPRegressor, mode:str="formula", restrict_lists: int=1) -> str:
    result = ""
    
    assert reg.activation == "relu"

    if mode == "normal":
        for neurons_weight, neurons_intercept in zip(reg.coefs_, reg.intercepts_):
            print("==========")
            print("Layer")

            print(f"{neurons_weight.shape=}")
            print(f"{neurons_intercept.shape=}")

            print(f"{neurons_weight.shape[-1]} Neurons")

            # +0 - input
            # +1 - intermediate
            # +2 - neuron weight
            # +3 - bias

            #intermediate output
            result += f"{neurons_weight.shape[-1]}->dim(L{restrict_lists+1}\n"
            result += f"Fill(0,L{restrict_lists+1}\n"

            result += "{"
            result += ",".join(list(map(lambda x: str(round(x, 3)), neurons_intercept)))
            result += f"->L{restrict_lists+3}\n"
            for i in range(neurons_weight.shape[-1]):
                # dump neuron weights into list

                result += "{"
                result += ",".join(list(map(lambda x: str(round(x, 3)), neurons_weight[:, i])))
                result += f"->L{restrict_lists+2}\n"

                # iterate list
                result += f"0->A\n"
                result += f"For(B,1,{len(neurons_weight[:, i])}\n"
                result += f"A+(L{restrict_lists+2}(B)*L{restrict_lists+0}(B))->A\n"
                result += "End\n"
                result += f"max(0,A+L{restrict_lists+3}({i+1})->L{restrict_lists+1}({i+1})\n"
            result += f"L{restrict_lists+1}->L{restrict_lists+0}\n"

            print(f"{neurons_intercept=}")
    elif mode == "formula":
        return nn_formula_simplify(reg.coefs_, reg.intercepts_, restrict_lists=1)
    else:
        raise ValueError(f"Invalid mode {mode}")

    return result

def to_val(tstr: str):
    try:
        return float(tstr)
    except ValueError:
        return None

def needcast(fl: float, compat=True) -> str:
    fl_str = str(fl)
    if fl_str[:2] == "0.":
        fl_str = fl_str[1:]
    elif fl_str[:3] == "-0.":
        fl_str = f"-{fl_str[2:]}"
    
    if compat and fl < 0:
        return f"(0{fl_str})"

    return fl_str

def nn_formula_write_archived(coef: np.ndarray, intercept: np.ndarray, lists: List[int]=[1,2,3], splitat=0) -> str:
    neurons_weight, neurons_intercept = coef, intercept
    lines = []
    # lines.append(f"ClrAllLists")
    lines.append(f"UnArchive L{lists[1]}")

    line = "{"
    for i in range(neurons_weight.shape[-1]):
        for j in range(len(neurons_weight[:, i])):
            weight_val = round(neurons_weight[j, i], 3)
            if weight_val == 0:
                continue
            
            # check if it's in bank 1
            if j <= splitat:
                line += f"L{lists[1]}({j+1})*{needcast(weight_val)}+"
        if line[-1]=="+":
            line = line[:-1]
        bias_val = round(neurons_intercept[i], 3)
        if bias_val != 0:
            if line[-1] in [",", "{"]:
                if bias_val < 0:
                    line += "0,"
            else:
                line += f"{ '+' + str(bias_val) if bias_val > 0 else str(bias_val)},"
        elif line[-1] in [",", "{"]:
            line += "0,"

    if line[-1]==",":
        line = line[:-1]

    lines.append(f"{line}->L{lists[0]}")
    lines.append(f"ClrList L{lists[1]}")
    lines.append(f"UnArchive L{lists[2]}")

    for i in range(neurons_weight.shape[-1]):
        line = ""
        for j in range(len(neurons_weight[:, i])):
            weight_val = round(neurons_weight[j, i], 3)
            if weight_val == 0:
                continue
            
            # check if it's in bank 2
            if j > splitat:
                line += f"L{lists[2]}({j-splitat})*{needcast(weight_val)}+"
        if line == "": continue
        if line[-1]=="+":
            line = line[:-1]
        line += f"+L{lists[0]}({i+1})->L{lists[0]}({i+1})"
        lines.append(line)
    lines.append(f"max(0,L{lists[0]}->L{lists[0]}")
    return '\n'.join(lines)

def nn_formula_simplify(coefs: List[np.ndarray], intercepts: List[np.ndarray], restrict_lists: int=1) -> str:
    lines = []
    
    c_incs = [1]*coefs[0].shape[0]
    trimmed_indices = list(np.cumsum(c_incs) - 1)
    for neurons_weight, neurons_intercept in zip(coefs, intercepts):
        line = "max(0,{"
        incs = []
        for i in range(neurons_weight.shape[-1]):
            for j in range(len(neurons_weight[:, i])):
                weight_val = round(neurons_weight[j, i], 3)
                if weight_val == 0:
                    continue
                
                if c_incs[j] == 0:
                    continue

                line += f"L{restrict_lists+0}({trimmed_indices[j]+1})*{needcast(weight_val)}+"
            if line[-1]=="+":
                line = line[:-1]
            bias_val = round(neurons_intercept[i], 3)
            if bias_val != 0:
                if line[-1] in [",", "{"]:
                    if bias_val < 0:
                        incs.append(0)
                else:
                    line += f"{ '+' + str(bias_val) if bias_val > 0 else str(bias_val)},"
                    incs.append(1)
            elif line[-1] in [",", "{"]:
                incs.append(0)
            else:
                incs.append(1)

            line = line.replace(",+",",")
        line = f"{line[:-1]}->L{restrict_lists+0}"
        trimmed_indices = np.cumsum(incs) - 1
        c_incs = incs
        lines.append(line)

    return '\n'.join(lines)