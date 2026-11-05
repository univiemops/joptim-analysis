import re
from copy import copy

# long

def map_labels_long(x: str) -> str:
    raw = copy(x)
    if x == "(Intercept)":
        return "Intercept"
    x = x.replace("spacing.1", "Threshold Spacing ")
    x = x.replace("central", "Central Threshold")
    # absolute differences
    x = x.replace("abs_diff_ac_RvL_z", "Heuristic-related complexity")
    x = re.sub(r"abs_diff_cc_[^: ]*_RvL[^: ]*_z", "Compositional complexity", x)
    x = x.replace("abs_diff_disorder_RvL_z", "Visual-order complexity")
    x = x.replace("abs_diff_diaga_RvL_z", "Diagonal dissimilarity")
    # signed differences
    x = x.replace("diff_ac_RvL_z", "Heuristic-related complexity")
    x = x.replace("diff_disorder_RvL_z", "Visual-order complexity")
    x = x.replace("diff_diaga_RvL_z", "Diagonal dissimilarity")
    x = re.sub(r"diff_cc_[^: ]*_RvL[^: ]*_z", "Compositional complexity", x)
    # moderators (no change for interactions, handled below)
    x = x.replace("load_capacity_ratio_z", "Difficulty")
    x = x.replace("psi_z", "PSI")
    x = x.replace("efficiency_z", "Problem-solving efficiency")
    x = x.replace("heuristic_optimality_z", "Heuristic optimality")
    x = x.replace("max_disorder_z", "Maximum disorder")
    
    x = x + " (" + map_labels(raw) + ")"
    return x

# short
def map_labels(x: str) -> str:
    if x == "(Intercept)":
        return "Intercept"
    x = x.replace("choice_cat_RvL", "Choice")
    x = x.replace("cbind(eye_R, eye_L)", "Gaze")
    x = x.replace("log_rt", "RT")
    x = x.replace("participant", "Subject")
    x = x.replace("spacing.1", "Threshold Spacing")
    x = x.replace("central", "Central Threshold")
    # absolute differences
    x = x.replace("abs_diff_ac_RvL_z", "|$\Delta$HC|")
    x = re.sub(r"abs_diff_cc_[^: ]*_RvL[^: ]*_z", r"|$\\Delta$CC|", x)
    x = x.replace("abs_diff_disorder_RvL_z", "|$\Delta$VC|")
    x = x.replace("abs_diff_diaga_RvL_z", "|$\Delta$DD|")
    # signed differences
    x = x.replace("diff_ac_RvL_z", "$\Delta$HC")
    x = x.replace("diff_disorder_RvL_z", "$\Delta$VC")
    x = x.replace("diff_diaga_RvL_z", "$\Delta$DD")
    x = re.sub(r"diff_cc_[^: ]*_RvL[^: ]*_z", r"$\\Delta$CC", x)
    # moderators (no change for interactions, handled below)
    x = x.replace("load_capacity_ratio_z", "Difficulty")
    x = x.replace("psi_z", "PSI")
    x = x.replace("efficiency_z", "PSE")
    x = x.replace("heuristic_optimality_z", "HO")
    x = x.replace("max_disorder_z", "MD")
    return x
    
# long
# RANDOM_EFFECTS_LABELS = [
#             "SD (Intercept)",
#             "SD (Heuristic-related complexity)",
#             "SD (Compositional complexity)",
#             "SD (Visual-order complexity)",
#             "SD (Diagonal dissimilarity)",
#         ]

# short
RANDOM_EFFECTS_LABELS = [
            "SD (Intercept | Subject)",
            "SD ($\Delta$HC | Subject)",
            "SD ($\Delta$CC | Subject)",
            "SD ($\Delta$VC | Subject)",
            "SD ($\Delta$DD | Subject)",
        ]