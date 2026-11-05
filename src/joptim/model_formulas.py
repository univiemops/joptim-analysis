from itertools import combinations, product

def extend_pc(pc, predictor_groups):
    pc_extended = []
    for pc_unit in pc:
        if ':' in pc_unit:
            pc_unit_split = pc_unit.split(":")
            interaction_extended = []
            for pc_unit_split_unit in pc_unit_split:
                if 'pred_gr_' in pc_unit_split_unit:
                    group_index = int(pc_unit_split_unit.replace("pred_gr_", ""))
                    pred_extended = predictor_groups[group_index]
                    interaction_extended.append(pred_extended)
                else:
                    interaction_extended.append([pc_unit_split_unit])
            result = [f"{a}:{b}" for a, b in product(interaction_extended[0], interaction_extended[1])]
            pc_extended.extend(result)
        
        else:
            group_index = int(pc_unit.replace("pred_gr_", ""))
            result = predictor_groups[group_index]
            pc_extended.extend(result)
    return pc_extended


def create_model_formulas(outcome, predictor_groups, group, moderators=[], no_interactions_with=[], no_random_effects_for=[], no_random_effects_for_interaction_terms=True, further_exclusions=[]):
    # Start with a random intercept/thresholds only model
    model_formulas = [f"{outcome} ~ 1 + (1 | {group})"]

    # translate to predictor groups
    fixed_effects = [f"pred_gr_{i}" for i in range(len(predictor_groups))]
    
    # create interactions terms

    interaction_terms = []

    for m in moderators:
        for f in fixed_effects:
            if m == f:
                continue
            interaction_terms.append(f"{f}:{m}")

    fixed_effects.extend(interaction_terms)

    predictor_combinations = []
    
    for n_preds in range(1, len(fixed_effects) + 1):
        result = list(combinations(fixed_effects, n_preds))
        predictor_combinations_per_n = [list(combo) for combo in result]
        predictor_combinations.extend(predictor_combinations_per_n)

   
    # translate back to individual predictors

    predictor_combinations_extended = []
    
    for pc in predictor_combinations:
        if any(item.startswith('pred_gr_') for item in pc):
            pc_extended = extend_pc(pc, predictor_groups)
            predictor_combinations_extended.append(pc_extended)
        
        else:
            predictor_combinations_extended.append(pc)

    predictor_combinations = predictor_combinations_extended
   
    # Remove specific interaction terms when one component is in "no_interactions_with" list
   
    predictor_combinations_filtered = []
    
    for pc in predictor_combinations:
        pc_checked = []
        for p in pc:
            if ":" in p:
                if len(set(p.split(":")).intersection(no_interactions_with)) > 0:
                    ...
                else:
                    pc_checked.append(p)
            else:
                pc_checked.append(p)
        predictor_combinations_filtered.append(pc_checked)
    predictor_combinations = predictor_combinations_filtered
    del predictor_combinations_filtered
    
    # Remove the interaction terms when the its first component is not present as a fixed effect

    predictor_combinations_filtered = []

    for pc in predictor_combinations:
        pc_checked = []
        for p in pc:
            if ":" in p:
                if p.split(":")[0] not in pc:
                    ...
                else:
                    pc_checked.append(p)
            else:
                pc_checked.append(p)
        predictor_combinations_filtered.append(pc_checked)

    predictor_combinations = predictor_combinations_filtered
    del predictor_combinations_filtered

    # Remove the empty combinations (is already added in the beginning)

    predictor_combinations_filtered = []

    for pc in predictor_combinations:
        if len(pc) > 0:
            predictor_combinations_filtered.append(pc)

    predictor_combinations = predictor_combinations_filtered
    del predictor_combinations_filtered

    # explicit removals 
    
    predictor_combinations_filtered = []
    
    for pc in predictor_combinations:
        pc_new = [x for x in pc if x not in further_exclusions]
        predictor_combinations_filtered.append(pc_new)

    predictor_combinations = predictor_combinations_filtered
    del predictor_combinations_filtered
    
    # create formulas

    for pc in predictor_combinations:
        formula = f"{outcome} ~ {' + '.join(pc)}"
        model_formulas.append(formula + f" + (1 | {group})")
    
    # create formulas incl. random effects

    predictor_combinations_random_effects = []
    for pc in predictor_combinations:
        pc_random_effects = [x for x in pc if x not in no_random_effects_for and (":" not in x or not no_random_effects_for_interaction_terms)]
        predictor_combinations_random_effects.append(pc_random_effects)
        
    for i, _ in enumerate(predictor_combinations):
        formula = f"{outcome} ~ {' + '.join(predictor_combinations[i])}"
        model_formulas.append(formula + f" + ({' + '.join(['1'] + predictor_combinations_random_effects[i])} | {group})")

    
    # remove duplicates (use this method instead of set() to maintain the order
    model_formulas_duplicates_removed = []
    for m in model_formulas:
        if m not in model_formulas_duplicates_removed:
            model_formulas_duplicates_removed.append(m)
    
    # model_formulas = list(set(model_formulas))
    
    return model_formulas_duplicates_removed