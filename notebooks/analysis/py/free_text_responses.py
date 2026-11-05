# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py//py:percent
#     notebook_metadata_filter: kernelspec,language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.0
# ---

# %% [markdown]
# # Strategies
#
# - Requires `<STUDY>/prolific_demographics.csv` for filtering to prolific participants

# %%
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from joptim.utils.export_path import get_export_paths
from joptim.demographics import get_prolific_participant_ids

EXPORT_PATH_DATA, _, _ = get_export_paths()
STUDY = "confirmatory"  # "exploratory" or "confirmatory"
ANALYSIS_NAME = "strategies"
OUT_DIR = Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

# requires to download data first in Download Experiment Data.ipynb
with open(Path(EXPORT_PATH_DATA, "results_no_exclusion.pkl"), "rb") as pickle_file:
    results = pickle.load(pickle_file)

with open(Path(EXPORT_PATH_DATA, "results.pkl"), "rb") as pickle_file:
    results_with_exclusion = pickle.load(pickle_file)

included_participant_ids = list(results_with_exclusion["user"]["id"])
del results_with_exclusion

# filter to completed users
bool_values_start_at = 3
results["user"] = results["user"][
    np.all(results["user"].iloc[:, bool_values_start_at:], axis=1)
]

df_user = results["user"]
df_user["included"] = df_user.id.isin(included_participant_ids)
df_user = df_user.merge(results["survey"][["id", "result_strategies_1"]], on="id")
df_user = df_user.merge(results["survey"][["id", "result_strategies_2"]], on="id")

# filter to users with accepted prolific participant IDs
prolific_participant_ids = get_prolific_participant_ids(EXPORT_PATH_DATA, STUDY)

df_user = df_user.loc[df_user["pid"].isin(prolific_participant_ids)]

del results

# %% [markdown]
# num of prolific participants

# %%
np.sum(df_user.pid.apply(lambda x: len(x)) == 24)

# %% [markdown]
# num of prolific participants that were included

# %%
np.sum(df_user.loc[df_user.pid.apply(lambda x: len(x)) == 24, "included"])

# %% [markdown]
# num of incompleted participants

# %%
np.sum(df_user.iloc[:, 2:] == False, axis=0)

# %% [markdown]
# ## Prompts for Summarizing with a Large Language Model

# %% [markdown]
# ### Problem-Solving Prompt

# %%
result_strategies_1 = [x["1"] for x in df_user["result_strategies_1"].values]

prompt_solving = f"""# Bin Packing Strategy Analysis Prompt

## Task Overview
You are analyzing qualitative data from an experiment where participants solved bin packing problems and then reported their strategies. Your task is to summarize and categorize the problem-solving strategies described by participants.

## Context: Bin Packing Problem Definition
The bin packing problem in this experiment is a variant where:
- Each item can be allocated to one of several bins
- Items have different sizes and bins have different capacities
- Goal: maximize the total amount packed into bins
- Constraints: 
  - Each bin can only hold items up to its capacity
  - Each item can only be assigned to one bin (no splitting)
  - Problem instances had 4-6 bins and 7-9 items
  - All problems had at least two optimal solutions

## Experimental Context
Participants solved 7 bin packing problems of increasing difficulty. After each trial, they received feedback showing whether their solution was optimal, and if not, they saw an optimal solution alongside their own. This was designed to help participants understand the problem better.

## Analysis Instructions
Please analyze the participant strategy reports and provide:

1. **Strategy Categories**: Identify and define distinct categories of problem-solving strategies mentioned by participants. For each category:
   - Provide a clear name and definition
   - List key characteristics or approaches
   - Note frequency of mention across participants

2. **Common Patterns**: Identify recurring themes, approaches, or decision-making processes across participants

3. **Strategy Evolution**: Note any mentions of how participants' strategies changed or developed across the 7 trials

4. **Specific Techniques**: Highlight concrete problem-solving techniques or heuristics mentioned (e.g., "fill largest bins first," "group similar-sized items," etc.)

5. **Metacognitive Aspects**: Identify any mentions of participants reflecting on their own problem-solving process, learning, or strategy adaptation

6. **Individual Differences**: Note any particularly unique or interesting individual approaches that stand out

## Output Format
Structure your analysis as follows:
- **Executive Summary** (2-3 sentences)
- **Main Strategy Categories** (with definitions and frequencies)
- **Key Findings** (patterns, techniques, evolution)
- **Notable Individual Approaches** (unique strategies)
- **Implications** (what this reveals about human problem-solving in bin packing)

## Participant Strategy Data

{result_strategies_1}

The data consists of strategy reports from participants, where each entry represents one participant's description of their problem-solving approach after completing all 7 bin packing trials."""

print(prompt_solving)

# %% [markdown]
# #### Results for exploratory study 14 July 2025 GPT-4o
#
# **Executive Summary**  
# Participants employed a variety of strategies to solve bin packing problems, with approaches broadly categorized into mathematical calculations, priority-based filling, trial and error, and intuitive matching. These strategies evolved as participants received feedback, demonstrating adaptation and refinement over time.
#
# **Main Strategy Categories**
#
# 1. *Mathematical Calculations*:
#    - Defined by using arithmetic and calculations to determine the best fit for items in bins.
#    - Key characteristics include precise computations, prioritizing sums that exactly match bin capacities.
#    - Mentioned by approximately 30% of participants.
#
# 2. *Priority-Based Filling*:
#    - Involving placing the largest items first or filling bins with the highest capacity.
#    - Approaches center around maximizing initial space usage with big items and refining placement for smaller items subsequently.
#    - Mentioned by roughly 40% of participants.
#
# 3. *Trial and Error*:
#    - Participants described experimenting with different combinations until they found one that worked.
#    - Characterized by iterative testing and adjustment rather than a fixed strategy.
#    - Mentioned by about 25% of participants.
#
# 4. *Intuitive Matching*:
#    - Strategy based on instinctively matching items to bin sizes with minimal calculations.
#    - Participants used gut feeling or visual estimation to make decisions.
#    - Mentioned by around 15% of participants.
#
# **Key Findings**
#
# - *Common Patterns*: Participants frequently began by targeting either the largest items or bins and then worked on optimization through combination or computational checks.
# - *Strategy Evolution*: Many participants noted improvement over trials, refining strategies based on feedback, particularly moving from trial and error to priority-based calculations.
# - *Specific Techniques*: Techniques such as "filling largest bins first," "pairing items to perfectly match bin capacity," and "combining smaller items together" were frequently cited.
# - *Metacognitive Aspects*: Numerous participants reflected on their learning process, particularly how feedback informed strategy adjustments over time.
#
# **Notable Individual Approaches**
#
# - One participant mentioned using "a strategy from dynamic programming," indicating a more structured and algorithmic approach.
# - Several participants reported not initially having a defined strategy but gradually developing their methods based on experience and reflection.
#
# **Implications**
#
# The variety of strategies employed underscores the complexity of human problem-solving processes in bin packing scenarios, highlighting differential application of mathematical reasoning, priority-based methods, and intuitive decision-making. This indicates an adaptability and learning mechanism driven by feedback, which can be capitalized on in educational or cognitive research settings.
#

# %% [markdown]
# #### Results for confirmatory study 3 December 2025 GPT-5

# %% [markdown]
# **Executive Summary**  
# Participants overwhelmingly used simple, greedy heuristics: matching item sums to bin capacities and placing largest items in the largest bins first. Many reported adapting across trials, adding constraint-first reasoning (place items with limited bin options early) and iterative refinement.
#
# **Main Strategy Categories**
# - Exact-fit/Capacity Matching
#   - Definition: Aim to fill each bin exactly or as close as possible to capacity to minimize leftover.
#   - Characteristics: Search for pairs/sets that sum to bin capacity (e.g., 55+45=100); “leave as little left over as possible”; fill each box to capacity before moving on.
#   - Frequency: Very common (~45–55%).
# - Largest-first Greedy
#   - Definition: Prioritize placing largest items into the largest-capacity bins first.
#   - Characteristics: “Fill the biggest ones first then go down”; “highest boxes first”; use small items to fill remaining gaps.
#   - Frequency: Common (~30–40%).
# - Constraint-first (Limited Placement)
#   - Definition: Place items that can only fit in specific bins (or have few options) before others.
#   - Characteristics: “Numbers that couldn’t go anywhere else first”; “high value with limited boxes”; “lock” unique placements early.
#   - Frequency: Moderate (~10–15%).
# - Quantity/Score Maximization (non-specific greedy)
#   - Definition: Maximize total packed or number of boxes filled without a precise capacity-matching plan.
#   - Characteristics: “Fit as many as possible”; “maximize usage/score”; accept non-full bins if more items fit overall.
#   - Frequency: Moderate (~20–25%).
# - Iterative Refinement/Verification
#   - Definition: Make an initial allocation, then re-check sums, reorganize, and adjust if suboptimal.
#   - Characteristics: “First cut then review”; check spare capacity vs unallocated items; reorganize if it looks suboptimal.
#   - Frequency: Some (~8–12%).
# - Trial-and-error/Intuition/No Clear Strategy
#   - Definition: Rely on instinct, random tries, or acknowledge no explicit strategy.
#   - Characteristics: “Instinct,” “no strategy,” “clicked randomly then realized sizes.”
#   - Frequency: Some (~15–20%).
# - Pattern/Spatial Anchoring
#   - Definition: Use visual patterns or fixed traversal orders.
#   - Characteristics: Diagonal fill, top-to-bottom/left-to-right, far-right placement, column alignment.
#   - Frequency: Low (~5–8%).
# - Math-driven Calculation
#   - Definition: Explicit arithmetic and cross-checking to construct sums and verify feasibility.
#   - Characteristics: Compute totals, cross-check sums, look for exact matches and factors (e.g., 45+55), elimination.
#   - Frequency: Some (~8–12%).
# - Feedback-driven Adaptation/Learning
#   - Definition: Strategy changes based on observing optimal solutions and prior performance.
#   - Characteristics: “Learning from previous round/other score,” refining from “obvious” to more nuanced combinations.
#   - Frequency: Low–moderate (~5–10%).
#
# **Key Findings**  
# - Common patterns:
#   - Greedy local optimization dominates: exact-fit matching and largest-first are the default approaches.
#   - Constraint awareness emerges: several prioritize items with limited bin options to avoid dead ends.
#   - Gap-filling with smaller items: widely used to top off bins after placing large items.
# - Specific techniques/heuristics:
#   - Fill largest bins/items first; then use small items to fill gaps.
#   - Seek exact sums per bin (e.g., 55+45=100); start with perfect matches, then near-matches.
#   - Place “forced” items (that fit only one bin) early.
#   - Left-to-right/top-to-bottom traversal; diagonal fills; align items to columns.
#   - Reorganize if the layout looks suboptimal; verify feasibility by comparing spare capacity to unallocated items.
#   - Use elimination and simple arithmetic to check combinations; group similar sizes.
# - Strategy evolution:
#   - Several moved from random/intuitive to structured matching and constraint-first after seeing feedback.
#   - Early trials focused on obvious matches and largest-first; later trials added combination search and re-checking.
#   - Some explicitly learned to consider bin capacity differences and refine via trial-and-error.
#
# **Notable Individual Approaches**  
# - “Lock” unique placements early and then optimize sums; desired a “lock button” to prevent moving forced items.
# - Diagonal filling pattern initially, abandoned as complexity increased.
# - Formal feasibility check: compare spare capacity across bins to total unallocated items before revising.
# - Willingness to leave a high-value item out to enable two better-fitting combinations (prioritizing total packed over single-bin perfection).
# - Column/row optimization and far-right placement strategies suggest visual/spatial anchoring not directly tied to capacity logic.
#
# **Implications**  
# - Humans default to simple, local heuristics (exact-fits, largest-first), which are effective but can miss global optima without constraint-first reasoning.
# - Exposure to feedback encourages strategy refinement: participants incorporate forced placements and iterative checks over time.
# - Cognitive load leads to reliance on visual patterns and intuition; arithmetic scaffolding (sum hints, forced-item indicators) could improve performance.
# - Multiple optimal solutions invite flexible selection criteria; some participants consider alternative scoring rules or trade-offs, highlighting metacognitive awareness and adaptive problem solving.
#

# %% [markdown]
# ### Evaluation Prompt

# %%
result_strategies_2 = [x["1"] for x in df_user["result_strategies_2"].values]

prompt_evaluation = f"""

# Prompt for Analyzing Bin Packing Solution Evaluation Strategies

## Task Overview
You are tasked with analyzing and summarizing qualitative text data from participants who completed an experiment on bin packing solution evaluation. Each participant provided a text response describing their strategies after completing 25 evaluation trials where they compared pairs of optimal bin packing solutions.

## Experimental Context

### Bin Packing Problem Definition
The experiment focused on a variant of the bin packing problem where:
- Each item can be allocated to one of several bins
- Rows represent items, columns represent bins
- Blocks and numbers indicate item sizes
- Dots in assignment matrix show item-to-bin assignments
- Goal: maximize total amount packed into bins (score shown in top right)
- This is a special case of the multi-knapsack problem where profit equals item size

### Problem Constraints
Each problem instance had:
- 4-6 bins and 7-9 items
- No item larger than the largest bin
- No bin smaller than the smallest item
- Ratio of sum of item sizes to sum of bin capacities between 0.8-1.0
- At least two different optimal solutions

### Evaluation Trials Structure
Participants completed 25 evaluation trials where they:
- Viewed pairs of optimal solutions to the same problem
- Answered "Which of the two solutions do you find easier to understand?"
- Chose from: "definitely left," "slightly left," "slightly right," "definitely right"
- Completed 2 catch trials (identical solutions, required "Duplicated solutions" response)
- Completed 3 coherence trials (linked solution pairs to assess judgment consistency)
- Had no time limit
- Evaluated 20 randomly sampled pairs plus the standard catch/coherence trials

## Analysis Instructions

Please analyze the participant strategy descriptions and provide:

1. **Common Strategy Themes**: Identify and categorize the main strategies participants used to evaluate solution interpretability/understandability

2. **Strategy Frequency**: Note how frequently different strategies appear across participants

3. **Strategy Sophistication**: Assess the complexity and depth of different strategic approaches

4. **Individual Differences**: Highlight notable variations in how participants approached the task

5. **Criteria for "Easier to Understand"**: Summarize what factors participants considered when determining which solution was more interpretable

6. **Cognitive Processes**: Identify any mentions of specific cognitive processes, heuristics, or decision-making patterns

## Output Format
Structure your analysis as follows:
- **Executive Summary**: Brief overview of main findings
- **Primary Strategy Categories**: Major themes with examples
- **Secondary Patterns**: Less common but notable approaches
- **Individual Variations**: Unique or outlier strategies
- **Implications**: What these strategies reveal about human evaluation of solution interpretability

## Participant Strategy Data
Below are the participant responses as a Python list, with each entry representing one participant's strategy description:

{result_strategies_2}"""

print(prompt_evaluation)

# %% [markdown]
# #### Results for exploratory study 14 July 2025 GPT-4o

# %% [markdown]
# **Executive Summary**
#
# Participants employed a variety of strategies to determine which bin packing solutions were easier to understand. The main themes included aligning solutions with personal strategies, prioritizing simplicity and patterns, focusing on numerical order and visualization, and using heuristic approaches. Strategy frequencies varied, with some strategies being notably more common and simpler than others. Individual differences were evident, with unique approaches highlighting diverse cognitive processes and interpretations.
#
# **Primary Strategy Categories**
#
# 1. *Alignment with Personal Strategy*: Many participants evaluated solutions based on their alignment with strategies they would personally employ, such as going from left to right or filling the highest-capacity bins first.
#    - Example: "I picked the solution which aligned most with how I would solve the problem."
#    - Frequency: High; mentioned in various forms by numerous participants.
#
# 2. *Simplicity and Visualization*: Solutions perceived as visually simpler and easier to interpret were favored. Participants mentioned preference for reduced complexity, fewer boxes, and clearer patterns.
#    - Example: "I found the less columns that were filled the easier on the eye."
#    - Frequency: Moderate; several participants noted this theme.
#
# 3. *Numerical Order and Patterns*: Some participants looked for sequential or logical numerical patterns, making solutions easier to understand when items and bins followed ascending or descending order.
#    - Example: "Looking at the order of the numbers in both axes as the ones in ascending or descending order were easier to work out."
#    - Frequency: Moderate; observed in several responses.
#
# **Secondary Patterns**
#
# 1. *Judgment by Intuition or Gut Feeling*: A subset relied on instinctual judgments rather than explicit strategies.
#    - Example: "gut feeling"
#    - Frequency: Lower; mentioned by a few participants.
#
# 2. *Specific Heuristics*: Participants referenced heuristics like the greedy strategy, filling the largest bins or using local optimal tactics.
#    - Example: "greedy strategy (a computer science theory), i.e., achieving a local optimal at each column."
#    - Frequency: Lower; few explicit mentions.
#
# **Individual Variations*
#
# - *Complexity of Approach*: Some participants displayed detailed strategies considering numerical sequences and visual order, showing greater strategic sophistication.
#    - Example: "Glancing at ALL the information on both I weighed up the importance of each factor. Also the order and neatness of the filled boxes."
#   
# - *Lack of Strategy*: Some noted having no clear strategy, highlighting confusion or reliance on randomness.
#    - Example: "I had no strategies at all because I'm not sure I even still understand the question."
#
# **Criteria for "Easier to Understand"**
#
# Participants often considered factors such as simplicity, alignment with personal tactics, reduced clutter, clear numerical or visual order, and visible patterns when identifying more interpretable solutions.
#
# **Cognitive Processes**
#
# - Decision-making patterns included heuristic approaches, personal alignment, and intuitive judgments, reflecting diverse cognitive processes and heuristics in evaluating solution interpretability.
#
# **Implications**
#
# These strategies reveal that interpretation of bin packing solutions involves personal biases, simplicity preferences, and heuristic reasoning. Understanding these cognitive processes can aid in designing more intuitive and accessible problem-solving interfaces or training methods that resonate with varied user preferences and cognitive styles.
#

# %% [markdown]
# #### Results for confirmatory study 3 December 2025 GPT-5

# %% [markdown]
# **Executive Summary**  
# Participants judged interpretability primarily by visual order and cognitive ease. Most preferred solutions that looked orderly (left-to-right, descending sizes), used “greedy” largest-first allocations, produced exact or fuller bins, and minimized mixing (fewer dots per column, fewer additions). Many anchored judgments to how they themselves would solve the problem; a minority relied on gut/visual appeal alone. Some applied systematic checks (e.g., duplicate detection, column-by-column scans). Overall, participants used fast, low-load visual/arithmetical heuristics and familiar mental models.
#
# **Primary Strategy Categories**  
# - Visual order and spatial regularity (≈23/108, ~21%)
#   - Left-to-right filling; largest bins/items on the left; neat/“descending steps,” symmetry, diagonal or uniform patterns.
# - Largest-first (“greedy”) heuristic (≈16/108, ~15%)
#   - Fill largest boxes first with biggest items; “make the most of big numbers.”
# - Minimize cognitive load via simplicity (≈11/108, ~10%)
#   - Choose the option understood fastest; more “straightforward,” “obvious,” or “organized.”
# - Fewer combinations per bin; easy sums (≈12/108, ~11%)
#   - Prefer single-item fills or few components; avoid many dots/colors per column; favor easy/round sums (e.g., 90 vs 95+).
# - Fullness/completeness cues (combined ≈17/108, ~16%)
#   - Exact column totals (exactly filled bars) and more full bins/fewer gaps; fewer leftover items.
# - Match to own strategy/mental simulation (≈15/108, ~14%)
#   - Pick the solution closest to how they would have solved it; imagine their own approach and compare.
# - Duplicate detection and comparison procedure (≈11/108, ~10%)
#   - Check if solutions are identical/re-ordered; use top bars/graphs or color patterns to detect duplicates.
#
# **Secondary Patterns** 
# - Visual scanning tactics (≈5/108): column-by-column scanning, follow the dots, check rows/columns and top bars first.
# - Gut instinct/personal preference (≈7/108) and “no strategy/none” (≈3/108); “both equal” (≈2/108).
# - Preferences for round numbers and chunking (≈2/108).
# - Layout specifics: top-left bias (dots near top-left felt clearer); dislike of empties in the middle; preference for spreading items across columns vs “stacking” multiple items in one bin (conflicting minority preferences).
# - Efficiency cues: fewer boxes used (rare) or faster to parse.
#
# **Individual Variations**  
# - Directionality differences: many preferred left-to-right; one explicitly worked rightmost-first.
# - Representation sensitivity: a few flagged isomorphic duplicates (same assignments in different order), one noted confusion about whether re-ordered graphs counted as duplicates.
# - Divergent fullness preferences: some prioritized exact fills; others prioritized fewer items per bin or broader distribution across columns.
# - Process structure: a few used a stepwise routine (check duplicates first, then pick the more “obvious”/orderly one); others anchored on the left panel and looked for disliked differences on the right.
# - Interpretability vs optimality conflation: a small number spoke about “better/optimized” solutions despite both being optimal.
#
# **Criteria for “Easier to Understand”**  
# - Predictable spatial order: bins sorted by capacity (left-to-right), allocations that form neat, descending or diagonal visual patterns.
# - Largest-first logic: intuitive greedy progression from big bins/items to smaller ones.
# - Reduced arithmetic/visual complexity: fewer dots per column, fewer mixed colors, round/“easy” sums, single-item fills.
# - Completion signals: exactly filled bars, more full bins, fewer leftover items.
# - Consistency with one’s own solving strategy: alignments with personal approach felt more interpretable.
# - Quick visual parsing: what could be grasped at a glance; clearer typography/figures; top bars aiding comparison.
# - Absence of irregularities: avoid empties in the middle; prefer allocations clustered to the left/top-left.
#
# **Cognitive Processes**  
# - Heuristics: greedy (largest-first), left-to-right/scanline bias, top-left anchoring, symmetry/gestalt preference, round-number/easy-sum bias, chunking (single-item columns).
# - Decision patterns: anchor-and-contrast (evaluate left, then find differences on right), elimination of duplicates before comparing, speed-accuracy emphasis (“fastest to understand”).
# - Mental simulation: comparing panels to one’s own method; using prior-trial strategy as a template.
# - Visual search: column-by-column/dot tracking; attention to top summary bars.
# - Load minimization: avoidance of multi-part sums and mixed colors to reduce working memory/arithmetical effort.
#
# **Implications**  
# - Interpretability is driven by visual regularity and cognitive economy: people prefer solutions that align with a simple greedy narrative and are easy to parse at a glance.
# - Sorting and layout matter: presenting bins/items in descending order and emphasizing left-to-right filling improves perceived clarity.
# - Visualization that highlights exact fills and reduces within-column mixing supports understanding by lowering arithmetic/WM demands.
# - Users map displays to their own mental strategies; interfaces that reflect common heuristics (largest-first, left-to-right) are perceived as clearer.
# - Support for equivalence recognition (e.g., indicating isomorphic duplicates) may reduce confusion when two optimal solutions differ only by ordering.
#
#

# %% [markdown]
# ## Save for further usage

# %%
df_user.to_csv(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, "strategies.csv"), index=False)
