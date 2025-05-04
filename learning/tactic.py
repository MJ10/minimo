"""Tactic induction utilities.

This file defines a very lightweight tactic–induction routine that can be
plugged into the existing learning loops.  We purposely keep the
implementation extremely simple – we look at contiguous slices of past
solutions and promote the most frequently-occurring ones to tactics.  The
heavy-lifting of *executing* a tactic is already handled by
`learning.action.TacticAction`.  Here we only need to generate the list of
`Tactic` objects that should be made available to the prover.

We reuse the canonical `Step` and `Tactic` definitions from
`learning.action` in order to avoid subtle bugs caused by having multiple
class definitions with the same name.
"""

from typing import List

# Canonical definitions coming from the action module.
from action import Step, Tactic  # noqa: F401 – re-export for external users

from worker import StudentResult

def _tactic_signature(t: Tactic):
    """Return a hashable descriptor of a tactic's *structure* (arrows + arity).

    Two tactics that only differ in the variable names for their parameters are
    considered equivalent here – they induce the same behaviour during proof
    search.
    """
    return tuple((tuple(step.arrows), len(step.arguments)) for step in t.steps)


def _num_parameters(t: Tactic) -> int:
    """Counts the *distinct* formal parameters introduced inside the tactic."""
    # Any token that starts with '?' denotes a result of a previous step *inside*
    # the tactic, therefore it is not a parameter.  Any other token that is not
    # a literal constant (heuristically: does not start with '!step') is treated
    # as a parameter.
    params: set[str] = set()
    internal_results: set[str] = {step.result for step in t.steps if step.result}

    for step in t.steps:
        for arg in step.arguments:
            if arg in internal_results:
                continue
            if arg.startswith('!step'):
                # Reference to something produced *outside* the tactic – treat as
                # parameter.
                params.add(arg)
            else:
                # Fallback: every other identifier is considered a parameter
                # unless it obviously looks like a literal constant (numbers).
                if not arg.isnumeric():
                    params.add(arg)
    return len(params) if params else 1  # avoid division by zero


def _utility(t: Tactic, matches: int) -> float:
    """Utility according to the paper (|t|-1)*matches / p(t)."""
    return matches * (len(t.steps) - 1) / _num_parameters(t)


def induce_tactics_from_proofs(student_results: List[StudentResult], max_tactics: int = 5, min_occurrences: int = 2, existing_tactics: int = 0) -> List[Tactic]:
    """Induce new tactics from previously-solved proofs.

    The procedure follows the description given in the prompt.  It proceeds
    in three stages:

    1.  Enumerate **all** contiguous subsequences of length ≥ 2 appearing in
       every successful proof seen so far.
    2.  Group subsequences that have the same structure (same list of arrows
       and same arity for each arrow).  For each group we keep track of how
       many times it appears (this is our *m(t, S)*).
    3.  Compute the utility of each candidate and keep the best
       ``max_tactics`` whose occurrence count is at least
       ``min_occurrences``.
    """

    # --- gather contiguous slices ------------------------------------------------
    successful_results = [sr for sr in student_results if sr.success and sr.solution_actions]
    if not successful_results:
        return []

    tactic_buckets: dict[tuple, list[Tactic]] = {}

    for sr_id, sr in enumerate(successful_results):
        # Solution actions alternate between arrow-string and argument-string.
        actions = sr.solution_actions[::2]
        args_raw = sr.solution_actions[1::2]

        # Normalise arguments into List[List[str]] of same length as actions.
        arguments: List[List[str]] = []
        for arg in args_raw:
            arguments.append(arg.split() if arg else [])

        # Pad arguments to same size as actions.
        while len(arguments) < len(actions):
            arguments.append([])

        n = len(actions)
        for start in range(n):
            for length in range(2, min(5, n - start) + 1):
                end = start + length
                name = f'tmp_{sr_id}_{start}_{length}'
                candidate = Tactic.from_solution_slice(name, start, actions[start:end], arguments[start:end])

                sig = _tactic_signature(candidate)
                tactic_buckets.setdefault(sig, []).append(candidate)

    # --- compute utilities -------------------------------------------------------
    scored: List[tuple[float, Tactic, int]] = []
    for sig, tactic_list in tactic_buckets.items():
        occur = len(tactic_list)
        if occur < min_occurrences:
            continue

        # Use *first* instance as representative (they only differ in names).
        rep = tactic_list[0]
        util = _utility(rep, occur)
        scored.append((util, rep, occur))

    # --- select top-k ------------------------------------------------------------
    scored.sort(key=lambda tup: tup[0], reverse=True)
    selected: List[Tactic] = []
    for rank, (util, tac, occur) in enumerate(scored[:max_tactics]):
        tac.name = f'tactic_{rank+existing_tactics}'
        print(f"[Tactic-Induction] Selected {tac.name}  util={util:.2f}  occur={occur}")
        selected.append(tac)

    return selected
