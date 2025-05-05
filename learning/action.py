#!/usr/bin/env python3
# Wrapper for PyProofAction that allows for sequences of actions to be chained together.

import functools
from typing import List, Tuple, Optional, Union, Dict

import peano
from dataclasses import dataclass


class ProofAction:
    def __init__(self, peano_actions: list):
        self._actions = peano_actions

    def __str__(self) -> str:
        return ' => '.join(map(str, self._actions))

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, rhs) -> bool:
        return isinstance(rhs, ProofAction) and self._actions == rhs._actions

    def is_intro(self) -> bool:
        return len(self._actions) == 1 and self._actions[0].is_intro()

    def is_construct(self):
        return len(self._actions) > 0 and len(self._actions) <= 2 and self._actions[0].is_construct()

    def is_apply(self):
        return len(self._actions) > 0 and len(self._actions) <= 2 and self._actions[0].is_apply()
        
    def is_tactic(self):
        return False

    def execute(self, state: peano.PyProofState) -> peano.PyProofState:
        return functools.reduce(lambda s, a: s[0].execute_action(a),
                                self._actions, [state])

    def is_eager(self):
        return self.is_intro() or len(self._actions) == 2

    def arrow_name(self):
        assert not self.is_intro()
        return str(self._actions[0]).split()[1]

    def construction_dtype(self):
        c_a = (self._actions[1]
               if self._actions[0].is_construct()
               else self._actions[0])
        dtype, _value = c_a.selected_construction()
        return str(dtype)


@dataclass(eq=True, frozen=True)
class Step:
    """Represents one step in a tactic."""
    arrows: tuple[str]
    arguments: tuple[str]
    result: str
    branch: Optional[int]

    def __init__(self, arrows: List[str], arguments: List[str], result: str,
                 branch: Optional[int] = None):
        object.__setattr__(self, 'arrows', tuple(arrows))
        object.__setattr__(self, 'arguments', tuple(arguments))
        object.__setattr__(self, 'result', result)
        object.__setattr__(self, 'branch', branch)

    def __str__(self):
        c = f' ~~> {self.branch}' if self.branch is not None else ''
        arrows = self.arrows[0] if len(self.arrows) == 1 else f'({"|".join(self.arrows)})'
        return f'{self.result} <- {arrows} {", ".join(self.arguments)}{c}'


class Tactic:
    def __init__(self, name: str, steps: List[Step]):
        self.steps = tuple(steps)
        self.name = name
        
    def __str__(self):
        return f'TACTIC:{self.name}'
    
    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def from_solution_slice(name: str, start_index: int, arrows: List[str], 
                           arguments: List[List[str]], abstract_constants: bool = True) -> 'Tactic':
        """Constructs a tactic from a slice of a solution found in a search episode."""
        steps = []
        rewrites = {}

        for i, (arrow, args) in enumerate([(arr, args) 
                                          for arr, args in zip(arrows, arguments) 
                                          if args is not None]):
            result = f'?{i}'
            rewrites[f'!step{start_index + i}'] = result
            
            # Rewrite argument names based on previous steps
            rewritten_args = []
            for arg in args:
                if arg in rewrites:
                    rewritten_args.append(rewrites[arg])
                else:
                    rewritten_args.append(arg)
            
            steps.append(Step([arrow], rewritten_args, result))
            
        return Tactic(name, steps)


class TacticAction(ProofAction):
    """Class representing a tactic as an action in the proof search."""
    
    def __init__(self, tactic: Tactic):
        self.tactic = tactic
        # Initialize with empty actions - they will be constructed during execution
        super().__init__([])
        
    def __str__(self) -> str:
        return f"TACTIC:{self.tactic.name}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, rhs) -> bool:
        if not isinstance(rhs, TacticAction):
            return False
        return self.tactic.name == rhs.tactic.name
    
    def is_tactic(self):
        return True
        
    def execute(self, state: peano.PyProofState) -> peano.PyProofState:
        """Execute the tactic on the given proof state.

        This rewritten version avoids brittle substring matching by
        1.  Extracting the *exact* arrow name of every Peano action via
            a simple token split (first token).
        2.  Normalising tokens (stripping common punctuation) before
            comparison so that, e.g., `(succ x)` and `succ x` both match.
        3.  Using a small helper to perform argument substitution &
            matching, allowing ``*`` wild-cards and placeholders that
            refer to results of previous steps.
        """
        def _split_action(action_obj):
            """Return `(arrow, [arg0, arg1, …])` from a Peano action."""
            parts = str(action_obj).strip().split()
            arrow = parts[0] if parts else ''
            return arrow, parts[1:]

        def _norm(tok: str) -> str:
            """Remove common punctuation around a token for robust matching."""
            return tok.strip(',()')

        current_states = [state]
        result_substitutions: dict[str, str] = {}

        for step in self.tactic.steps:
            if not current_states:
                # All branches failed.
                return []

            next_states_acc: list[peano.PyProofState] = []

            for current_state in current_states:
                matched_in_this_state = False
                for action in current_state.actions():
                    arrow_name, act_args = _split_action(action)

                    # Arrow must match one of the allowed names.
                    if arrow_name not in step.arrows:
                        continue

                    # --- argument matching ---------------------------------
                    if step.arguments:
                        # Prepare expected arguments after substitution.
                        expected_args: list[str] = []
                        for arg in step.arguments:
                            if arg in result_substitutions:
                                expected_args.append(result_substitutions[arg])
                            else:
                                expected_args.append(arg)

                        # Length must agree (unless wildcard absorbs the rest).
                        if len(act_args) != len(expected_args):
                            continue

                        # Check each argument.
                        ok = True
                        for exp, real in zip(expected_args, act_args):
                            exp_n, real_n = _norm(exp), _norm(real)
                            if exp_n == '*' or exp_n == real_n:
                                continue
                            ok = False
                            break
                        if not ok:
                            continue
                    # -------------------------------------------------------

                    # At this point we have a match; execute it.
                    next_states = current_state.execute_action(action)
                    if not next_states:
                        continue

                    # Register produced result if the step defines one.
                    if step.result:
                        construction = next_states[0].construction_from_last_action()
                        if construction:
                            result_substitutions[step.result] = construction

                    next_states_acc.extend(next_states)
                    matched_in_this_state = True
                    # Do *not* break here – there might be alternative
                    # actions in the same state that also match and we want
                    # to explore every branch.
                # end for action

                if not matched_in_this_state:
                    # This particular branch failed → discard it.
                    continue
            # end for current_state

            # Advance to the union of successor states.
            current_states = next_states_acc

        return current_states

    # ------------------------------------------------------------------
    # Applicability check ------------------------------------------------
    # ------------------------------------------------------------------

    def is_applicable(self, state: 'peano.PyProofState') -> bool:
        """Returns True if executing the tactic on *state* yields at least one
        successor state.  It is a lightweight wrapper around *execute* that
        avoids constructing TreeSearch nodes when the tactic clearly fails.
        """
        try:
            next_states = self.execute(state)
            return bool(next_states)
        except Exception:
            # If anything goes wrong we consider the tactic not applicable.
            return False

