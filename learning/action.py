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
        """Execute the tactic on the given proof state."""
        current_states = [state]
        result_substitutions = {}  # Maps from result placeholders to actual terms
        
        # Apply each step of the tactic in sequence
        for step in self.tactic.steps:
            if not current_states:
                break
                
            # Get the current state to work with
            current_state = current_states[0]
            
            # Find the matching action in the current state
            found_action = False
            for action in current_state.actions():
                # Attempt to match the action with the step
                action_str = str(action)
                
                for arrow in step.arrows:
                    # Use exact matching instead of substring matching
                    if action_str.startswith(arrow + " ") or action_str == arrow:
                        # If this step has arguments, check if they match what we need
                        if step.arguments:
                            # Parse the action arguments
                            action_parts = action_str.split(" ")
                            if len(action_parts) > 1:  # Has arguments
                                # Check if arguments match
                                arg_match = True
                                actual_args = []
                                
                                for arg in step.arguments:
                                    # If the argument is a result reference from a previous step
                                    if arg in result_substitutions:
                                        actual_args.append(result_substitutions[arg])
                                    else:
                                        # Use the argument directly
                                        actual_args.append(arg)
                                
                                # If arguments don't match or wrong number of arguments, continue to next action
                                if len(actual_args) != len(action_parts) - 1:
                                    continue
                                    
                                # Try to match each argument (this is a simple check, might need refinement)
                                for i, arg in enumerate(actual_args):
                                    if arg != action_parts[i + 1] and arg != "*":  # Allow "*" as wildcard
                                        arg_match = False
                                        break
                                        
                                if not arg_match:
                                    continue
                        
                        # Found a matching action, execute it
                        next_states = current_state.execute_action(action)
                        if next_states:
                            # If this step produces a result, store it in our substitutions
                            if step.result:
                                # For now, we'll just use the first construction from the last action
                                # This is a simplification - in reality we might need more sophisticated logic
                                construction = next_states[0].construction_from_last_action()
                                if construction:
                                    result_substitutions[step.result] = construction
                            
                            current_states = next_states
                            found_action = True
                            break
                
                if found_action:
                    break
            
            # If no matching action was found, tactic execution fails
            if not found_action:
                return []
        
        return current_states

