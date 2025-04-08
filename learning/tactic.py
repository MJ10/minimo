from dataclasses import dataclass
from typing import List, Optional

from worker import StudentResult

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
        return f'{self.name}:\n' + '\n'.join(map(str, self.steps))
    
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


def induce_tactics_from_proofs(student_results: List[StudentResult], max_tactics: int = 5, min_occurrences: int = 2) -> List[Tactic]:
    # Extract successful proofs
    successful_results = [sr for sr in student_results if sr.success and sr.solution_actions]
    
    if not successful_results:
        print("No successful proofs to induce tactics from.")
        return []
    
    # Create tactics from solution slices
    tactics_from_slices = []
    
    for i, sr in enumerate(successful_results):
        # Extract actions and arguments
        # Actions are in odd positions, arguments in even positions
        actions = sr.solution_actions[::2]
        
        # Create arguments list (may be None for some steps)
        arguments = []
        for j in range(len(actions)):
            if j*2+1 < len(sr.solution_actions):
                arg_str = sr.solution_actions[j*2+1]
                # Parse arguments by splitting the string, assuming arguments are space-separated
                if arg_str:
                    arguments.append(arg_str.split())
                else:
                    arguments.append([])
            else:
                arguments.append([])
        
        # Generate tactics from different slices of the proof
        for start in range(len(actions) - 1):
            # Consider different lengths for tactics - longer tactics are more powerful but less reusable
            for length in range(2, min(len(actions) - start + 1, 5)):  # Limit length to avoid too complex tactics
                tactic_name = f't_{i}_{start}_{length}'
                
                # Skip creating tactics from disconnected parts of the proof 
                # (e.g. where one step doesn't depend on the previous)
                is_connected = True
                for step_idx in range(start + 1, start + length):
                    # Check if this step depends on a previous result within this slice
                    has_dependency = False
                    for prev_step in range(start, step_idx):
                        if prev_step < len(arguments) and step_idx < len(arguments):
                            # Check if any argument in the current step references a result from a previous step
                            for arg in arguments[step_idx]:
                                if arg.startswith("!step") and int(arg[5:]) >= start and int(arg[5:]) < start + length:
                                    has_dependency = True
                                    break
                    
                    if not has_dependency:
                        is_connected = False
                        break
                
                if not is_connected:
                    continue
                
                t = Tactic.from_solution_slice(
                    tactic_name, 
                    start,
                    actions[start:start+length],
                    arguments[start:start+length],
                    True
                )
                tactics_from_slices.append(t)
    
    print(f"Generated {len(tactics_from_slices)} tactic slices")
    
    # Count occurrences of similar tactics
    tactic_counts = {}
    
    for t in tactics_from_slices:
        # Create a signature for the tactic based on its structure
        # Using a more detailed signature considering both arrows and argument counts
        signature = tuple((step.arrows, len(step.arguments)) for step in t.steps)
        
        if signature in tactic_counts:
            tactic_counts[signature][1] += 1
        else:
            tactic_counts[signature] = [t, 1]
    
    # Filter and sort tactics by occurrence count
    filtered_tactics = [(t, count) for (t, count) in tactic_counts.values() 
                        if count >= min_occurrences]
    filtered_tactics.sort(key=lambda x: x[1], reverse=True)
    
    # Select top tactics
    selected_tactics = []
    for t, count in filtered_tactics[:max_tactics]:
        # Rename the tactic with a more meaningful name
        t.name = f"tactic_{len(selected_tactics)}"
        print(f"Selected tactic with {count} occurrences:\n{t}")
        selected_tactics.append(t)
    
    return selected_tactics
