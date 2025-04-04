#!/usr/bin/env python3

"""Implements the conjecture-prove bootstrapping learning loop with Ray parallelization."""

import asyncio
import os
import io
import json
import datetime
import sys
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Union

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from tqdm import tqdm
import ray

import peano
from worker import StudentResult, BackgroundTheory
from hindsight import HindsightExample
from util import format_blocks_with_indent, sample_batch, setup_wandb, value_color, save_json
from conjecture import AgentLM, Context, sample_conjecture
from proofsearch import make_agent, ProofSearchAgent
from problems import load_problemset
import wandb
import random

def now() -> str:
    return '[' + datetime.datetime.now().isoformat() + ']'


FAIL = "fail"


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def create_placement_groups(num_gpus_per_worker: float = 1.0, 
                           num_cpus_per_worker: float = 1.0,
                           max_workers: int = None) -> List:
    """
    Create resource allocation placement groups for efficient GPU utilization.
    
    Args:
        num_gpus_per_worker: Number of GPUs to allocate per worker
        num_cpus_per_worker: Number of CPUs to allocate per worker
        max_workers: Maximum number of workers to create placement groups for
        
    Returns:
        List of placement group IDs
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before creating placement groups")
    
    # Get available resources
    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    available_cpus = int(ray.cluster_resources().get("CPU", 0))
    
    if available_gpus == 0:
        print("No GPUs available, skip creating placement groups")
        return []
    
    # Calculate how many workers we can support
    max_workers_by_gpu = max(1, int(available_gpus / num_gpus_per_worker))
    max_workers_by_cpu = max(1, int(available_cpus / num_cpus_per_worker))
    num_workers = min(max_workers_by_gpu, max_workers_by_cpu)
    
    if max_workers is not None:
        num_workers = min(num_workers, max_workers)
    
    print(f"Creating {num_workers} placement groups with {num_gpus_per_worker} GPUs and {num_cpus_per_worker} CPUs each")
    
    placement_groups = []
    for i in range(num_workers):
        bundle = {"GPU": num_gpus_per_worker, "CPU": num_cpus_per_worker}
        pg = ray.placement_group([bundle], strategy="STRICT_PACK")
        ray.get(pg.ready())
        placement_groups.append(pg)
    
    return placement_groups


def setup_ray(num_gpus_per_worker: float = 1.0, 
              num_cpus_per_worker: float = 1.0,
              address: str = None) -> int:
    """
    Initialize Ray with the specified resources.
    
    Args:
        num_gpus_per_worker: Number of GPUs per worker
        num_cpus_per_worker: Number of CPUs per worker
        address: Optional Ray cluster address to connect to
        
    Returns:
        The number of workers that can be created
    """
    # Initialize Ray
    if address:
        ray.init(address=address)
    else:
        ray.init(ignore_reinit_error=True)
    
    # Get available resources
    available_gpus = ray.cluster_resources().get("GPU", 0)
    available_cpus = ray.cluster_resources().get("CPU", 0)
    
    print(f"Available resources - GPUs: {available_gpus}, CPUs: {available_cpus}")
    
    # Calculate maximum number of workers
    if available_gpus > 0:
        max_workers = max(1, int(available_gpus / num_gpus_per_worker))
        print(f"Can create {max_workers} workers with {num_gpus_per_worker} GPUs each")
    else:
        # If no GPUs, use CPU only mode (use half of available CPUs at most)
        max_workers = max(1, int(available_cpus / (2 * num_cpus_per_worker)))
        print(f"No GPUs available. Using CPU mode with {max_workers} workers")
    
    return max_workers


def ensure_module_imports():
    """
    Ensure that all necessary modules can be imported by the workers.
    """
    # Add the current directory to sys.path if not already there
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)


@ray.remote
class ProofWorker:
    """Worker for proving theorems in parallel using Ray."""
    
    def __init__(self, gpu_id: Optional[int] = None):
        """
        Initialize the worker, setup device, and import necessary modules.
        
        Args:
            gpu_id: Optional GPU ID to use for this worker
        """
        ensure_module_imports()
        
        # Determine which device to use
        if gpu_id is not None and torch.cuda.is_available():
            self.device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
        else:
            self.device = "cpu"
        
        print(f"ProofWorker initialized with device: {self.device}")
        
        # Import needed modules on the worker
        import peano
        import torch
        from proofsearch import ProofSearchAgent
        from worker import StudentResult, BackgroundTheory
        from hindsight import HindsightExample
    
    def try_prove(self, agent_dump: bytes, theory: BackgroundTheory, statement: str) -> StudentResult:
        """
        Attempt to prove a statement using the provided agent.
        
        Args:
            agent_dump: Serialized agent model
            theory: Background theory containing axioms and definitions
            statement: Statement to prove
            
        Returns:
            StudentResult containing the proof attempt results
        """
        from worker import StudentResult
        from hindsight import HindsightExample
        
        try:
            # Load the agent
            with io.BytesIO(agent_dump) as f:
                agent = torch.load(f, map_location=self.device)
                # Move the model to the worker's device
                agent._policy._lm._lm.to(self.device)
            
            # Create derivation from the theory
            d = peano.PyDerivation()
            d.incorporate(theory.theory)
            
            # Create initial state
            try:
                proof_state = d.initial_state(statement, theory.premises)
            except Exception as e:
                return StudentResult(
                    error=f"Error initializing proof state: {str(e)}",
                    success=False,
                    problem=statement,
                    solution_actions=None,
                    proof=None,
                    extracted_examples=[],
                    hindsight_examples=[],
                    iterations=0,
                    logprob=float('-inf')
                )
            
            # Run proof search
            search_result = agent.proof_search(proof_state.goal(), proof_state)
            
            # Extract solution actions if successful
            solution_actions = None
            if search_result.success:
                solution = search_result.root.solution()
                solution_actions = []
                
                # Extract the actions and arguments from the solution
                for step in solution:
                    solution_actions.append(step.action)
                    if step.arguments:
                        solution_actions.append(' '.join(step.arguments))
                    else:
                        solution_actions.append('')
            
            # Compute log probability of the proof if successful
            logprob = float('-inf')
            if search_result.success and solution:
                logprob = sum(step.logprob for step in solution)
            
            # Prepare the result
            return StudentResult(
                error=None,
                success=search_result.success,
                problem=statement,
                solution_actions=solution_actions,
                proof=d.proof(proof_state) if search_result.success else None,
                extracted_examples=search_result.examples,
                hindsight_examples=[], # Hindsight examples would be added in a more complex implementation
                iterations=search_result.iterations,
                logprob=logprob
            )
        
        except Exception as e:
            import traceback
            error_msg = f"Error in proof worker: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            return StudentResult(
                error=error_msg,
                success=False,
                problem=statement,
                solution_actions=None,
                proof=None,
                extracted_examples=[],
                hindsight_examples=[],
                iterations=0,
                logprob=float('-inf')
            )


@ray.remote
class EvaluationWorker:
    """Worker for evaluating the agent on problem sets in parallel using Ray."""
    
    def __init__(self, gpu_id: Optional[int] = None):
        """
        Initialize the worker, setup device, and import necessary modules.
        
        Args:
            gpu_id: Optional GPU ID to use for this worker
        """
        ensure_module_imports()
        
        # Determine which device to use
        if gpu_id is not None and torch.cuda.is_available():
            self.device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
        else:
            self.device = "cpu"
        
        print(f"EvaluationWorker initialized with device: {self.device}")
        
        # Import needed modules on the worker
        import peano
        import torch
        from problems import ProblemSet
    
    def evaluate_problem(self, agent_dump: bytes, problem_data: dict) -> Tuple[str, bool]:
        """
        Evaluate the agent on a single problem.
        
        Args:
            agent_dump: Serialized agent model
            problem_data: Problem data including theory, problem name, and problem specification
            
        Returns:
            Tuple of (problem_name, success)
        """
        try:
            import peano
            
            # Load the agent
            with io.BytesIO(agent_dump) as f:
                agent = torch.load(f, map_location=self.device)
                # Move the model to the worker's device
                agent._policy._lm._lm.to(self.device)
            
            # Extract problem information
            problem_name = problem_data['name']
            theory_text = problem_data['theory']
            premises = problem_data.get('premises', [])
            problem_spec = problem_data['specification']
            
            # Create derivation
            d = peano.PyDerivation()
            d.incorporate(theory_text)
            
            # Initialize the problem
            try:
                initial_state = d.initial_state(problem_spec, premises)
            except Exception as e:
                print(f"Error initializing problem {problem_name}: {str(e)}")
                return problem_name, False
            
            # Run proof search
            try:
                agent_result = agent.proof_search(initial_state.goal(), initial_state)
                return problem_name, agent_result.success
            except Exception as e:
                print(f"Error in proof search for {problem_name}: {str(e)}")
                return problem_name, False
                
        except Exception as e:
            import traceback
            print(f"Error in evaluation worker: {str(e)}\n{traceback.format_exc()}")
            return problem_data.get('name', 'unknown'), False


@ray.remote
class ConjectureWorker:
    """Worker for generating conjectures in parallel using Ray."""
    
    def __init__(self, gpu_id: Optional[int] = None):
        """
        Initialize the worker, setup device, and import necessary modules.
        
        Args:
            gpu_id: Optional GPU ID to use for this worker
        """
        ensure_module_imports()
        
        # Determine which device to use
        if gpu_id is not None and torch.cuda.is_available():
            self.device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
        else:
            self.device = "cpu"
        
        print(f"ConjectureWorker initialized with device: {self.device}")
        
        # Import needed modules on the worker
        import peano
        import torch
        from conjecture import AgentLM, Context, sample_conjecture
    
    def generate_conjectures(self, agent_dump: bytes, theory_text: str, 
                            num_conjectures: int, known_conjectures: List[str], 
                            max_attempts: int = 100) -> List[str]:
        """
        Generate new conjectures using the agent.
        
        Args:
            agent_dump: Serialized agent model
            theory_text: Theory text containing axioms and definitions
            num_conjectures: Number of conjectures to generate
            known_conjectures: List of already known conjectures to filter out
            max_attempts: Maximum number of attempts to generate conjectures
            
        Returns:
            List of generated conjectures
        """
        try:
            import peano
            import torch
            from conjecture import AgentLM, Context, sample_conjecture
            
            # Load the agent
            with io.BytesIO(agent_dump) as f:
                agent = torch.load(f, map_location=self.device)
                # Move the model to the worker's device
                agent._policy._lm._lm.to(self.device)
            
            # Setup context
            d = peano.PyDerivation()
            d.incorporate(theory_text)
            context = Context(d, None, [])
            
            # Generate conjectures
            conjectures = []
            attempts = 0
            
            while len(conjectures) < num_conjectures and attempts < max_attempts:
                attempts += 1
                
                try:
                    # Generate a new conjecture
                    proposal = sample_conjecture(AgentLM(agent, 'Conj:(hard) '), context)
                    
                    if proposal and proposal not in conjectures + known_conjectures:
                        # Contract conjectures to make them Peano-parseable
                        contracted_proposal = d.contract(proposal)
                        if contracted_proposal not in conjectures + known_conjectures:
                            conjectures.append(contracted_proposal)
                except Exception as e:
                    print(f"Error generating conjecture: {str(e)}")
                    continue
            
            return conjectures
            
        except Exception as e:
            import traceback
            print(f"Error in conjecture worker: {str(e)}\n{traceback.format_exc()}")
            return []


def test_on_pset_parallel(agent, problemset, premises: List[str], max_workers: int) -> float:
    """
    Evaluate the agent on a problem set in parallel using Ray.
    
    Args:
        agent: The agent to evaluate
        problemset: The problem set to evaluate on
        premises: Premises to use for the problems
        max_workers: Maximum number of workers to use
        
    Returns:
        Success rate (float between 0 and 1)
    """
    print(now(), f"Evaluating agent on {len(problemset.problem_names())} problems in parallel...")
    
    # Serialize the agent
    agent_buffer = io.BytesIO()
    torch.save(agent, agent_buffer)
    agent_dump = agent_buffer.getvalue()
    
    # Prepare problems data
    problems = problemset.problem_names()
    theory_text = problemset.get_theory_text()
    
    # Structure problem data for serialization
    problem_data_list = []
    for problem in problems:
        problem_data = {
            'name': problem,
            'theory': theory_text,
            'premises': premises,
            'specification': problemset.get_problem_specification(problem)
        }
        problem_data_list.append(problem_data)
    
    # Create worker pool
    workers = []
    for i in range(min(max_workers, len(problems))):
        # Check if GPUs are available
        if torch.cuda.is_available():
            worker = EvaluationWorker.remote(gpu_id=i % torch.cuda.device_count())
        else:
            worker = EvaluationWorker.options(num_gpus=0).remote(gpu_id=None)
        workers.append(worker)
    
    # Distribute tasks to workers (round-robin assignment)
    tasks = []
    for i, problem_data in enumerate(problem_data_list):
        worker_idx = i % len(workers)
        task = workers[worker_idx].evaluate_problem.remote(agent_dump, problem_data)
        tasks.append(task)
    
    # Collect results with progress bar
    results = []
    for i, result_id in enumerate(tqdm(tasks, desc="Evaluating problems")):
        problem_name, success = ray.get(result_id)
        results.append((problem_name, success))
    
    # Calculate success rate
    successes = [success for _, success in results]
    success_rate = np.mean(successes)
    
    print(f"Evaluation complete. Success rate: {success_rate:.4f}")
    return success_rate


def generate_conjectures_parallel(agent, theory_text: str, n_conjectures: int, 
                                 proven_conjectures: List[str], max_workers: int) -> List[str]:
    """
    Generate conjectures in parallel using Ray.
    
    Args:
        agent: The agent to use for generation
        theory_text: Theory text containing axioms and definitions
        n_conjectures: Number of conjectures to generate
        proven_conjectures: List of already proven conjectures to filter out
        max_workers: Maximum number of workers to use
        
    Returns:
        List of generated conjectures
    """
    print(now(), f"Generating {n_conjectures} conjectures in parallel...")
    
    # Serialize the agent
    agent_buffer = io.BytesIO()
    torch.save(agent, agent_buffer)
    agent_dump = agent_buffer.getvalue()
    
    # Create worker pool
    workers = []
    for i in range(min(max_workers, n_conjectures)):
        # Check if GPUs are available
        if torch.cuda.is_available():
            worker = ConjectureWorker.remote(gpu_id=i % torch.cuda.device_count())
        else:
            worker = ConjectureWorker.options(num_gpus=0).remote(gpu_id=None)
        workers.append(worker)
    
    # Calculate conjectures per worker
    base_conjs_per_worker = n_conjectures // len(workers)
    extra_conjs = n_conjectures % len(workers)
    
    # Distribute tasks to workers
    tasks = []
    for i in range(len(workers)):
        # Workers with index < extra_conjs get one extra conjecture
        worker_conjs = base_conjs_per_worker + (1 if i < extra_conjs else 0)
        
        if worker_conjs > 0:
            task = workers[i].generate_conjectures.remote(
                agent_dump, 
                theory_text, 
                worker_conjs, 
                proven_conjectures,
                max_attempts=worker_conjs * 10  # Allow more attempts than required conjectures
            )
            tasks.append(task)
    
    # Collect results with progress bar
    conjectures = []
    for result_id in tqdm(tasks, desc="Generating conjectures"):
        worker_conjectures = ray.get(result_id)
        conjectures.extend(worker_conjectures)
    
    print(f"Generated {len(conjectures)} conjectures")
    
    # If we didn't get enough conjectures, try to generate more sequentially
    if len(conjectures) < n_conjectures:
        print(f"Warning: Generated only {len(conjectures)}/{n_conjectures} conjectures. Trying to generate more sequentially.")
        
        # Import necessary modules
        from conjecture import AgentLM, Context, sample_conjecture
        
        # Create context
        d = peano.PyDerivation()
        d.incorporate(theory_text)
        context = Context(d, None, [])
        
        # Generate more conjectures sequentially
        with tqdm(total=(n_conjectures - len(conjectures))) as pbar:
            attempts = 0
            while len(conjectures) < n_conjectures and attempts < (n_conjectures - len(conjectures)) * 10:
                attempts += 1
                try:
                    proposal = sample_conjecture(AgentLM(agent, 'Conj:(hard) '), context)
                    
                    if proposal and proposal not in conjectures + proven_conjectures:
                        # Contract conjectures to make them Peano-parseable
                        contracted_proposal = d.contract(proposal)
                        if contracted_proposal not in conjectures + proven_conjectures:
                            conjectures.append(contracted_proposal)
                            pbar.update(1)
                except Exception as e:
                    print(f"Error generating conjecture: {str(e)}")
                    continue
    
    print(f"Final conjecture count: {len(conjectures)}/{n_conjectures}")
    return conjectures


def prove_conjectures_parallel(agent, theory_text: str, premises: List[str], 
                              conjectures: List[str], max_workers: int) -> List[StudentResult]:
    """
    Attempt to prove conjectures in parallel using Ray.
    
    Args:
        agent: The agent to use for proving
        theory_text: Theory text containing axioms and definitions
        premises: Premises to use for the proofs
        conjectures: List of conjectures to prove
        max_workers: Maximum number of workers to use
        
    Returns:
        List of StudentResult objects containing proof results
    """
    print(now(), f"Attempting to prove {len(conjectures)} conjectures in parallel...")
    
    # Serialize the agent
    agent_buffer = io.BytesIO()
    torch.save(agent, agent_buffer)
    agent_dump = agent_buffer.getvalue()
    
    # Create background theory object
    theory = BackgroundTheory(theory_text, premises)
    
    # Create worker pool
    workers = []
    for i in range(min(max_workers, len(conjectures))):
        # Check if GPUs are available
        if torch.cuda.is_available():
            worker = ProofWorker.remote(gpu_id=i % torch.cuda.device_count())
        else:
            worker = ProofWorker.options(num_gpus=0).remote(gpu_id=None)
        workers.append(worker)
    
    # Distribute tasks to workers (round-robin assignment)
    tasks = []
    for i, conjecture in enumerate(conjectures):
        worker_idx = i % len(workers)
        task = workers[worker_idx].try_prove.remote(agent_dump, theory, conjecture)
        tasks.append(task)
    
    # Collect results with progress bar
    results = []
    for result_id in tqdm(tasks, desc="Proving conjectures"):
        student_result = ray.get(result_id)
        if not student_result.error:  # Filter out errors
            results.append(student_result)
        else:
            print(f"Error in proof attempt: {student_result.error}")
    
    # Report success rate
    successes = [r.success for r in results]
    success_rate = np.mean(successes) if successes else 0
    print(f"Proof attempts complete. Success rate: {success_rate:.4f}")
    
    return results


async def teacher_loop(cfg: DictConfig):
    print('Running with Ray parallelization.')
    
    # Initialize Ray
    max_workers = setup_ray(
        num_gpus_per_worker=cfg.get('num_gpus_per_worker', 1.0),
        num_cpus_per_worker=cfg.get('num_cpus_per_worker', 1.0),
        address=cfg.get('ray_address', None)
    )
    
    # Create placement groups for efficient resource allocation
    placement_groups = create_placement_groups(
        num_gpus_per_worker=cfg.get('num_gpus_per_worker', 1.0),
        num_cpus_per_worker=cfg.get('num_cpus_per_worker', 1.0),
        max_workers=max_workers
    )
    
    # Create the agent
    agent = make_agent(cfg)
    
    # Load theory
    with open(os.path.join(os.path.dirname(__file__), 'theories', cfg.theory.name + '.p')) as f:
        theory = f.read()
    
    # Setup difficulty buckets
    difficulty_buckets = sorted([list(cfg.difficulty_buckets[i].items())[0]
                               for i in range(len(cfg.difficulty_buckets))],
                              key=lambda kv: kv[1])
    
    # Load test problems
    premises = cfg.theory.premises
    test_problemset = load_problemset(cfg.test_problems)
    
    # Initialize Peano derivation
    d = peano.PyDerivation()
    d.incorporate(theory)
    
    # Initialize tracking variables
    proven_conjectures = []
    seen_hindsight_goals = set()
    proofs = []
    outcomes = []
    induced_tactics = []
    
    # Check if continuing from previous run
    continue_dir = cfg.get('continue')
    start_iteration = 0
    
    if continue_dir is not None:
        os.chdir(continue_dir)
        print('Continuing run from', continue_dir)
        # Find largest iteration number such that i.pt exists.
        i = 0
        while os.path.exists(f'{i}.pt'):
            i += 1
        i -= 1
        start_iteration = i
        agent = torch.load(f'{i}.pt')
        print('Loaded agent from', f'{i}.pt')
        # Load examples and outcomes.
        if i > 0:
            with open(f'outcomes_{i-1}.json', 'r') as f:
                outcomes = json.load(f)
                proven_conjectures = [o['problem'] for o in outcomes
                                      if o['hindsight'] is False and
                                         o['proof'] is not None]
                seen_hindsight_goals = {o['problem'] for o in outcomes
                                        if o['hindsight'] and o['proof'] is not None}
    
        print('Loaded', len(proven_conjectures), 'proven conjectures from previous run.')
    
    # Check if conjecturer should be frozen (ablation)
    if cfg.get('freeze_conjecturer', False):
        print('Ablation: Freezing conjecturer.')
    
    # Main training loop
    with open('log.jsonl', 'w') as log:
        for i in range(start_iteration, cfg.iterations):
            # Step 2: Evaluate agent on test problems
            test_success_rate = test_on_pset_parallel(
                agent, 
                test_problemset, 
                premises,
                max_workers
            )
            
            print('Test success rate:', test_success_rate)
            log.write(json.dumps({'iteration': i,
                                  'msg': f'Test success rate: {test_success_rate}'}))
            log.write('\n')
            wandb.log({'test_success_rate': test_success_rate})
            torch.save(agent, f'{i}.pt')
            
            # Step 3: Generate conjectures
            print(now(), f'Iteration #{i}: making conjectures...')
            
            conjectures = generate_conjectures_parallel(
                agent,
                theory,
                cfg.n_conjectures,
                proven_conjectures,
                max_workers
            )
            
            print(now(), 'done, have', len(conjectures), 'conjectures')
            print(conjectures)
            
            log.write(json.dumps({'iteration': i,
                                  'msg': f'It #{i}: posing {len(conjectures)} conjectures.',
                                  'conjectures': conjectures}))
            log.write('\n')
            log.flush()
            
            # Step 4: Try to prove the conjectures
            student_results = prove_conjectures_parallel(
                agent,
                theory,
                premises,
                conjectures,
                max_workers
            )
            
            # Process results
            success_logprobs = []
            examples = []
            
            # Look at all the success logprobs and compute the easy/hard threshold
            for student_result in student_results:
                if student_result.success:
                    success_logprobs.append(student_result.logprob)
                
                outcomes.append({'iteration': i,
                                 'problem': student_result.problem,
                                 'proof': student_result.proof,
                                 'logprob': student_result.logprob,
                                 'actions': student_result.solution_actions,
                                 'hindsight': False
                                 })
                
                for h in student_result.hindsight_examples:
                    outcomes.append({'iteration': i,
                                     'problem': h.statement,
                                     'proof': h.proof,
                                     'logprob': h.logprob,
                                     'actions': h.solution_actions,
                                     'hindsight': True
                                     })
            
            if not success_logprobs:
                print(f'No solutions found in iteration {i} - stopping learning loop...')
                break
            
            # Induce tactics from successful proofs if enabled
            if cfg.get('induce_tactics', False):
                new_tactics = induce_tactics_from_proofs(
                    student_results, 
                    max_tactics=cfg.get('max_tactics', 5),
                    min_occurrences=cfg.get('min_tactic_occurrences', 2)
                )
                
                if new_tactics:
                    induced_tactics.extend(new_tactics)
                    print(f"Induced {len(new_tactics)} new tactics, total: {len(induced_tactics)}")
                    
                    # Save induced tactics
                    with open(f'tactics_{i}.json', 'w') as f:
                        tactics_data = [{'name': t.name, 
                                        'steps': [{'arrows': list(s.arrows), 
                                                  'arguments': list(s.arguments), 
                                                  'result': s.result} 
                                                 for s in t.steps]} 
                                       for t in new_tactics]
                        json.dump(tactics_data, f, indent=2)
                    
                    # Also save a combined tactics file for easier loading
                    with open('tactics.json', 'w') as f:
                        tactics_data = [{'name': t.name, 
                                        'steps': [{'arrows': list(s.arrows), 
                                                  'arguments': list(s.arguments), 
                                                  'result': s.result} 
                                                 for s in t.steps]} 
                                       for t in induced_tactics]
                        json.dump(tactics_data, f, indent=2)
                    
                    log.write(json.dumps({'iteration': i,
                                         'msg': f'Induced {len(new_tactics)} new tactics, total: {len(induced_tactics)}'}))
                    log.write('\n')
                    
                    # Update the agent with the new tactics if enabled
                    if cfg.get('use_induced_tactics', True):
                        from proofsearch import HolophrasmNode
                        HolophrasmNode.set_tactics(induced_tactics)
                        print(f"Updated agent with {len(induced_tactics)} tactics")
            
            # Calculate thresholds for difficulty buckets
            thresholds = [np.percentile(success_logprobs, p)
                          for _, p in difficulty_buckets]
            
            print('Thresholds:',
                  list(zip([k for k, _ in difficulty_buckets], thresholds)),
                  'min =', np.min(success_logprobs),
                  'max =', np.max(success_logprobs))
            
            # Classify problems into easy/hard
            for student_result in student_results:
                # Outcome is the name of the first difficulty bucket that is larger than the logprob
                if student_result.success:
                    outcome = next(k
                                   for i, (k, _) in enumerate(difficulty_buckets)
                                   if (student_result.logprob <= thresholds[i] or
                                       i + 1 == len(difficulty_buckets)))
                else:
                    outcome = FAIL
                
                if not cfg.get('freeze_conjecturer', False):
                    examples.append(f'Conj:({outcome}) ' + d.elaborate(student_result.problem))
                
                if student_result.success:
                    proven_conjectures.append(student_result.problem)
                    proofs.append(student_result.proof)
                
                examples.extend(student_result.extracted_examples)
                
                if cfg.train_policy_on_hindsight_examples:
                    for h in student_result.hindsight_examples:
                        if h.goal not in seen_hindsight_goals:
                            outcome = next(k
                                           for i, (k, _) in enumerate(difficulty_buckets)
                                           if h.logprob <= thresholds[i] or i + 1 == len(difficulty_buckets))
                            
                            if not cfg.get('freeze_conjecturer', False):
                                examples.append(f'Conj:({outcome}) ' + d.elaborate(student_result.problem))
                            examples.extend(h.examples)
                            seen_hindsight_goals.add(h.goal)
            
            log.write(json.dumps({'iteration': i,
                                  'msg': f'Training on {len(examples)} examples.'}))
            log.write('\n')
            
            # Train model on conjecturing and proof search examples
            if i + 1 < cfg.iterations:
                print(len(examples), 'accumulated training examples.')
                agent.train(examples)
            
            # Save results
            save_json(examples, f'examples_{i}.json')
            save_json(outcomes, f'outcomes_{i}.json')
            torch.save(student_results, f'results_{i}.json')
    
    # Shut down Ray
    ray.shutdown()


@hydra.main(version_base="1.2", config_path="config", config_name="bootstrap")
def main(cfg: DictConfig):
    print('Running from:', os.getcwd())
    set_seed(cfg.seed)
    setup_wandb(cfg)
    if cfg.task == 'teacher':
        asyncio.run(teacher_loop(cfg))


if __name__ == '__main__':
    main()