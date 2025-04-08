#!/usr/bin/env python3

"""Implements the conjecture-prove bootstrapping learning loop with Ray parallelization."""

import asyncio
import os
import io
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple
import time

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from tqdm import tqdm
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import placement_group, PlacementGroup

import peano
from util import format_blocks_with_indent, sample_batch, setup_wandb, value_color, save_json
from conjecture import AgentLM, Context, sample_conjecture
from proofsearch import make_agent
from problems import load_problemset
import wandb
from dataclasses import dataclass
from bootstrap import Step, Tactic, induce_tactics_from_proofs, HolophrasmNode
from hindsight import HindsightExample
from worker import StudentResult, BackgroundTheory

def now() -> str:
    return '[' + datetime.datetime.now().isoformat() + ']'


FAIL = "fail"


@ray.remote(num_gpus=1, num_cpus=1)
class ProofWorker:
    """Ray worker for proof search tasks."""
    
    def __init__(self, gpu_id: int = 0):
        # Import necessary modules
        import sys
        import os
        
        # Add the current directory to the path so workers can find modules
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
            
        # Import modules needed by this worker
        import torch
        import peano
        import policy
        import proofsearch
        import hindsight
        import worker
        from worker import StudentResult, BackgroundTheory
        
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        print(f"Initialized ProofWorker with device: {self.device}")

    def try_prove(self, agent_dump: bytes, theory: BackgroundTheory, statement: str) -> StudentResult:
        """Attempt to prove a given statement."""
        try:
            # Load the agent on the worker's device
            with io.BytesIO(agent_dump) as f:
                agent = torch.load(f, weights_only=False)
                # Move agent to this worker's device
                agent._policy._lm._lm.to(self.device)

            print(f'Proving {statement} on {self.device}')
            
            state = peano.PyProofState(theory.theory,
                                      theory.premises,
                                      statement)

            agent_result = agent.proof_search(statement, state)

            if agent_result.success:
                proof = agent_result.root.state_node.reconstruct_proof(
                    agent_result.root.get_solution_actions())
                solution_actions = agent_result.root.get_solution_actions()
                logprob = agent_result.root.solution_logprob_under_policy(agent._policy, solution_actions)
            else:
                solution_actions, proof, logprob = None, None, None

            examples = []
            # Policy examples for the proved goal.
            examples.extend(agent._policy.extract_examples(root=agent_result.root))
            # Hindsight examples (policy + conjecturing).
            hindsight_examples = []
            from hindsight import extract_hindsight_examples
            hindsight_examples = extract_hindsight_examples(
                    agent_result.root,
                    theory.theory,
                    theory.premises,
                    agent._policy)

            return StudentResult(
                None,
                agent_result.success,
                statement,
                list(map(str, solution_actions)) if solution_actions else None,
                proof,
                agent_result.examples,
                hindsight_examples,
                agent_result.iterations,
                logprob,
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print('Error in try_prove!')
            print(tb)
            return StudentResult(tb, False, statement, None, None, [],
                              None, None, None)


@ray.remote(num_gpus=1, num_cpus=1)
class EvaluationWorker:
    """Ray worker for evaluating the agent on test problems."""
    
    def __init__(self, gpu_id: int = 0):
        # Import necessary modules
        import sys
        import os
        
        # Add the current directory to the path so workers can find modules
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
            
        # Import modules needed by this worker
        import torch
        import peano
        import policy
        import proofsearch
        
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        print(f"Initialized EvaluationWorker with device: {self.device}")
    
    def evaluate_problem(self, agent_dump: bytes, problem_name: str, theory_text: str, 
                        initial_library: List[str], statement: str, premises: List[str]) -> bool:
        """Evaluate a specific problem."""
        try:
            # Local imports to ensure these modules are available
            import torch
            import io
            import peano
            import proofsearch
            import policy
            
            # Load the agent
            with io.BytesIO(agent_dump) as f:
                agent = torch.load(f, weights_only=False)
                # Move agent to this worker's device
                agent._policy._lm._lm.to(self.device)
            
            # Create the proof state locally
            initial_state = peano.PyProofState(
                theory_text,
                initial_library + premises,
                statement
            )
            
            agent_result = agent.proof_search(statement, initial_state)
            return agent_result.success
        except Exception as e:
            print(f'Error in proof search for problem {problem_name}:', e)
            import traceback
            print(traceback.format_exc())
            return False


@ray.remote(num_gpus=1, num_cpus=1)
class ConjectureWorker:
    """Ray worker for generating conjectures."""
    
    def __init__(self, gpu_id: int = 0):
        # Import necessary modules
        import sys
        import os
        
        # Add the current directory to the path so workers can find modules
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
            
        # Import modules needed by this worker
        import torch
        import peano
        import policy
        import proofsearch
        import conjecture
        from conjecture import AgentLM, Context, sample_conjecture
        
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        print(f"Initialized ConjectureWorker with device: {self.device}")
    
    def generate_conjectures(self, agent_dump: bytes, theory_text: str, n_conjectures: int, 
                            proven_conjectures: List[str]) -> List[str]:
        """Generate conjectures using the provided agent."""
        try:
            # Local imports to ensure these modules are available
            import torch
            import io
            import peano
            import proofsearch
            import policy
            import conjecture
            from conjecture import AgentLM, Context, sample_conjecture
            
            # Load the agent
            with io.BytesIO(agent_dump) as f:
                agent = torch.load(f, weights_only=False)
                # Move agent to this worker's device
                agent._policy._lm._lm.to(self.device)
            
            # Create fresh derivation and context
            d = peano.PyDerivation()
            d.incorporate(theory_text)
            context = Context(d, None, [])
            
            conjectures = []
            attempts = 0
            max_attempts = n_conjectures * 10  # Limit the number of attempts to avoid infinite loops
            
            while len(conjectures) < n_conjectures and attempts < max_attempts:
                attempts += 1
                proposal = sample_conjecture(AgentLM(agent, 'Conj:(hard) '), context)
                
                if proposal and proposal not in conjectures + proven_conjectures:
                    # Contract conjectures to make them Peano-parseable
                    try:
                        contracted_proposal = d.contract(proposal)
                        if contracted_proposal not in conjectures + proven_conjectures:
                            conjectures.append(contracted_proposal)
                    except Exception as e:
                        print(f"Error contracting proposal: {e}")
                        continue
            
            print(f"Generated {len(conjectures)} conjectures after {attempts} attempts")
            return conjectures
        except Exception as e:
            print('Error in conjecture generation:', e)
            import traceback
            print(traceback.format_exc())
            return []


def create_placement_groups(num_gpus_available: int, num_gpus_per_worker: int = 1) -> List[PlacementGroup]:
    """Create placement groups for efficient GPU utilization."""
    num_workers = num_gpus_available // num_gpus_per_worker
    
    placement_groups = []
    for i in range(num_workers):
        bundles = [{"GPU": num_gpus_per_worker, "CPU": 1}]
        pg = placement_group(bundles)
        ray.get(pg.ready())
        placement_groups.append(pg)
    
    return placement_groups


def setup_ray(num_gpus_per_worker: int = 1) -> int:
    """Initialize Ray and return the number of workers that can be created."""
    if not ray.is_initialized():
        try:
            ray.init()
        except Exception as e:
            print(f"Error initializing Ray: {e}")
            print("Falling back to single-process mode.")
            return 1
    
    try:
        num_gpus_available = ray.cluster_resources().get("GPU", 0)
        print(f"Available GPUs: {num_gpus_available}")
        
        # Fall back to CPU if no GPUs available
        if num_gpus_available == 0:
            print("No GPUs available, will use CPU workers")
            # Use CPU workers, but limit based on available CPUs
            num_cpus = int(ray.cluster_resources().get("CPU", 2))
            return max(1, num_cpus // 2)  # Use at most half of available CPUs
            
        return max(1, int(num_gpus_available // num_gpus_per_worker))
    except Exception as e:
        print(f"Error accessing Ray cluster resources: {e}")
        print("Falling back to single-process mode.")
        return 1


def test_on_pset_parallel(agent, problemset, num_workers: int) -> float:
    """Evaluate agent on a problem set in parallel using Ray."""
    # Dump the agent to be sent to workers
    agent_buffer = io.BytesIO()
    torch.save(agent, agent_buffer)
    agent_dump = agent_buffer.getvalue()
    
    problems = problemset.problem_names()
    
    # Extract theory information instead of pickling ProofState
    theory_text = problemset._theory
    initial_library = problemset._initial_library
    
    # Create a dictionary of problem statements and premises
    problem_data = {}
    for problem in problems:
        statement = problemset._statements[problem].statement
        premises = problemset._statements[problem].premises
        problem_data[problem] = (statement, premises)
    
    # Create workers if needed
    workers = []
    for i in range(min(num_workers, len(problems))):
        # When creating remote workers, need to handle the num_gpus argument
        # based on whether GPUs are available
        gpu_available = ray.cluster_resources().get("GPU", 0) > 0
        
        if gpu_available:
            workers.append(EvaluationWorker.remote(i % num_workers))
        else:
            # Override the default GPU request with zero
            workers.append(EvaluationWorker.options(num_gpus=0).remote(0))
    
    # Submit evaluation tasks
    tasks = []
    for i, problem in enumerate(problems):
        worker_idx = i % len(workers)
        statement, premises = problem_data[problem]
        tasks.append(workers[worker_idx].evaluate_problem.remote(
            agent_dump, problem, theory_text, initial_library, statement, premises))
    
    # Collect results
    successes = ray.get(tasks)
    
    return np.mean(successes)


def generate_conjectures_parallel(agent, context, n_conjectures: int, 
                                 proven_conjectures: List[str], 
                                 num_workers: int) -> List[str]:
    """Generate conjectures in parallel using Ray."""
    # Prepare data for workers
    agent_buffer = io.BytesIO()
    torch.save(agent, agent_buffer)
    agent_dump = agent_buffer.getvalue()
    
    # Extract theory from context instead of serializing the entire context
    theory_text = ""
    if hasattr(context.derivation, '_theory_text'):
        theory_text = context.derivation._theory_text
    elif hasattr(context.derivation, 'serialize'):
        theory_text = context.derivation.serialize()
    else:
        # Fallback: Try to find the theory by context
        theory_path = os.path.join(os.path.dirname(__file__), 'theories')
        for filename in os.listdir(theory_path):
            if filename.endswith('.p'):
                with open(os.path.join(theory_path, filename), 'r') as f:
                    file_theory = f.read()
                    # Create a test derivation and see if it can be incorporated
                    test_d = peano.PyDerivation()
                    try:
                        test_d.incorporate(file_theory)
                        theory_text = file_theory
                        print(f"Using theory from {filename}")
                        break
                    except:
                        pass
    
    # Determine how many conjectures each worker should generate
    conjs_per_worker = n_conjectures // num_workers
    remainder = n_conjectures % num_workers
    
    # Create workers
    workers = []
    for i in range(num_workers):
        # When creating remote workers, need to handle the num_gpus argument
        # based on whether GPUs are available
        gpu_available = ray.cluster_resources().get("GPU", 0) > 0
        
        if gpu_available:
            workers.append(ConjectureWorker.remote(i % num_workers))
        else:
            # Override the default GPU request with zero
            workers.append(ConjectureWorker.options(num_gpus=0).remote(0))
    
    # Submit tasks
    tasks = []
    for i in range(num_workers):
        worker_conjs = conjs_per_worker + (1 if i < remainder else 0)
        if worker_conjs > 0:
            tasks.append(workers[i].generate_conjectures.remote(
                agent_dump, theory_text, worker_conjs, proven_conjectures))
    
    # Collect results
    conjecture_lists = ray.get(tasks)
    
    # Combine results
    all_conjectures = []
    for conj_list in conjecture_lists:
        all_conjectures.extend(conj_list)
    
    return all_conjectures


def prove_conjectures_parallel(agent, theory: BackgroundTheory, conjectures: List[str], 
                              num_workers: int) -> List[StudentResult]:
    """Attempt to prove conjectures in parallel using Ray."""
    # Dump agent to be sent to workers
    agent_buffer = io.BytesIO()
    torch.save(agent, agent_buffer)
    agent_dump = agent_buffer.getvalue()
    
    # Create workers
    workers = []
    for i in range(min(num_workers, len(conjectures))):
        # When creating remote workers, need to handle the num_gpus argument
        # based on whether GPUs are available
        gpu_available = ray.cluster_resources().get("GPU", 0) > 0
        
        if gpu_available:
            workers.append(ProofWorker.remote(i % num_workers))
        else:
            # Override the default GPU request with zero
            workers.append(ProofWorker.options(num_gpus=0).remote(0))
    
    # Submit tasks
    tasks = []
    for i, conjecture in enumerate(conjectures):
        worker_idx = i % len(workers)
        tasks.append(workers[worker_idx].try_prove.remote(
            agent_dump, theory, conjecture))
    
    # Collect results (with progress tracking)
    student_results = []
    for result in tqdm(ray.get(tasks), total=len(tasks), desc="Proving conjectures"):
        if not result.error:
            student_results.append(result)
        else:
            print('Error in prover process!')
            print(result.error)
    
    return student_results


def ensure_module_imports():
    """Make sure all necessary modules are imported for Ray workers"""
    # These need to be imported before Ray workers are started
    import torch
    import peano
    import sys
    import os
    
    # Add the current directory to the path so workers can find modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # Make explicit imports of local modules
    import policy
    import proofsearch
    import worker
    import conjecture
    import hindsight
    import problems
    import util

async def teacher_loop(cfg: DictConfig):
    # Make sure all modules are imported before creating workers
    ensure_module_imports()
    
    # Initialize Ray with GPU support
    num_workers = setup_ray(num_gpus_per_worker=1)
    print(f"Running in parallel mode with {num_workers} workers")

    agent = make_agent(cfg)

    with open(os.path.join(os.path.dirname(__file__), 'theories', cfg.theory.name + '.p')) as f:
        theory = f.read()

    difficulty_buckets = sorted([list(cfg.difficulty_buckets[i].items())[0]
                               for i in range(len(cfg.difficulty_buckets))],
                              key=lambda kv: kv[1])

    premises = cfg.theory.premises
    test_problemset = load_problemset(cfg.test_problems)
    d = peano.PyDerivation()
    d.incorporate(theory)
    proven_conjectures = []
    seen_hindsight_goals = set()
    proofs = []
    outcomes = []
    induced_tactics = []

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

    if cfg.get('freeze_conjecturer', False):
        print('Ablation: Freezing conjecturer.')

    with open('log.jsonl', 'w') as log:
        for i in range(start_iteration, cfg.iterations):
            # STEP 2: Evaluate agent on test problems (in parallel)
            test_start_time = time.time()
            test_success_rate = test_on_pset_parallel(agent, test_problemset, num_workers)
            test_end_time = time.time()
            
            print(f'Test success rate: {test_success_rate}, time: {test_end_time - test_start_time:.2f}s')
            log.write(json.dumps({'iteration': i,
                                'msg': f'Test success rate: {test_success_rate}'}))
            log.write('\n')
            wandb.log({'test_success_rate': test_success_rate})
            torch.save(agent, f'{i}.pt')

            # Create a fresh context for this iteration
            d = peano.PyDerivation()
            d.incorporate(theory)
            context = Context(d, None, [])
            
            # STEP 3: Generate conjectures in parallel
            print(now(), f'Iteration #{i}: making conjectures...')
            conj_start_time = time.time()
            conjectures = generate_conjectures_parallel(
                agent, context, cfg.n_conjectures, proven_conjectures, num_workers)
            
            # If we didn't get all the requested conjectures, try again in non-parallel mode as fallback
            if len(conjectures) < cfg.n_conjectures:
                print(f"Only generated {len(conjectures)} of {cfg.n_conjectures} conjectures in parallel mode. Trying sequential generation.")
                
                while len(conjectures) < cfg.n_conjectures:
                    proposal = sample_conjecture(AgentLM(agent, 'Conj:(hard) '), context)
                    
                    if proposal and proposal not in conjectures + proven_conjectures:
                        # Contract conjectures to make them Peano-parseable
                        contracted_proposal = d.contract(proposal)
                        if contracted_proposal not in conjectures + proven_conjectures:
                            conjectures.append(contracted_proposal)
                
            conj_end_time = time.time()
            
            print(now(), f'Done. Generated {len(conjectures)} conjectures, time: {conj_end_time - conj_start_time:.2f}s')
            print(conjectures)

            log.write(json.dumps({'iteration': i,
                                'msg': f'It #{i}: posing {len(conjectures)} conjectures.',
                                'conjectures': conjectures}))
            log.write('\n')
            log.flush()

            # STEP 4: Prove conjectures in parallel
            prove_start_time = time.time()
            student_results = prove_conjectures_parallel(
                agent, BackgroundTheory(theory, premises), conjectures, num_workers)
            prove_end_time = time.time()
            
            print(f'Collected {len(student_results)} results from workers, time: {prove_end_time - prove_start_time:.2f}s')

            success_logprobs = []

            # Post-processing: analyze results
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

            # STEP 5: Induce tactics from successful proofs if enabled
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
                                       for t in induced_tactics]
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

            thresholds = [np.percentile(success_logprobs, p)
                        for _, p in difficulty_buckets]

            print('Thresholds:',
                list(zip([k for k, _ in difficulty_buckets], thresholds)),
                'min =', np.min(success_logprobs),
                'max =', np.max(success_logprobs))

            # Post-processing: classify problems into easy/hard
            examples = []
            for student_result in student_results:
                # Outcome is the name of the first difficulty bucket that is larger than the logprob.
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

            # STEP 6: Train model on conjecturing and proof search examples
            if i + 1 < cfg.iterations:
                print(len(examples), 'accumulated training examples.')
                agent.train(examples)

            save_json(examples, f'examples_{i}.json')
            save_json(outcomes, f'outcomes_{i}.json')
            torch.save(student_results, f'results_{i}.json')


@hydra.main(version_base="1.2", config_path="config", config_name="bootstrap")
def main(cfg: DictConfig):
    print('Running from:', os.getcwd())
    setup_wandb(cfg)
    if cfg.task == 'teacher':
        asyncio.run(teacher_loop(cfg))

if __name__ == '__main__':
    main()