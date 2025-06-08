#!/usr/bin/env python3

"""Implements the conjecture-prove bootstrapping learning loop."""

import asyncio
import os
import io
import json
import datetime
import traceback
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
import gc  # For explicit garbage collection
import time
import signal
from functools import partial
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError

import peano
import worker
from worker import StudentResult  # noqa
from hindsight import HindsightExample, extract_hindsight_examples  # noqa
from util import format_blocks_with_indent, sample_batch, setup_wandb, value_color, save_json
from conjecture import AgentLM, Context, sample_conjecture
from proofsearch import make_agent
from problems import load_problemset
import wandb
from dataclasses import dataclass
from typing import Optional, List, Tuple
from tactic import induce_tactics_from_proofs, rewrite_solutions
import random

def now() -> str:
    return '[' + datetime.datetime.now().isoformat() + ']'


FAIL = "fail"


try:
    mp.set_start_method('spawn', force=True)
    print("Spawn start method set for multiprocessing.")
except RuntimeError:
    print("Spawn start method already set or failed to set.") # Handle cases where it might already be set


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_WORKER_GPU_COUNT = None  # Populated in initializer
_WORKER_ID = None  # Track worker ID for debugging


def _init_worker(num_gpus: int, workers_per_gpu: int):
    """Set the proper CUDA device for each worker process.
    
    Multiple workers can share the same GPU - PyTorch/CUDA handles this well.
    We distribute workers evenly across available GPUs.
    """
    global _WORKER_GPU_COUNT, _WORKER_ID
    _WORKER_GPU_COUNT = num_gpus

    # Get worker ID for debugging
    worker_info = mp.current_process()._identity
    _WORKER_ID = worker_info[0] if worker_info else 0

    if num_gpus == 0 or not torch.cuda.is_available():
        print(f"Worker {_WORKER_ID}: Running on CPU")
        return  # CPU-only

    # Distribute workers across GPUs
    # For example, with 4 GPUs and 8 workers (2 per GPU):
    # Workers 1,2 -> GPU 0
    # Workers 3,4 -> GPU 1
    # Workers 5,6 -> GPU 2
    # Workers 7,8 -> GPU 3
    worker_idx = _WORKER_ID - 1  # Convert to 0-based
    gpu_id = worker_idx // workers_per_gpu
    
    # Handle case where we have more workers than evenly distributable
    if gpu_id >= num_gpus:
        gpu_id = worker_idx % num_gpus
    
    # Set CUDA device for this worker
    torch.cuda.set_device(gpu_id)
    
    # Enable CUDA memory allocation strategies for better multi-process behavior
    # This allows better memory sharing between processes on the same GPU
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Set CUDA to not reserve all memory upfront
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        # Reserve only a fraction of GPU memory per process
        # This allows multiple processes to share the GPU
        memory_fraction = 0.9 / workers_per_gpu  # Leave 10% free
        torch.cuda.set_per_process_memory_fraction(memory_fraction, gpu_id)
    
    print(f"Worker {_WORKER_ID}: Assigned to GPU {gpu_id} (sharing with {workers_per_gpu} workers)")




def _prove(agent_dump: bytes, theory: worker.BackgroundTheory, statement: str, is_eval: bool = False):
    """Run proof search on the *current* CUDA device assigned to this worker."""
    
    # Clear any existing CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    current_device = torch.device(
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )

    agent = None
    try:
        # Deserialize agent onto the current device only.
        with io.BytesIO(agent_dump) as f:
            agent = torch.load(f, map_location=current_device, weights_only=False)

        print(f'Worker {_WORKER_ID}: Proving {statement} on {current_device}')

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

        if is_eval:
            examples, hindsight_examples = [], []
        else:
            examples = []
            # Policy examples for the proved goal.
            examples.extend(agent._policy.extract_examples(root=agent_result.root))
            # Hindsight examples (policy + conjecturing).
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
            examples,
            hindsight_examples,
            agent_result.iterations,
            logprob,
        )
    except BaseException as e:
        tb = traceback.format_exception(e)
        print(f'Worker {_WORKER_ID}: Error in _prove!')
        print(''.join(tb))
        return StudentResult(tb, False, statement, None, None, [],
                             [], None, None)
    finally:
        # Aggressive cleanup
        if 'agent' in locals() and agent is not None:
            del agent
        if 'agent_result' in locals():
            del agent_result
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _prove_with_timeout(tasks, num_workers, timeout_per_problem, num_gpus, num_workers_per_gpu):
    """Execute proof tasks with proper timeout using ProcessPoolExecutor."""
    results = []
    
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(num_gpus, num_workers_per_gpu),
    ) as executor:
        # Submit all tasks
        futures = []
        for task in tasks:
            future = executor.submit(_prove, *task)
            futures.append(future)
        
        # Collect results with individual timeouts
        for i, future in enumerate(tqdm(futures, desc="Processing proofs")):
            try:
                result = future.result(timeout=timeout_per_problem)
                results.append(result)
            except FutureTimeoutError:
                statement = tasks[i][2] if len(tasks[i]) > 2 else "unknown"
                print(f"Timeout proving {statement} after {timeout_per_problem}s")
                timeout_result = StudentResult(
                    ["Timeout"], False, statement, None, None, [], [], None, None
                )
                results.append(timeout_result)
            except Exception as e:
                statement = tasks[i][2] if len(tasks[i]) > 2 else "unknown"
                print(f"Error proving {statement}: {e}")
                error_result = StudentResult(
                    [str(e)], False, statement, None, None, [], [], None, None
                )
                results.append(error_result)
    
    return results


def load_problems(problems_path: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    problems_path = os.path.join(current_dir, problems_path)
    with open(problems_path, 'r') as f:
        problems_text = f.readlines()
    problems = [p.split(". ")[1] for p in problems_text if ". " in p]
    return problems


def test_on_pset(
    agent,
    theory: worker.BackgroundTheory,
    test_problems_path: str,
    num_gpus: int = 1,
    num_workers_per_gpu: int = 1,
    timeout_per_problem: int = 300,
):
    """
    Tests the agent on a given problemset in parallel.

    Args:
        agent: The agent object to test.
        theory: The background theory.
        test_problems_path: Path to test problems.
        num_gpus: Number of GPUs to use.
        num_workers_per_gpu: Number of workers per GPU.
        timeout_per_problem: Timeout in seconds for each problem.
    """
    # Load problemset once to get names (could also pass names directly)
    problems = load_problems(test_problems_path)
    if not problems:
        return 0.0

    # Serialize the agent's state once for all workers.
    buff = io.BytesIO()
    torch.save(agent, buff)
    agent_dump = buff.getvalue()

    # Prepare tasks with timeout wrapper
    tasks = [(agent_dump, theory, problem, True) for problem in problems]

    successes = {}

    total_requested_workers = (num_gpus or 1) * num_workers_per_gpu
    actual_workers = min(total_requested_workers, len(problems), os.cpu_count())
    print(
        f"Evaluating {len(problems)} problems using {actual_workers} workers across {num_gpus} GPU(s)."
    )

    # Use ProcessPoolExecutor with proper timeout
    results = _prove_with_timeout(tasks, actual_workers, timeout_per_problem, num_gpus, num_workers_per_gpu)

    # Process results
    for result in results:
        successes[result.problem] = result.success

    # Calculate success rate (ensure order doesn't matter)
    num_successful = sum(successes.values())
    total_problems = len(problems)

    print(f"Evaluation complete: {num_successful}/{total_problems} successful.")

    # Force cleanup
    del agent_dump
    gc.collect()

    return num_successful / total_problems if total_problems > 0 else 0.0


def teacher_loop(cfg: DictConfig):
    # Delay agent creation to avoid CUDA initialization in parent
    print("Initializing teacher loop...")
    
    # First, set up non-CUDA resources
    with open(os.path.join(os.path.dirname(__file__), 'theories', cfg.theory.name + '.p')) as f:
        theory = f.read()

    difficulty_buckets = sorted([list(cfg.difficulty_buckets[i].items())[0]
                                 for i in range(len(cfg.difficulty_buckets))],
                                key=lambda kv: kv[1])

    premises = cfg.theory.premises
    d = peano.PyDerivation()
    d.incorporate(theory)
    proven_conjectures = []
    seen_hindsight_goals = set()
    proofs = []
    outcomes = []
    induced_tactics = []

    continue_dir = cfg.get('continue')
    start_iteration = 0

    # Now create the agent (this might initialize CUDA)
    if continue_dir is not None:
        os.chdir(continue_dir)
        print('Continuing run from', continue_dir)
        # Find largest iteration number such that i.pt exists.
        i = 0
        while os.path.exists(f'{i}.pt'):
            i += 1
        i -= 1
        start_iteration = i
        
        # Load agent with CPU first to avoid CUDA issues
        agent = torch.load(f'{i}.pt', map_location='cpu')
        if torch.cuda.is_available():
            agent = agent.cuda()
        
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
    else:
        # Create agent fresh
        agent = make_agent(cfg)


    if cfg.get('freeze_conjecturer', False):
        print('Ablation: Freezing conjecturer.')

    # Get timeout config
    timeout_per_problem = cfg.get('timeout_per_problem', 300)

    with open('log.jsonl', 'w') as log:
        for i in range(start_iteration, cfg.iterations):
            # ---------------- Evaluation phase ----------------
            num_gpus = min(cfg.get('num_gpus', torch.cuda.device_count()), torch.cuda.device_count())
            num_workers_per_gpu = cfg.get('num_workers_per_gpu', cfg.get('num_workers', 1))

            test_success_rate = test_on_pset(
                agent,
                worker.BackgroundTheory(theory, premises),
                cfg.test_problems_path,
                num_gpus=num_gpus,
                num_workers_per_gpu=num_workers_per_gpu,
                timeout_per_problem=timeout_per_problem,
            )
            print('Test success rate:', test_success_rate)
            log.write(json.dumps({'iteration': i,
                                  'msg': f'Test success rate: {test_success_rate}'}))
            log.write('\n')
            wandb.log({'test_success_rate': test_success_rate})
            torch.save(agent, f'{i}.pt')

            context = Context(d, None, [])
            # 1- Run conjecturing model to obtain N conjectures.
            print(now(), f'Iteration #{i}: making conjectures...')

            progress_bar = tqdm(total=cfg.n_conjectures)

            conjectures = []

            while len(conjectures) < cfg.n_conjectures:
                proposal = sample_conjecture(AgentLM(agent, 'Conj:(hard) '), context)

                if proposal and proposal not in conjectures + proven_conjectures:
                    # Contract conjectures to make them Peano-parseable.
                    contracted_proposal = d.contract(proposal)
                    if contracted_proposal not in conjectures + proven_conjectures:
                        conjectures.append(contracted_proposal)
                        progress_bar.update(1)

            progress_bar.close()
            train_conjectures = conjectures[:cfg.n_conjectures // 2]
            test_conjectures = conjectures[cfg.n_conjectures // 2:]

            print(now(), 'done, have', len(conjectures), 'conjectures')
            print(conjectures)

            log.write(json.dumps({'iteration': i,
                                  'msg': f'It #{i}: posing {len(conjectures)} conjectures.',
                                  'train_conjectures': train_conjectures,
                                  'test_conjectures': test_conjectures}))
            log.write('\n')
            log.flush()

            # 2- Try to prove each of the conjectures
            tasks = []

            # Dump current agent.
            buff = io.BytesIO()
            torch.save(agent, buff)
            agent_dump = buff.getvalue()


            tasks = [(agent_dump, worker.BackgroundTheory(theory, premises), conjecture) for conjecture in train_conjectures]

            # ---------------- Proof search with consolidated pool ----------------
            num_gpus = min(cfg.get('num_gpus', torch.cuda.device_count()), torch.cuda.device_count())
            num_workers_per_gpu = cfg.get('num_workers_per_gpu', cfg.get('num_workers', 1))
            total_requested_workers = (num_gpus or 1) * num_workers_per_gpu
            max_tasks = max(len(train_conjectures), len(test_conjectures))
            actual_workers = min(total_requested_workers, max_tasks, os.cpu_count())

            print(
                f"Using consolidated pool with {actual_workers} workers across {num_gpus} GPU(s)."
            )

            # Proof search for training conjectures with timeout
            print(f"Proving {len(train_conjectures)} training conjectures...")
            results = _prove_with_timeout(tasks, actual_workers, timeout_per_problem, num_gpus, num_workers_per_gpu)
            
            # Proof search for test conjectures (before tactics)
            test_tasks = [(agent_dump, worker.BackgroundTheory(theory, premises), conjecture, False) for conjecture in test_conjectures]
            print(f"Proving {len(test_conjectures)} test conjectures...")
            test_results = _prove_with_timeout(test_tasks, actual_workers, timeout_per_problem, num_gpus, num_workers_per_gpu)

            # Force cleanup after pool closes automatically
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            student_results = results
            test_results = test_results
            examples = []

            success_logprobs = []

            # 3a- Look at all the success logprobs and compute the easy/hard threhsold.
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

            # 3b- Induce tactics from successful proofs if enabled
            # if cfg.get('induce_tactics', False):
            new_tactics = induce_tactics_from_proofs(
                max_tactics=cfg.get('max_tactics', 5),
                student_results=student_results, 
                min_occurrences=cfg.get('min_tactic_occurrences', 2),
                existing_tactics=len(induced_tactics)
            )
            
            if new_tactics:
                induced_tactics.extend(new_tactics)
                print(f"Induced {len(new_tactics)} new tactics, total: {len(induced_tactics)}")
                
                # --- rewrite previously-found solutions so that policy
                # training uses compressed traces.
                student_results_rewritten = rewrite_solutions(student_results, induced_tactics)
                
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
                    agent.set_tactics(induced_tactics)
                    print(f"Updated agent with {len(induced_tactics)} tactics")

            train_lens_without_tactics = [len(student_result.solution_actions) for student_result in student_results if student_result.success]
            train_lens_with_tactics = [len(student_result.solution_actions) for student_result in student_results_rewritten if student_result.success]

            print(f"Average length without tactics: {np.mean(train_lens_without_tactics)}")
            print(f"Average length with tactics: {np.mean(train_lens_with_tactics)}")


            thresholds = [np.percentile(success_logprobs, p)
                          for _, p in difficulty_buckets]

            print('Thresholds:',
                  list(zip([k for k, _ in difficulty_buckets], thresholds)),
                  'min =', np.min(success_logprobs),
                  'max =', np.max(success_logprobs))

            # 3c- Classify problems into easy/hard.
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

            # 3d- Train model on conjecturing and proof search examples.
            if i + 1 < cfg.iterations:
                print(len(examples), 'accumulated training examples.')
                agent.train(examples)

            # Re-serialize the agent after it has been updated with tactics
            if induced_tactics and cfg.get('use_induced_tactics', True):
                print("Re-serializing agent with tactics for test evaluation...")
                buff = io.BytesIO()
                torch.save(agent, buff)
                agent_dump_with_tactics = buff.getvalue()
            else:
                agent_dump_with_tactics = agent_dump

            # After inducing tactics, re-run proof search on test conjectures
            tasks_after_tactics = [
                (agent_dump_with_tactics, worker.BackgroundTheory(theory, premises), conjecture)
                for conjecture in test_conjectures
            ]

            print(
                f"Proving {len(test_conjectures)} conjectures after tactics using {actual_workers} workers across {num_gpus} GPU(s)."
            )

            # Proof search for test conjectures after tactics with timeout
            test_results_after_tactics = _prove_with_timeout(tasks_after_tactics, actual_workers, timeout_per_problem, num_gpus, num_workers_per_gpu)

            # Force cleanup after pool closes automatically
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            rewritten_test_results = rewrite_solutions(test_results, induced_tactics)

            test_len_without_tactics = [len(student_result.solution_actions) for student_result in test_results if student_result.success]
            test_len_with_tactics = [len(student_result.solution_actions) for student_result in test_results_after_tactics if student_result.success]
            rewritten_test_len_with_tactics = [len(student_result.solution_actions) for student_result in rewritten_test_results if student_result.success]

            print(f"Test Average length without tactics: {np.mean(test_len_without_tactics)}")
            print(f"Test Average length with tactics: {np.mean(test_len_with_tactics)}")
            print(f"Test Average length with tactics after rewriting: {np.mean(rewritten_test_len_with_tactics)}")
            wandb.log({'test_len_without_tactics': np.mean(test_len_without_tactics),
                       'test_len_with_tactics': np.mean(test_len_with_tactics),
                       'test_len_with_tactics_after_rewriting': np.mean(rewritten_test_len_with_tactics),
                       "train_len_without_tactics": np.mean(train_lens_without_tactics),
                       "train_len_with_tactics": np.mean(train_lens_with_tactics)})
            
            save_json(examples, f'examples_{i}.json')
            save_json(outcomes, f'outcomes_{i}.json')
            torch.save(student_results, f'results_{i}.json')
            torch.save(test_results, f'test_results_{i}.json')
            torch.save(test_results_after_tactics, f'test_results_after_tactics_{i}.json')

@hydra.main(version_base="1.2", config_path="config", config_name="bootstrap")
def main(cfg: DictConfig):
    print('Running from:', os.getcwd())
    set_seed(cfg.seed)
    setup_wandb(cfg)
    if cfg.task == 'teacher':
        teacher_loop(cfg)

if __name__ == '__main__':
    main()