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
from tactics import induce_tactics_from_proofs
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


def _prove(agent_dump: bytes, theory: worker.BackgroundTheory, statement: str, is_eval: bool = False):
    with io.BytesIO(agent_dump) as f:
        agent = torch.load(f, weights_only=False)

    print('Proving', statement, 'on', agent._policy._lm._lm.device)

    state = peano.PyProofState(theory.theory,
                               theory.premises,
                               statement)
    
    try:
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
        print('Error in try_prove!')
        print(tb)
        return StudentResult(tb, False, statement, None, None, [],
                             [], None, None)


def load_problems(problems_path: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    problems_path = os.path.join(current_dir, problems_path)
    with open(problems_path, 'r') as f:
        problems_text = f.readlines()
    problems = [p.split(". ")[1] for p in problems_text if ". " in p]
    return problems


def test_on_pset(agent, theory: worker.BackgroundTheory, test_problems_path: str, num_workers: int = 4):
    """
    Tests the agent on a given problemset in parallel.

    Args:
        agent: The agent object to test.
        problemset_cfg: The configuration DictConfig for loading the problemset.
        num_workers: Number of parallel processes to use.
    """
    # Load problemset once to get names (could also pass names directly)
    problems = load_problems(test_problems_path)
    if not problems:
        return 0.0

    # Serialize the agent's state
    buff = io.BytesIO()
    # Note: Saving the whole agent might be large. If possible, saving only
    # the model state_dict() and relevant config, then reconstructing
    # the agent in the worker might be more efficient if the agent object
    # itself is complex but the core model is standard.
    # For now, we save the whole agent as done in the main loop.
    torch.save(agent, buff)
    agent_dump = buff.getvalue()

    tasks = [(agent_dump, theory, problem) for problem in problems]

    successes = {}
    # Use context manager for the pool
    # Limit workers if fewer problems than workers
    actual_workers = min(num_workers, len(problems), os.cpu_count())
    print(f"Evaluating {len(problems)} problems using {actual_workers} workers...")

    with mp.Pool(processes=actual_workers) as pool:
        # Use starmap to pass arguments tuple
        results = pool.starmap(_prove, tasks)

    # Process results
    for result in results:
        successes[result.problem] = result.success

    # Calculate success rate (ensure order doesn't matter)
    num_successful = sum(successes.values())
    total_problems = len(problems)

    print(f"Evaluation complete: {num_successful}/{total_problems} successful.")

    return num_successful / total_problems if total_problems > 0 else 0.0


def teacher_loop(cfg: DictConfig):
    agent = make_agent(cfg)

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
            # Pass the problemset config and desired number of workers
            num_eval_workers = cfg.get('num_eval_workers', 4) # Default to 4 workers if not specified
            test_success_rate = test_on_pset(agent, worker.BackgroundTheory(theory, premises),
                                              cfg.test_problems_path, num_workers=num_eval_workers)
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


            print(now(), 'done, have', len(conjectures), 'conjectures')
            print(conjectures)

            log.write(json.dumps({'iteration': i,
                                  'msg': f'It #{i}: posing {len(conjectures)} conjectures.',
                                  'conjectures': conjectures}))
            log.write('\n')
            log.flush()

            # 2- Try to prove each of the conjectures
            tasks = []

            # Dump current agent.
            buff = io.BytesIO()
            torch.save(agent, buff)
            agent_dump = buff.getvalue()


            tasks = [(agent_dump, worker.BackgroundTheory(theory, premises), conjecture) for conjecture in conjectures]

            # Use context manager for the pool
            # Limit workers if fewer problems than workers
            num_workers = cfg.get('num_workers', 4)
            actual_workers = min(num_workers, len(conjectures), os.cpu_count())
            print(f"Proving {len(conjectures)} conjectures using {actual_workers} workers...")

            with mp.Pool(processes=actual_workers) as pool:
                # Use starmap to pass arguments tuple
                results = pool.starmap(_prove, tasks)

            student_results = results
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

            save_json(examples, f'examples_{i}.json')
            save_json(outcomes, f'outcomes_{i}.json')
            torch.save(student_results, f'results_{i}.json')


@hydra.main(version_base="1.2", config_path="config", config_name="bootstrap")
def main(cfg: DictConfig):
    print('Running from:', os.getcwd())
    set_seed(cfg.seed)
    setup_wandb(cfg)
    if cfg.task == 'teacher':
        teacher_loop(cfg)

if __name__ == '__main__':
    main()
