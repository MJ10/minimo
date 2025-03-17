#!/usr/bin/env python3

"""Implements the conjecture-prove bootstrapping learning loop."""

import asyncio
import os
import io
import json
import datetime

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from tqdm import tqdm

import peano
import worker
from worker import StudentResult  # noqa
from hindsight import HindsightExample  # noqa
from util import format_blocks_with_indent, sample_batch, setup_wandb, value_color, save_json
from conjecture import AgentLM, Context, sample_conjecture
from proofsearch import make_agent
from problems import load_problemset
import wandb
from dataclasses import dataclass
from typing import Optional, List, Tuple

def now() -> str:
    return '[' + datetime.datetime.now().isoformat() + ']'


FAIL = "fail"


DISTRIBUTED = os.environ.get('DISTRIBUTED', False)


def submit_task(agent_dump: bytes, theory: worker.BackgroundTheory, statement: str):
    if DISTRIBUTED:
        return worker.try_prove.apply_async((agent_dump, theory, statement))
    else:
        return worker.try_prove.run(agent_dump, theory, statement)


def get_task_result(task):
    if DISTRIBUTED:
        return task.get()
    else:
        return task


def test_on_pset(agent, problemset, premises):
    problems = problemset.problem_names()
    successes = []
    for problem in problems:
        initial_state = problemset.initialize_problem(problem)
        try:
            agent_result = agent.proof_search(initial_state.goal(), initial_state)
        except:
            print('Error in proof search!', problem)
            successes.append(False)
            continue
        # print(f'{problem}:', agent_result.success)
        successes.append(agent_result.success)
    return np.mean(successes)

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
                # Parse arguments - this is a simplification, might need adjustment
                arguments.append([arg_str])
            else:
                arguments.append(None)
        
        # Generate tactics from different slices of the proof
        for start in range(len(actions) - 1):
            for length in range(2, min(len(actions) - start + 1, 5)):  # Limit length to avoid too complex tactics
                tactic_name = f't_{i}_{start}_{length}'
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
        signature = tuple((step.arrows, len(step.arguments)) for step in t.steps)
        
        if signature in tactic_counts:
            tactic_counts[signature][1] += 1
        else:
            tactic_counts[signature] = (t, 1)
    
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
    
    return [t for t, _ in selected_tactics]

async def teacher_loop(cfg: DictConfig):
    print('Running in', 'distributed mode.' if DISTRIBUTED else 'single-process mode.')

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
            test_success_rate = test_on_pset(agent, test_problemset, premises)
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

            print('Submitting tasks...')
            for conjecture in tqdm(conjectures, miniters=1):
                tasks.append(submit_task(
                    agent_dump,
                    worker.BackgroundTheory(theory, premises),
                    conjecture))

            # 3- Train model on proofs and outcome of conjectures (easy, hard, timeout)
            examples = []
            student_results = []

            print('Collecting', len(tasks), 'results from workers.')

            for task in tqdm(tasks, miniters=1):
                student_result = get_task_result(task)

                if student_result.error:
                    print('Error in prover process!')
                    print(student_result.error)
                    continue

                student_results.append(student_result)

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
                                       for t in induced_tactics]
                        json.dump(tactics_data, f, indent=2)
                    
                    log.write(json.dumps({'iteration': i,
                                         'msg': f'Induced {len(new_tactics)} new tactics, total: {len(induced_tactics)}'}))
                    log.write('\n')

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
    setup_wandb(cfg)
    if cfg.task == 'teacher':
        asyncio.run(teacher_loop(cfg))

if __name__ == '__main__':
    main()
