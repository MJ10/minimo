import os
import json
import hydra
import torch
import peano
import transformers
import io
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import worker
from proofsearch import TreeSearchNode, make_agent

from util import (
    log, softmax, pop_max, batch_strings, encode_batch, decode_batch,
    sample_batch, PAD, EOS, BOS, POSITIVE, NEGATIVE, EMPTY,
    batch_inference
)

def create_attention_mask(sequence_lengths):
    # Initialize the attention mask with zeros
    max_length = max(sequence_lengths)
    attention_mask = torch.zeros((len(sequence_lengths), max_length), dtype=torch.int)
    
    for i, length in enumerate(sequence_lengths):
        # Set the first 'length' positions to 1
        attention_mask[i, :length] = 1
    
    return attention_mask


def recompute_logprobs(agent, examples, theory):
    for example in examples:
        state = peano.PyProofState(theory.theory,
                               theory.premises,
                               example["problem"])
        
        # agent_result = agent.proof_search(example["problem"], state)
        
        
        root = TreeSearchNode(agent._node_type([state]))
        
        
        
        
        action_strs = example["actions"]
        actions = []
        current = root

        for action in action_strs:
            current.expand()
            for i, ac in enumerate(current.actions):
                if str(ac) == action:
                    actions.append(ac)
                    current = current.children()[i]
                    break
        import pdb; pdb.set_trace();
        
        logprob = root.solution_logprob_under_policy(agent._policy, actions)


class LMHardnessPredictor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        if torch.cuda.is_available():
            cfg = transformers.GPT2Config(
                vocab_size=128,
                n_layer=config.get('n_layer', 8),
                n_head=config.get('n_head', 8),
                n_embd=config.get('n_embd', 512),
                bos_token_id=BOS,
                eos_token_id=EOS,
                pad_token_id=PAD,
                n_positions=1024)
            device = torch.device(0)
        else:
            # Debugging on a CPU
            cfg = transformers.GPT2Config(
                vocab_size=128,
                n_layer=2,
                n_head=2,
                n_embd=128,
                bos_token_id=BOS,
                eos_token_id=EOS,
                pad_token_id=PAD,
                n_positions=512)
            device = torch.device('cpu')
        self._lm = transformers.GPT2Model(cfg).to(device)
        self.output = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_embd),
            nn.ReLU(),
            nn.Linear(cfg.n_embd, 1),
        ).to(device)

    def load_lm(self, agent):
        self._lm = agent._lm

    def forward(self, x, lengths=None):
        attention_mask = create_attention_mask(lengths).to(self._lm.device) if lengths is not None else None
        out = self._lm(x, attention_mask=attention_mask).last_hidden_state[:, -1:, :].squeeze()
        return self.output(out)

    def _strs_to_token_ids(self, strs: list[str], eos=False) -> torch.tensor:
        # strs = [s.replace(' ', '').replace('\n', '') for s in strs[:]]
        # Trim strings if too long.
        for i in range(len(strs)):
            if len(strs[i]) > 490:
                strs[i] = '[...] ' + strs[i][-490:]

        ids = [[BOS] + list(s.encode('ascii')) + eos*[EOS]
               for s in strs]

        lengths = list(map(len, ids))
        max_length = max(lengths)
        ids = [l + (max_length - len(l)) * [PAD] for l in ids]

        assert 0 <= np.min(ids) and np.max(ids) < 128
        return lengths, torch.tensor(ids, device=self._lm.device)


@hydra.main(version_base="1.2", config_path="config", config_name="hardness_predictor")
def main(cfg):
    agent = make_agent(cfg)
    with open(os.path.join(os.path.dirname(__file__), 'theories', cfg.theory.name + '.p')) as f:
        theory = f.read()

    premises = cfg.theory.premises

    d = peano.PyDerivation()
    d.incorporate(theory)
    proven_conjectures = []
    seen_hindsight_goals = set()
    proofs = []
    outcomes = []
    
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
            with open(f'outcomes_{i}.json', 'r') as f:
                outcomes = json.load(f)
                print('Loaded outcomes from', f'outcomes_{i}.json')

        examples = [(o["problem"], o["logprob"]) for o in outcomes if o["logprob"] is not None]
        successful_outcomes = [o for o in outcomes if o["logprob"] is not None]
    import pdb; pdb.set_trace();
    recompute_logprobs(agent, successful_outcomes, worker.BackgroundTheory(theory, premises))
    predictor = LMHardnessPredictor(cfg)
    # if agent is not None:
    #     predictor.load_lm(agent)
    #     print("Initialized predictor with agent's LM")

    optimizer = torch.optim.Adam(predictor.parameters(), lr=0.0001)
    for i in range(1000):
        # sample a random batch from `examples`
        idxs = np.random.choice(len(examples), 64, replace=False)
        batch_x = [examples[j][0] for j in idxs]
        batch_y = [examples[j][1] for j in idxs]

        # encode the batch
        lengths, x = predictor._strs_to_token_ids(batch_x)
        y = torch.tensor(batch_y, device=predictor._lm.device)

        # forward pass
        logits = predictor(x, lengths)
        loss = torch.nn.functional.mse_loss(logits.squeeze(), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(loss.item())

    import pdb; pdb.set_trace();
    

if __name__ == '__main__':
    main()