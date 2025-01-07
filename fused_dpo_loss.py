import torch
from cut_cross_entropy import linear_cross_entropy

@torch.compile(fullgraph=True)
def compute_loss(loss, chosen_cum_lengths, rejected_cum_lengths, prompt_lengths, ref_logprobs, beta):
    cum_loss = torch.cumsum(loss, dim=0)
    cum_loss = torch.cat([
        torch.zeros(1, dtype=cum_loss.dtype, device=cum_loss.device),
        cum_loss
    ])

    start_chosen = chosen_cum_lengths[:-1] + prompt_lengths
    end_chosen = chosen_cum_lengths[1:]
    chosen_logprobs = -(cum_loss[end_chosen] - cum_loss[start_chosen])

    start_rejected = rejected_cum_lengths[:-1] + prompt_lengths
    end_rejected = rejected_cum_lengths[1:]
    rejected_logprobs = -(cum_loss[end_rejected] - cum_loss[start_rejected])

    policy_logprobs = torch.stack([chosen_logprobs, rejected_logprobs], dim=1)
    rewards = policy_logprobs - ref_logprobs
    chosen_rewards = rewards[:, 0]
    rejected_rewards = rewards[:, 1]
    dpo_loss = -torch.nn.functional.logsigmoid((chosen_rewards - rejected_rewards) * beta).mean()
    return dpo_loss

embeddings = torch.randn(40000, 4096, device="cuda", dtype=torch.bfloat16)
classifier = torch.randn(128256, 4096, device="cuda", dtype=torch.bfloat16)
labels = torch.randint(0, 10, (40000,), device="cuda")
prompt_lengths = torch.tensor([1000, 2000, 3000, 4000], device="cuda")
chosen_cum_lengths = torch.tensor([0, 5000, 10000, 15000, 20000], device="cuda")
rejected_cum_lengths = torch.tensor([20000, 25000, 30000, 35000, 40000], device="cuda")
ref_logprobs = torch.randn(4, 2, device="cuda", dtype=torch.bfloat16)
beta = 0.1
loss = linear_cross_entropy(embeddings, classifier, labels, reduction="none")
dpo_loss = compute_loss(loss, chosen_cum_lengths, rejected_cum_lengths, prompt_lengths, ref_logprobs, beta)
print(dpo_loss)
