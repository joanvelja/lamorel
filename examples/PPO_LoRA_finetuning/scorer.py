import torch
from lamorel import BaseModuleFunction


class LogScoringModuleFn(BaseModuleFunction):
    """
    Module function to compute log probabilities of sequences.

    Args:
        model_type (str): Type of model (causal or non-causal).
        pre_encoded_input (bool): Whether input is pre-encoded.

    Returns:
        torch.Tensor: Log probabilities of sequences.

    Examples:

        >>> forward_outputs = {
        ...     "logits": torch.randn(2, 5, 50264)
        ... }
        >>> minibatch = {
        ...     "input_ids": torch.randint(0, 50264, (2, 5))
        ... }
        >>> tokenized_contexts = [
        ...     {"input_ids": torch.randint(0, 50264, (2, 5))}
        ... ]
        >>> module_fn = LogScoringModuleFn("causal", False)
        >>> module_fn(forward_outputs, minibatch, tokenized_contexts)
        tensor([0.0, 0.0])
    """

    def __init__(self, model_type, pre_encoded_input):
        super().__init__()
        self._model_type = model_type
        self._pad_token = 0
        self._pre_encoded_input = pre_encoded_input

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        if self._model_type == "causal":  # causal model, i.e., GPT (auto-regressive)
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"]
                )  # inputs are padded so all of same size (skip the prompt tokens / context)

            logits = forward_outputs[
                "logits"
            ][
                :, end_of_context_position:-1, :
            ]  # LOGITS : Start from end of context and skip EoS token appended by tokenizer (logits are lookahead so start from end of context to end of sequence - 1)
            output_tokens = minibatch[
                "input_ids"
            ][
                :, end_of_context_position + 1 :
            ]  # TOKENS : Skip the prompt tokens / context and start from the first token of the sequence
        else:  # non-causal model, i.e., T5
            logits = forward_outputs["logits"][
                :, :-1, :
            ]  # skip </s> token appended by tokenizer
            output_tokens = minibatch["decoder_input_ids"][:, 1:]  # skip pad token

        tokens_logprobs = (
            torch.gather(logits, 2, output_tokens[:, :, None])
            .squeeze(-1)
            .to(torch.float32)
        )  # filter with sequence tokens

        # Compute mask to assign probability 1 to padding tokens
        mask = torch.ones(tokens_logprobs.shape, dtype=torch.bool, device=self.device)
        for i, _output in enumerate(output_tokens):
            for j, _token in enumerate(_output):
                if _token != self._pad_token:
                    mask[i, j] = False
        masked_token_probs = tokens_logprobs.masked_fill(mask, 0.0)  # apply mask
        minibatch_probs = masked_token_probs.sum(
            -1
        )  # compute final sequences' probability

        return minibatch_probs.cpu()
