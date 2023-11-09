from typing import Optional, Union
from .. import LanguageModel,GenerateOutput
import numpy as np



import torch
from typing import Union, List, Any


class FakeLLM(LanguageModel):
    """
    A fake language model for demonstration purposes.
    """

    def generate(self, inputs: List[str], **kwargs: Any) -> GenerateOutput:
        """
        Generate output based on the given inputs.
        """
        out = 'RETRIEVE'
        return GenerateOutput([out] * len(inputs), [np.array([0.0])] * len(inputs))
    
    @torch.no_grad()
    def get_next_token_logits(
        self,
        prompt: Union[str, List[str]],
        candidates: Union[List[str], List[List[str]]]) -> List[np.ndarray]:
        """
        Get the next token logits for the given prompt and candidates.
        """
        if not isinstance(prompt, (str, list)):
            raise TypeError('Prompt must be a string or a list of strings.')
        
        if isinstance(prompt, str):
            prompt = [prompt]
        
        return [np.random.random_sample((2,)) * len(prompt)]

    def get_loglikelihood(self,
                    prompt: Union[str, list[str]],
                    **kwargs) -> list[np.ndarray]:
        
        raise NotImplementedError("GPTCompletionModel does not support get_log_prob")