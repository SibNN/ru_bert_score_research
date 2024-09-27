import torch
import numpy as np
import math
from typing import List, Tuple, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LongformerForMaskedLM, LongformerTokenizerFast

MAX_TEXT_LEN: int = 16_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_token_embeddings(texts: List[str], embedder: Tuple[torch.nn.Module, torch.nn.Module],
                               batch_size: Optional[int] = None, hidden_state_idx: int = -1, 
                               use_global_attention: bool = False) -> List[Union[np.ndarray, None]]:
    input_ids = []
    attention_mask = []
    useful_token_indices = []

    embedder[1].to(device)

    if batch_size is None:
        tokenized = embedder[0].batch_encode_plus(texts, max_length=MAX_TEXT_LEN, return_length=True,
                                                  truncation=True, padding=True, return_special_tokens_mask=True,
                                                  return_tensors='pt')
        input_ids.append(tokenized.input_ids)
        attention_mask.append(tokenized.attention_mask)
        for sample_idx in range(tokenized.special_tokens_mask.shape[0]):
            useful_token_indices_of_cur_text = []
            for time_idx in range(tokenized.special_tokens_mask.shape[1]):
                if time_idx >= tokenized.length[sample_idx]:
                    break
                mask_val = int(tokenized.special_tokens_mask[sample_idx, time_idx])
                if mask_val not in {0, 1}:
                    raise RuntimeError(f'The mask value = {mask_val} is wrong!')
                if mask_val == 0:
                    useful_token_indices_of_cur_text.append(time_idx)
            useful_token_indices.append(useful_token_indices_of_cur_text)
            del useful_token_indices_of_cur_text
        del tokenized
    else:
        if batch_size < 1:
            raise ValueError(f'The minibatch size = {batch_size} is wrong!')
        n_batches = math.ceil(len(texts) / batch_size)
        for idx in range(n_batches):
            batch_start = idx * batch_size
            batch_end = min(len(texts), batch_start + batch_size)
            tokenized = embedder[0].batch_encode_plus(texts[batch_start:batch_end], max_length=MAX_TEXT_LEN,
                                                      return_length=True, truncation=True, padding=True,
                                                      return_special_tokens_mask=True, return_tensors='pt')
            input_ids.append(tokenized.input_ids)
            attention_mask.append(tokenized.attention_mask)
            for sample_idx in range(tokenized.special_tokens_mask.shape[0]):
                useful_token_indices_of_cur_text = []
                for time_idx in range(tokenized.special_tokens_mask.shape[1]):
                    if time_idx >= tokenized.length[sample_idx]:
                        break
                    mask_val = int(tokenized.special_tokens_mask[sample_idx, time_idx])
                    if mask_val not in {0, 1}:
                        raise RuntimeError(f'The mask value = {mask_val} is wrong!')
                    if mask_val == 0:
                        useful_token_indices_of_cur_text.append(time_idx)
                useful_token_indices.append(useful_token_indices_of_cur_text)
                del useful_token_indices_of_cur_text
            del tokenized
    text_idx = 0
    embeddings = []
    for batched_input_ids, batched_attention_mask in zip(input_ids, attention_mask):
        batched_input_ids = batched_input_ids.to(device)
        batched_attention_mask = batched_attention_mask.to(device)

        global_attention_mask = None
        if use_global_attention:
            global_attention_mask = [
                [1 if token_id == embedder[0].cls_token_id else 0 for token_id in cur_input_ids]
                for cur_input_ids in batched_input_ids
            ]
            global_attention_mask = torch.tensor(global_attention_mask).to(device)
        
        with torch.no_grad():
            if use_global_attention:
                outputs = embedder[1](input_ids=batched_input_ids, attention_mask=batched_attention_mask,
                                      global_attention_mask=global_attention_mask,
                                      return_dict=True, output_hidden_states=True)
            else:
                outputs = embedder[1](input_ids=batched_input_ids, attention_mask=batched_attention_mask,
                                      return_dict=True, output_hidden_states=True)
        
        cur_hidden_state = outputs.hidden_states[hidden_state_idx].cpu().numpy()
        
        for idx in range(cur_hidden_state.shape[0]):
            if len(useful_token_indices[text_idx]) > 0:
                emb_matrix = cur_hidden_state[idx, useful_token_indices[text_idx], :]
                embeddings.append(emb_matrix)
            else:
                embeddings.append(None)
            text_idx += 1

    return embeddings

def bert_score(references: List[str], predictions: List[str],
               evaluator: Tuple[LongformerTokenizerFast, LongformerForMaskedLM],
               batch_size: Optional[int] = None, hidden_state_idx: int = -1, 
               use_global_attention: bool = False) -> List[float]:
    if len(references) != len(predictions):
        raise ValueError(f'The reference texts do not correspond to the predicted texts! {len(references)} != {len(predictions)}')

    embeddings_of_references = calculate_token_embeddings(references, evaluator, batch_size, hidden_state_idx, use_global_attention)
    embeddings_of_predictions = calculate_token_embeddings(predictions, evaluator, batch_size, hidden_state_idx, use_global_attention)
    
    scores = []
    for ref, pred in zip(embeddings_of_references, embeddings_of_predictions):
        if ref is None or pred is None:
            scores.append(1.0 if ref is None and pred is None else 0.0)
        else:
            similarity_matrix = cosine_similarity(ref, pred)
            recall = np.mean([similarity_matrix[ref_idx, np.argmax(similarity_matrix[ref_idx, :])] for ref_idx in range(ref.shape[0])])
            precision = np.mean([similarity_matrix[np.argmax(similarity_matrix[:, pred_idx]), pred_idx] for pred_idx in range(pred.shape[0])])
            f1 = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0.0
            scores.append(f1)
    return scores
