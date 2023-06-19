import torch
from transformers import CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


def encode_text_word_embedding(text_encoder: CLIPTextModel, input_ids: torch.tensor, word_embeddings: torch.tensor,
                               num_vstar: int = 1) -> BaseModelOutputWithPooling:
    """
    Encode text by replacing the '$' with the PTEs extracted with the inversion adapter.
    Heavily based on hugginface implementation of CLIP.
    """
    existing_indexes = (input_ids == 259).nonzero(as_tuple=True)[0]  # 259 is the index of '$' in the vocabulary
    existing_indexes = existing_indexes.unique()
    if len(existing_indexes) > 0:  # if there are '$' in the text
        _, counts = torch.unique((input_ids == 259).nonzero(as_tuple=True)[0], return_counts=True)
        cum_sum = torch.cat((torch.zeros(1, device=input_ids.device).int(), torch.cumsum(counts, dim=0)[:-1]))
        first_vstar_indexes = (input_ids == 259).nonzero()[cum_sum][:,
                              1]  # get the index of the first '$' in each sentence
        rep_idx = torch.cat([(first_vstar_indexes + n).unsqueeze(0) for n in range(num_vstar)])
        word_embeddings = word_embeddings.to(input_ids.device)

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])

    seq_length = input_ids.shape[-1]
    position_ids = text_encoder.text_model.embeddings.position_ids[:, :seq_length]
    input_embeds = text_encoder.text_model.embeddings.token_embedding(input_ids)

    if len(existing_indexes) > 0:
        assert word_embeddings.shape[0] == input_embeds.shape[0]
        if len(word_embeddings.shape) == 2:
            word_embeddings = word_embeddings.unsqueeze(1)
        input_embeds[torch.arange(input_embeds.shape[0]).repeat_interleave(
            num_vstar).reshape(input_embeds.shape[0], num_vstar)[existing_indexes.cpu()], rep_idx.T] = \
            word_embeddings.to(input_embeds.dtype)[existing_indexes]  # replace the '$' with the PTEs

    position_embeddings = text_encoder.text_model.embeddings.position_embedding(position_ids)
    hidden_states = input_embeds + position_embeddings

    bsz, seq_len = input_shape
    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )

    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=None,
        causal_attention_mask=causal_attention_mask,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_encoder.text_model.final_layer_norm(last_hidden_state)

    # text_embeds.shape = [batch_size, sequence_length, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
    ]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )
