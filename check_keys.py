from safetensors import safe_open

with safe_open('exported-fastvlm-0.5b/model.safetensors', framework='pt') as st:
    keys = list(st.keys())
    
    # Show unique top-level prefixes after our mapping
    mapped_prefixes = set()
    for k in keys:
        # Apply same mapping as Swift code
        if k.startswith('language_model.model.'):
            k = k[21:]
        elif k.startswith('language_model.'):
            k = k[15:]
        elif k.startswith('multi_modal_projector.'):
            k = 'projector.' + k[22:].replace('linear_0', 'linear1').replace('linear_2', 'linear2')
        
        k = k.replace('embed_tokens', 'embedTokens')
        k = k.replace('lm_head', 'lmHead')
        
        top = k.split('.')[0]
        mapped_prefixes.add(top)
    
    print('Mapped top-level keys:', sorted(mapped_prefixes))
    
    # Show a few full mapped keys
    print()
    print('Sample mapped keys:')
    for orig in keys[:10]:
        k = orig
        if k.startswith('language_model.model.'):
            k = k[21:]
        elif k.startswith('language_model.'):
            k = k[15:]
        elif k.startswith('multi_modal_projector.'):
            k = 'projector.' + k[22:].replace('linear_0', 'linear1').replace('linear_2', 'linear2')
        
        k = k.replace('embed_tokens', 'embedTokens').replace('lm_head', 'lmHead')
        k = k.replace('self_attn', 'selfAttn').replace('input_layernorm', 'inputLayernorm')
        k = k.replace('post_attention_layernorm', 'postAttentionLayernorm')
        k = k.replace('q_proj', 'qProj').replace('k_proj', 'kProj').replace('v_proj', 'vProj').replace('o_proj', 'oProj')
        k = k.replace('gate_proj', 'gateProj').replace('up_proj', 'upProj').replace('down_proj', 'downProj')
        print(f'  {orig} -> {k}')
