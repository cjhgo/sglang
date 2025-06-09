
def convert_weights(model):
    import os
    named_parameters = model.named_parameters()
    # 合并llm.model.layers.*.self_attn.q_proj.*,llm.model.layers.*.self_attn.k_proj.*,llm.model.layers.*.self_attn.v_proj.* 为 llm.model.layers.*.self_attn.qkv_proj.*
    
    # Convert named_parameters to dict for easier manipulation
    param_dict = dict(named_parameters)
    new_param_dict = {}
    
    # Track which layers we've processed
    processed_layers = set()
    with torch.no_grad():
        for name, param in param_dict.items():
            # Check if this is a q_proj, k_proj, or v_proj parameter
            if '.self_attn.q_proj.' in name or '.self_attn.k_proj.' in name or '.self_attn.v_proj.' in name:
                # Extract layer identifier (everything before .self_attn)
                layer_prefix = name.split('.self_attn.')[0]
                param_suffix = name.split('.self_attn.')[1].split('.', 1)[1]  # Get everything after q_proj/k_proj/v_proj
                
                if layer_prefix not in processed_layers:
                    # Get q, k, v parameters for this layer
                    q_name = f"{layer_prefix}.self_attn.q_proj.{param_suffix}"
                    k_name = f"{layer_prefix}.self_attn.k_proj.{param_suffix}"
                    v_name = f"{layer_prefix}.self_attn.v_proj.{param_suffix}"
                    
                    if q_name in param_dict and k_name in param_dict and v_name in param_dict:
                        # Concatenate q, k, v parameters
                        q_param = param_dict[q_name]
                        k_param = param_dict[k_name]
                        v_param = param_dict[v_name]
                        
                        # Concatenate along the first dimension
                        qkv_param = torch.cat([q_param, k_param, v_param], dim=0)
                        
                        # Create new parameter name
                        qkv_name = f"{layer_prefix}.self_attn.qkv_proj.{param_suffix}"
                        new_param_dict[qkv_name] = qkv_param
                        
                        processed_layers.add(layer_prefix)
            else:
                pass
                # new_param_dict[name] = param
        
    # Convert back to named_parameters format
    named_parameters = list(new_param_dict.items())
    return named_parameters