"""Read-only expert offload for bounded-memory diagnostics, never training/benchmarking."""
import torch


def offload_experts(model, resident_gib=25):
    """Keep complete expert modules on CPU between calls; all arithmetic stays on CUDA.

    CPU masters are restored without copying results back: weights are immutable.
    Encoder/decoder must share these expert modules. The caller must use inference_mode.
    """
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Load the model and adapter on CPU first")
    model.requires_grad_(False)
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    resident, selected = total, []
    for enc, dec in zip(model.model.encoder.language_model.layers, model.model.decoder.layers, strict=True):
        if resident <= resident_gib * 2**30:
            break
        assert enc.experts is dec.experts
        module = enc.experts
        selected.append(module)
        resident -= sum(p.numel() * p.element_size() for p in module.parameters())
    if resident > resident_gib * 2**30:
        raise ValueError("Resident budget too small even with all experts offloaded")
    masters = {id(p): p.detach() for module in selected for p in module.parameters()}
    # _apply preserves the tied Parameter objects under PyTorch's default conversion policy.
    model._apply(lambda tensor: tensor if id(tensor) in masters else tensor.to("cuda"))
    handles = []
    for module in selected:
        parameters = list(module.parameters())
        assert all(id(p) in masters and p.device.type == "cpu" for p in parameters)

        def enter(module, args, parameters=parameters):
            if torch.is_grad_enabled():
                raise RuntimeError("Diagnostic offload requires inference_mode")
            for parameter in parameters:
                parameter.data = masters[id(parameter)].to("cuda")

        def leave(module, args, output, parameters=parameters):
            for parameter in parameters:
                parameter.data = masters[id(parameter)]

        handles.extend([module.register_forward_pre_hook(enter),
                        module.register_forward_hook(leave, always_call=True)])
    return {"total_parameter_gib": total / 2**30, "resident_parameter_gib": resident / 2**30,
            "offloaded_expert_modules": len(selected)}, handles
