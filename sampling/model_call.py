from __future__ import annotations


def model_forward(
    model,
    input_ids,
    past_key_values=None,
    use_cache: bool = False,
    profiler=None,
    phase: str = "target_forward",
    logits_start: int | None = None,
    logits_end: int | None = None,
    **metadata,
):
    if getattr(model, "supports_logits_window", False):
        return model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_start=logits_start,
            logits_end=logits_end,
            profiler=profiler,
            profile_metadata=metadata,
        )

    if profiler is None:
        return model(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache)

    with profiler.time(phase, logits_start=logits_start, logits_end=logits_end, **metadata):
        return model(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache)
