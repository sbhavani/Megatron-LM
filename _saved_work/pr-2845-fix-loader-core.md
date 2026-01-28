# PR Draft: Fix model_provider return value in loader_core.py

## PR Details

- **Branch:** `fix/loader-core-model-provider-bug`
- **Target:** `NVIDIA/Megatron-LM:main`
- **Fixes:** #2845

## Create PR Command

```bash
gh pr create \
  --repo NVIDIA/Megatron-LM \
  --base main \
  --head sbhavani:fix/loader-core-model-provider-bug \
  --title "Fix model_provider return value in loader_core.py" \
  --body "$(cat <<'EOF'
## Summary

- Fix bug in `loader_core.py` where `import_model_provider()` returned the raw `model_provider` function instead of the partial with `gpt_builder` bound
- This caused `TypeError: model_provider() missing 1 required positional argument: 'model_builder'` when converting GPT checkpoints

## Root Cause

The `model_provider` function signature was updated to require `model_builder` as its first positional argument:

```python
def model_provider(
    model_builder: Callable, pre_process=True, post_process=True, vp_stage: Optional[int] = None
) -> Union[GPTModel, megatron.legacy.model.GPTModel, MambaModel]:
```

In `loader_core.py`, the code correctly creates a partial function with `gpt_builder` bound:
```python
self.model_provider = partial(model_provider, gpt_builder)
```

But then incorrectly returns the raw function instead of the partial:
```python
return model_provider  # Bug: should be self.model_provider
```

## Fix

Changed line 61 in `tools/checkpoint/loader_core.py`:
```diff
- return model_provider
+ return self.model_provider
```

## Test plan

- [ ] Run checkpoint converter functional tests
- [ ] Test GPT checkpoint conversion with `--loader core`

Fixes #2845
EOF
)"
```

## Issue Comment Draft (for #2845)

```markdown
@littlebeanfang Thanks for reporting this - you've encountered a bug in the checkpoint converter code. The `model_provider` function signature was updated to require a `model_builder` argument, but `loader_core.py` wasn't updated to pass the partial function correctly.

That said, for converting Megatron checkpoints to HuggingFace format, I'd recommend checking out [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge).

Megatron Bridge provides bidirectional converters that support:
• Hugging Face → Megatron (for continued pretraining)  
• Megatron → Hugging Face (for exporting trained models back to HF format)

This is the recommended path for checkpoint format conversion between Megatron and HuggingFace ecosystems.
```

## Files Changed

- `tools/checkpoint/loader_core.py` (1 line change)
