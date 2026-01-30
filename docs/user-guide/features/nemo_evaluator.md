# Evaluating Checkpoints with NeMo Evaluator

This guide explains how to evaluate Megatron-LM checkpoints using 
[NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator).

## Overview

NeMo Evaluator is an open-source library for evaluating large language models across standard 
benchmarks (MMLU, ARC, HellaSwag, etc.). Megatron-LM checkpoints saved in MCore distributed 
checkpoint format can be deployed as OpenAI-compatible endpoints and evaluated using the 
NeMo Evaluator SDK.

The evaluation workflow consists of two steps:

1. **Deploy** the checkpoint as an inference endpoint using NeMo Export-Deploy
2. **Evaluate** the deployed model using NeMo Evaluator SDK

## Prerequisites

- Megatron-LM checkpoint saved with `--ckpt-format torch_dist`
- GPUs matching the checkpoint's tensor/pipeline parallelism requirements

**Recommended: Use the NeMo Framework Container**

The NeMo Framework container includes all required dependencies:

```bash
docker pull nvcr.io/nvidia/nemo:25.04

docker run --gpus all -it --rm \
    -v /path/to/checkpoints:/workspace/checkpoints \
    --shm-size=4g \
    --ipc=host \
    nvcr.io/nvidia/nemo:25.04
```

For manual installation, see [NeMo Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy) 
and [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator).

## Checkpoint Format Requirements

Only checkpoints saved in **MCore distributed checkpoint format** are supported.

To save checkpoints in the supported format during training:

```bash
python pretrain_gpt.py \
    --ckpt-format torch_dist \
    --save /path/to/checkpoints
```

The checkpoint directory structure should look like:

```
checkpoint/
├── iter_0010000/
│   ├── common.pt
│   ├── metadata.json
│   └── __0_0.distcp/
```

Legacy Megatron-LM checkpoints (`mp_rank_XX/model_optim_rng.pt` format) are not supported. 
See [Legacy Checkpoint Workarounds](#legacy-checkpoint-workarounds) for conversion options.

## Tokenizer Limitation

Megatron-LM checkpoints save tokenizer **paths** in `args`, not the actual tokenizer files. 
This means the original tokenizer path must be accessible at inference time, or you need to 
apply one of the following workarounds.

**Workaround 1: Copy tokenizer files to checkpoint (Recommended)**

```bash
mkdir -p /path/to/checkpoint/iter_0010000/tokenizer
cp -r /original/path/to/tokenizer/* /path/to/checkpoint/iter_0010000/tokenizer/
```

Export-Deploy will automatically use `{checkpoint}/tokenizer` when the original path is unavailable.

**Workaround 2: Specify tokenizer explicitly during evaluation**

```python
eval_params = ConfigParams(
    extra={
        "tokenizer": "/accessible/path/to/tokenizer",  # or HF model ID like "meta-llama/Llama-3.1-8B"
        "tokenizer_backend": "huggingface",
    },
)
```

## Step 1: Deploy the Checkpoint

Use NeMo Export-Deploy to serve the checkpoint as an OpenAI-compatible endpoint.

**Using Ray Serve:**

```bash
python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
    --megatron_checkpoint /workspace/checkpoints/iter_0010000 \
    --model_id megatron_model \
    --model_type gpt \
    --port 8080 \
    --num_gpus 4 \
    --tensor_model_parallel_size 2
```

**Using PyTriton:**

```bash
python /opt/Export-Deploy/scripts/deploy/nlp/deploy_triton_inframework.py \
    --megatron_checkpoint /workspace/checkpoints/iter_0010000 \
    --model_id megatron_model \
    --model_type gpt \
    --triton_port 8080 \
    --num_gpus 4 \
    --tensor_model_parallel_size 2
```

**Key Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--megatron_checkpoint` | Path to checkpoint directory (e.g., `iter_0010000/`) |
| `--model_id` | Identifier for API requests |
| `--model_type` | Model architecture (`gpt`, `mamba`, etc.) |
| `--tensor_model_parallel_size` | Must match checkpoint's TP degree |
| `--num_gpus` | Total GPUs for inference |

When using Ray Serve, ensure sufficient CPU cores are available. Ray may request multiple 
CPUs per replica.

## Step 2: Evaluate the Deployed Model

Once deployed, use the NeMo Evaluator SDK to run benchmarks.

```python
from nemo_evaluator.api import check_endpoint, evaluate
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint, ConfigParams, EvaluationConfig, EvaluationTarget
)

api_endpoint = ApiEndpoint(
    url="http://localhost:8080/v1/completions/",
    type="completions",
    model_id="megatron_model",
)

eval_target = EvaluationTarget(api_endpoint=api_endpoint)

eval_params = ConfigParams(
    top_p=0,
    temperature=0,
    limit_samples=100,  # Remove for full evaluation
    extra={
        "tokenizer": "/workspace/checkpoints/iter_0010000/tokenizer",
        "tokenizer_backend": "huggingface",
    },
)

eval_config = EvaluationConfig(
    type="mmlu",  # or arc_challenge, hellaswag, winogrande, gsm8k, humaneval
    params=eval_params,
    output_dir="./results"
)

check_endpoint(
    endpoint_url=eval_target.api_endpoint.url,
    endpoint_type=eval_target.api_endpoint.type,
    model_name=eval_target.api_endpoint.model_id,
)

evaluate(target_cfg=eval_target, eval_cfg=eval_config)
```

For the full list of supported benchmarks, see the 
[NeMo Evaluator documentation](https://docs.nvidia.com/nemo/evaluator/latest/).

## Legacy Checkpoint Workarounds

For legacy Megatron-LM checkpoints (`mp_rank_XX/model_optim_rng.pt` format):

1. **Resave in distributed format**: Load in the original training environment and resave 
   with `--ckpt-format torch_dist`

2. **Convert via HuggingFace**: Convert to HuggingFace format, then use Megatron-Bridge:

```python
from megatron.bridge import AutoBridge
AutoBridge.import_ckpt(
    model="meta-llama/Llama-3.1-8B",
    output_path="/path/to/mbridge_checkpoint",
)
```

## Troubleshooting

**Tokenizer not found**: Megatron-LM checkpoints don't include tokenizer files. 
See [Tokenizer Limitation](#tokenizer-limitation) for workarounds.

**AttributeError when loading checkpoint**: Your checkpoint may be from an older Megatron-LM 
version with missing args. Consider resaving with a newer version.

**Tensor parallelism mismatch**: Ensure `--tensor_model_parallel_size` matches the 
parallelism used when saving the checkpoint.

## References

- [NeMo Evaluator Repository](https://github.com/NVIDIA-NeMo/Evaluator)
- [NeMo Evaluator Documentation](https://docs.nvidia.com/nemo/evaluator/latest/)
- [NeMo Export-Deploy Repository](https://github.com/NVIDIA-NeMo/Export-Deploy)
- [NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)
