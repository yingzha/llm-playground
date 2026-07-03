# Prompt attribution

The prompts in this directory are **adapted from NVIDIA's Apache-2.0 code**, with
the warehouse-monitoring domain generalized to arbitrary video. Kept here as a
NOTICE (rather than inline SPDX headers) because each `.txt` file is loaded
verbatim as prompt text.

> Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
> Licensed under the Apache License, Version 2.0.
> http://www.apache.org/licenses/LICENSE-2.0

Sources:

| File | Adapted from |
|------|--------------|
| `caption.txt` | VSS `services/video-summarization/.../via_stream_handler.py::_create_vlm_prompt` and `vss-ctx-rag/data/configs/vlm.yaml` (`caption`) — generalized from the warehouse prompt |
| `map_summarize.txt` | `vss-ctx-rag/data/configs/vlm.yaml` (`caption_summarization`) |
| `reduce_summarize.txt` | `vss-ctx-rag/data/configs/vlm.yaml` (`summary_aggregation`) |
| `qa_answer.txt` | `vss-ctx-rag/src/vss_ctx_rag/utils/prompts.py` (`CHAT_SYSTEM_TEMPLATE_PREFIX/SUFFIX`) |
| `question_transform.txt` | `vss-ctx-rag/src/vss_ctx_rag/utils/prompts.py` (`QUESTION_TRANSFORM_TEMPLATE`) |

Repos: <https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization> ·
<https://github.com/NVIDIA/context-aware-rag>
