"""Hugging Face export of a trained robo-jev checkpoint (Task hf0).

* :mod:`~robo_jev.hf.export` — the training checkpoint (``robo-jev-checkpoint-v0``, LoRA) → a PEFT adapter folder and a merged
  ``transformers`` folder, each with the tokenizer, ``robo_jev.json`` (prompt, answer codes, training summary, provenance), the model
  card, the standalone runner and the hero chart. Safetensors only — no pickle files.
* :mod:`~robo_jev.hf.hub` — the token file, private repo creation, upload, the ``v0.1`` tag and the listing of what is on the Hub.
* ``robo_jev_decide.py`` — the standalone runner shipped in both repos (imports nothing from this package; tests/test_infer.py pins it
  to :class:`robo_jev.infer.RoboJev`).

Entry point: ``scripts/export_hf.py``.
"""
