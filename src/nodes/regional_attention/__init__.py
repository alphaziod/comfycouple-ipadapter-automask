"""Regional (character-aware) attention for the Saya Couple pipeline.

Modules:

* :mod:`.node` - the ``Attention couple`` ComfyUI node (installs the patches).
* :mod:`.cross_attention` / :mod:`.self_attention` - the ``attn2`` / ``attn1``
  replacements.
* :mod:`.ownership` - the tri-state ``P1/P2/Unknown`` maths shared by both.
* :mod:`.anima` - the non-UNet (ANIMA / DiT) forward-hook variant.
* :mod:`.masks` / :mod:`.context` / :mod:`.operations` - mask, context and
  attention-execution helpers.
* :mod:`.replacement` - the callable stored in ComfyUI's ``patches_replace`` table.
"""
