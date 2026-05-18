"""Expose FA2-style APIs on top of FA4. Imported once at startup."""
import flash_attn

try:
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen
except ImportError:
    _fa4_varlen = None


def flash_attn_varlen_qkvpacked_func(
    qkv, cu_seqlens, max_seqlen,
    dropout_p=0.0, softmax_scale=None, causal=False, **_kwargs,
):
    """FA2-compatible shim over FA4's flash_attn_varlen_func."""
    if _fa4_varlen is None:
        raise RuntimeError("FA4 not installed; cannot use flash_attn shim.")
    if dropout_p != 0.0:
        raise NotImplementedError("FA4 shim does not yet pass dropout through.")
    q, k, v = qkv.unbind(dim=1)
    return _fa4_varlen(
        q, k, v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        causal=causal, softmax_scale=softmax_scale,
    )


# Make the shim discoverable as if it were FA2.
flash_attn.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
