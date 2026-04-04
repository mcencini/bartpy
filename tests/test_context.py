"""Tests for bartorch.core.context (BartContext, bart_context)."""

import pytest
from bartorch.core.context import BartContext, bart_context


class TestBartContext:
    def test_not_active_by_default(self):
        assert not BartContext.is_active()
        assert BartContext.current() is None

    def test_active_inside_with(self):
        ctx = BartContext()
        with ctx:
            assert BartContext.is_active()
            assert BartContext.current() is ctx

    def test_not_active_after_with(self):
        ctx = BartContext()
        with ctx:
            pass
        assert not BartContext.is_active()

    def test_fresh_name_unique(self):
        ctx = BartContext()
        names = {ctx.fresh_name() for _ in range(100)}
        assert len(names) == 100

    def test_fresh_name_mem_suffix(self):
        ctx = BartContext()
        assert ctx.fresh_name().endswith(".mem")

    def test_register_unregister(self):
        import torch
        ctx = BartContext()
        t = torch.zeros(1)
        ctx.register("foo.mem", t)
        assert "foo.mem" in ctx._registered
        ctx.unregister("foo.mem")
        assert "foo.mem" not in ctx._registered


class TestBartContextManager:
    def test_yields_context(self):
        with bart_context() as ctx:
            assert isinstance(ctx, BartContext)
            assert BartContext.is_active()

    def test_cleanup_on_exit(self):
        with bart_context():
            pass
        assert not BartContext.is_active()
