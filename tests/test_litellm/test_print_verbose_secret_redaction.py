"""[ARC-BUG-09] print_verbose() stdout must not leak secrets and must neutralize
control chars / record separators when set_verbose is True."""

import litellm
from litellm._logging import _redact_and_sanitize

SECRET_VALUE = "sk-ant-test-DO-NOT-LEAK-12345"


def test_redact_and_sanitize_redacts_and_neutralizes():
    out = _redact_and_sanitize({"api_key": SECRET_VALUE, "note": "a\r\nb\x9b\x85"})
    assert SECRET_VALUE not in out and "REDACTED" in out
    assert "\\r\\n" in out and "\x9b" not in out and "\x85" not in out


def test_utils_print_verbose_redacts_stdout(capsys, monkeypatch):
    from litellm.utils import print_verbose as utils_print_verbose

    monkeypatch.setattr(litellm, "set_verbose", True)
    utils_print_verbose({"api_key": SECRET_VALUE, "model": "gpt-4o"})
    out = capsys.readouterr().out
    assert SECRET_VALUE not in out and "REDACTED" in out


def test_logging_print_verbose_redacts_stdout(capsys, monkeypatch):
    import litellm._logging as _logging_mod

    monkeypatch.setattr(_logging_mod, "set_verbose", True)
    _logging_mod.print_verbose({"api_key": SECRET_VALUE})
    out = capsys.readouterr().out
    assert SECRET_VALUE not in out and "REDACTED" in out
