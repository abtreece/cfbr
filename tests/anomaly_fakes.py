"""A fake Anthropic client for agent tests: no network, scripted responses."""

from types import SimpleNamespace as NS


def usage(i=1000, o=200):
    return NS(input_tokens=i, output_tokens=o, cache_creation_input_tokens=0, cache_read_input_tokens=0)


def tool_use(name, inp, id_):
    return NS(type="tool_use", name=name, input=inp, id=id_)


def text_block(t):
    return NS(type="text", text=t)


def thinking_block(t="pondering"):
    return NS(type="thinking", thinking=t, signature="sig")


def fallback_block():
    return NS(type="fallback", trigger="refusal")


def message(content, stop="tool_use", model="claude-opus-5-5", details=None, usage_=None):
    return NS(content=content, stop_reason=stop, model=model, usage=usage_ or usage(), _request_id="req_1",
              stop_details=details)


class _Stream:
    request_id = "req_stream"

    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.msg


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Stream(item(kw) if callable(item) else item)


class FakeAPIError(Exception):
    def __init__(self, status_code, type_=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.type = type_
