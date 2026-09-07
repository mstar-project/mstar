from mstar.model.kimi_k3.tokenizer import KimiK3Tokenizer


def test_tokenizer_special_ids_and_chat(tiny_dir):
    tok = KimiK3Tokenizer(tiny_dir)
    assert tok.bos_id == 163584 and tok.eos_id == 163586 and tok.pad_id == 163839
    assert tok.media_pad_id == 163605 and tok.vocab_size == 163840
    text = "Hello, world! 你好 123"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    msgs = [{"role": "system", "content": "You are Kimi."}, {"role": "user", "content": "Hi there"}]
    rendered = tok.render_chat(msgs, thinking=True)
    assert rendered.startswith('<|open|>message role="system"')
    assert rendered.endswith('<|open|>message role="assistant"<|sep|><|open|>think<|sep|>')
    ids = tok.apply_chat_template(msgs, thinking=True)
    assert ids[0] == 163587 and ids[-1] == 163589  # <|open|> ... <|sep|>
    assert tok.decode(ids) == rendered
    # special strings inside user text are NOT special tokens
    assert 163587 not in tok.encode("<|open|>message")
    assert tok.encode("<|open|>", allow_special_tokens=True) == [163587]
    assert tok.decode([163587, 163586], skip_special_tokens=True) == "<|open|>"
    instruct = tok.render_chat(msgs, thinking=False)
    assert instruct.endswith('<|open|>response<|sep|>')
