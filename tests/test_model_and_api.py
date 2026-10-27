from concurrent.futures import ThreadPoolExecutor
import math

from fastapi.testclient import TestClient
import pytest
import torch

from tinyllm.api import create_app
from tinyllm.artifacts import load_checkpoint, save_checkpoint
from tinyllm.data import CharTokenizer, get_batch
from tinyllm.model import Config, TinyGPT, generate
from tinyllm.training import evaluate


@pytest.fixture
def model_and_tokenizer():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    tok = CharTokenizer.fit("hello model abcdefghijklmnopqrstuvwxyz. \n")
    return TinyGPT(Config(tok.vocab_size, context=8, width=16, heads=2, layers=1, dropout=0.1)), tok


def test_future_cannot_change_past(model_and_tokenizer):
    model, tok = model_and_tokenizer
    model.eval()
    first = torch.tensor([tok.encode("hello")])
    changed = first.clone()
    changed[0, 3:] = torch.tensor(tok.encode("ab"))
    torch.testing.assert_close(model(first)[0][:, :3], model(changed)[0][:, :3])


def test_shift_and_parameter_update(model_and_tokenizer):
    model, tok = model_and_tokenizer
    source = torch.tensor(tok.encode("hello model " * 5))
    x, y = get_batch(source, 8, 2, torch.Generator().manual_seed(7))
    assert torch.equal(x[:, 1:], y[:, :-1])
    old = model.output.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    loss = model(x, y)[1]
    loss.backward()
    assert torch.isfinite(model.output.weight.grad).all()
    optimizer.step()
    assert not torch.equal(old, model.output.weight)


def test_checkpoint_logits_roundtrip(model_and_tokenizer, tmp_path):
    model, tok = model_and_tokenizer
    model.eval()
    ids = torch.tensor([tok.encode("hello")])
    save_checkpoint(model, tok, tmp_path / "model.pt", {"step": 1})
    restored, restored_tok, meta = load_checkpoint(tmp_path / "model.pt")
    torch.testing.assert_close(model(ids)[0], restored(ids)[0], rtol=0, atol=0)
    assert restored_tok.encode("hello") == ids[0].tolist()
    assert meta["step"] == 1


def test_sampling_and_eval_preserve_state(model_and_tokenizer):
    model, tok = model_and_tokenizer
    ids = torch.tensor([tok.encode("hello")])
    rng_state = torch.get_rng_state().clone()
    a = generate(model, ids, count=12, seed=17)
    b = generate(model, ids, count=12, seed=17)
    assert torch.equal(a, b)
    assert torch.equal(rng_state, torch.get_rng_state())
    assert model.training
    data = torch.tensor(tok.encode("hello model " * 5))
    assert evaluate(model, data) == evaluate(model, data)
    assert model.training
    assert torch.equal(rng_state, torch.get_rng_state())
    assert generate(model, ids, count=3, temperature=0).shape == (1, 8)
    with pytest.raises(ValueError):
        generate(model, ids, temperature=float("nan"))


def test_api_contract_and_concurrent_sampling(model_and_tokenizer, tmp_path):
    model, tok = model_and_tokenizer
    path = tmp_path / "model.pt"
    save_checkpoint(model, tok, path, {"step": 1})
    with TestClient(create_app(path)) as client:
        assert client.get("/health").status_code == 200
        body = {"prompt": "hello", "max_new_tokens": 8, "seed": 17}
        responses = list(ThreadPoolExecutor(2).map(lambda _: client.post("/generate", json=body), range(2)))
        assert all(r.status_code in (200, 429) for r in responses)
        accepted = [r.json() for r in responses if r.status_code == 200]
        assert accepted
        repeat = client.post("/generate", json=body).json()
        assert all(r["text"] == repeat["text"] for r in accepted)
        assert repeat["generated_tokens"] == 8
        for invalid in ({"prompt": ""}, {"prompt": "😀"}, {"prompt": "h" * 9},
                        {"prompt": "hello", "temperature": -1},
                        {"prompt": "hello", "max_new_tokens": 1000}):
            assert client.post("/generate", json=invalid).status_code == 422


def test_bad_config_and_unknown_character():
    with pytest.raises(ValueError):
        Config(vocab=10, width=17, heads=4)
    tok = CharTokenizer.fit("ab")
    with pytest.raises(ValueError):
        tok.encode("z")
    assert tok.encode("z", strict=False) == [0]
