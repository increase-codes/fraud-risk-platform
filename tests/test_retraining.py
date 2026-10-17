
from fraud_risk import retraining


def test_rejects_candidate_below_recall_threshold(
    monkeypatch, tmp_path, capsys
):
    candidate_path = tmp_path / "candidate.joblib"
    monkeypatch.setattr(retraining, "CANDIDATE_PATH", candidate_path)
    monkeypatch.setattr(
        retraining,
        "train_candidate",
        lambda: (object(), {"pr_auc": 0.2, "recall": 0.60}),
    )

    retraining.main()

    assert not candidate_path.exists()
    assert "REJECTED" in capsys.readouterr().out


def test_saves_candidate_meeting_recall_threshold(
    monkeypatch, tmp_path, capsys
):
    candidate_path = tmp_path / "candidate.joblib"
    candidate_model = {"model": "candidate"}
    monkeypatch.setattr(retraining, "CANDIDATE_PATH", candidate_path)
    monkeypatch.setattr(
        retraining,
        "train_candidate",
        lambda: (candidate_model, {"pr_auc": 0.2, "recall": 0.70}),
    )

    saved = {}

    def fake_dump(model, path):
        saved["model"] = model
        saved["path"] = path

    monkeypatch.setattr(retraining.joblib, "dump", fake_dump)

    retraining.main()

    assert saved["model"] is candidate_model
    assert saved["path"] == candidate_path
    assert "ACCEPTED" in capsys.readouterr().out
