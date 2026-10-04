"""把官方金标 + data.zip 精标 + 华南理工 15k 精标混成三组训练 JSONL。

只用这三份来源。官方全部保留；精标按情绪定额抽样，避免 neutral/gratitude/helplessness 把分布带偏。
输出写到 E:\\Datasets\\chuangzhi-mimo\\mix\\。
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

OFFICIAL = Path(r"c:\Users\ASUS\Desktop\创智AI\官方数据\训练-验证-数据集\train\train_public.jsonl")
DATAZIP = Path(r"E:\Datasets\chuangzhi-mimo\datazip\datazip_mimo_labeled_official.jsonl")
SCUT15K = Path(r"E:\Datasets\chuangzhi-mimo\scut15k\scut15k_mimo_labeled_official.jsonl")
OUT_DIR = Path(r"E:\Datasets\chuangzhi-mimo\mix")
SEED = 42

# 两份精标合计的抽取上限。不够就全取，不复制。
QUOTA_B = {
    "surprise": 200, "care": 800, "disgust": 300, "joy": 250, "pride": 80,
    "fear": 1200, "helplessness": 1200, "sadness": 1200, "loneliness": 800,
    "shame": 700, "anger": 600, "gratitude": 700, "mixed": 1200, "neutral": 700,
    "anxiety": 1500, "relaxed": 2000,
}
QUOTA_C = {
    "surprise": 200, "care": 1200, "disgust": 320, "joy": 250, "pride": 80,
    "fear": 1800, "helplessness": 1800, "sadness": 1600, "loneliness": 1200,
    "shame": 1000, "anger": 800, "gratitude": 500, "mixed": 1500, "neutral": 400,
    "anxiety": 800, "relaxed": 2000,
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten_points(rows: list[dict], source: str) -> list[dict]:
    points = []
    for row in rows:
        for point in row.get("prediction_points") or []:
            emotion = (point.get("target") or {}).get("emotion_label")
            if not emotion:
                continue
            points.append({
                "source": source,
                "conversation_id": row["conversation_id"],
                "turns": row["turns"],
                "point": point,
                "emotion": emotion,
            })
    return points


def sample_by_quota(points: list[dict], quota: dict[str, int], seed: int) -> list[dict]:
    grouped = defaultdict(list)
    for item in points:
        grouped[item["emotion"]].append(item)
    rng = random.Random(seed)
    chosen = []
    for emotion, limit in quota.items():
        pool = grouped.get(emotion, [])
        rng.shuffle(pool)
        chosen.extend(pool[:limit])
    return chosen


def regroup(items: list[dict]) -> list[dict]:
    by_id = defaultdict(list)
    turns = {}
    for item in items:
        cid = item["conversation_id"]
        turns[cid] = item["turns"]
        by_id[cid].append(item["point"])
    rows = []
    for cid, points in by_id.items():
        points = sorted(points, key=lambda p: p["target_user_turn_id"])
        seen = set()
        unique = []
        for point in points:
            key = (point["target_user_turn_id"], point["assistant_response_turn_id"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(point)
        rows.append({
            "conversation_id": cid,
            "turns": turns[cid],
            "prediction_points": unique,
        })
    rows.sort(key=lambda row: row["conversation_id"])
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_emotions(rows: list[dict]) -> Counter:
    counter = Counter()
    for row in rows:
        for point in row["prediction_points"]:
            counter[point["target"]["emotion_label"]] += 1
    return counter


def point_count(rows: list[dict]) -> int:
    return sum(len(row["prediction_points"]) for row in rows)


def official_as_items(rows: list[dict]) -> list[dict]:
    return flatten_points(rows, "official")


def summarize(name: str, rows: list[dict]) -> dict:
    emo = count_emotions(rows)
    n = point_count(rows)
    return {
        "file": name,
        "conversations": len(rows),
        "prediction_points": n,
        "emotion": dict(emo.most_common()),
    }


def main() -> None:
    official_rows = load_jsonl(OFFICIAL)
    datazip_rows = load_jsonl(DATAZIP)
    scut_rows = load_jsonl(SCUT15K)
    official_items = official_as_items(official_rows)
    silver_items = flatten_points(datazip_rows, "datazip") + flatten_points(scut_rows, "scut15k")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "mix_A_official.jsonl", official_rows)

    mix_b = regroup(official_items + sample_by_quota(silver_items, QUOTA_B, SEED))
    mix_c = regroup(official_items + sample_by_quota(silver_items, QUOTA_C, SEED + 1))
    write_jsonl(OUT_DIR / "mix_B_main.jsonl", mix_b)
    write_jsonl(OUT_DIR / "mix_C_rare.jsonl", mix_c)

    report = {
        "sources": {
            "official": summarize("train_public.jsonl", official_rows),
            "datazip": summarize(str(DATAZIP), datazip_rows),
            "scut15k": summarize(str(SCUT15K), scut_rows),
        },
        "mixes": {
            "A": summarize("mix_A_official.jsonl", official_rows),
            "B": summarize("mix_B_main.jsonl", mix_b),
            "C": summarize("mix_C_rare.jsonl", mix_c),
        },
        "rule": "只用官方 + data.zip 精标 + scut15k 精标。官方全保留。精标按定额抽样，不复制。",
        "out_dir": str(OUT_DIR),
    }
    (OUT_DIR / "mix_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: {"conversations": vv["conversations"], "prediction_points": vv["prediction_points"]} for kk, vv in report[k].items()} for k in ("sources", "mixes")}, ensure_ascii=False, indent=2))
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
