#!/usr/bin/env python3
"""生成 A/B/C 三套 LoRA 训练配方（官方 schema → SFT 样本）。

规则要点：
- A: 仅官方金标
- B: 官方 + 银标（压 neutral/gratitude，滤短回复/说教，银标画像不进 loss）
- C: 在 B 上对稀有类略过采样，更均衡（仍非完全均匀）

输出：
  mix_data/mix_{A,B,C}.jsonl   每行一个 SFT 样本（含 meta.loss_weight）
  mix_data/stats_{A,B,C}.json
"""
from __future__ import annotations

import json
import re
import random
import collections
from pathlib import Path

random.seed(42)

ROOT = Path(r"C:\Users\HUAWEI\XiaomiMiMoProjects\.mimo-sessions\2026\09\26\lora")
OFFICIAL = ROOT / "emotion_data" / "数字人综合情感陪伴对话模型" / "训练-验证-数据集" / "train" / "train_public.jsonl"
DATAZIP = ROOT / "label_datazip_mimo" / "out" / "datazip_mimo_labeled_official.jsonl"
SCUT = ROOT / "label_scut40k" / "out" / "datazip_mimo_labeled_official.jsonl"  # 实为 SCUT15k
OUT_DIR = ROOT / "mix_data"
OUT_DIR.mkdir(exist_ok=True)

EMOTIONS = [
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise", "mixed",
    "sadness", "loneliness", "anxiety", "anger", "fear", "disgust", "shame", "helplessness",
]
TRAITS = {
    "extroverted", "introverted", "open", "conservative", "high_conscientiousness",
    "casual", "agreeable", "assertive", "emotionally_stable", "sensitive",
}
INTERESTS = {
    "study_exam", "programming_technology", "reading_writing", "film_animation", "music",
    "games", "sports_fitness", "travel_outdoor", "pets", "social", "career_development", "art_design",
}
STYLES = {
    "brief", "detailed", "colloquial", "formal", "direct", "indirect", "humorous",
    "rational", "high_emotional_expression", "low_emotional_expression", "emoji_user",
}

SYSTEM_PROMPT = (
    "你是情感陪伴助手。根据对话历史，输出一个 JSON 对象，不要 Markdown。\n"
    "字段：response_text（中文共情回复）、emotion_label（最新用户情绪）、"
    "user_profile（personality_traits/interests/style）、memory_refs（固定 []）。\n"
    "emotion_label 只能从：" + ", ".join(EMOTIONS) + "\n"
    "画像无充分证据就输出空数组；memory_refs 始终 []。"
)

ADVICE_RE = re.compile(r"你可以尝试|建议你|步骤如下|第一[,、]|首先[，,]你|做法是：|方法有")
EMPATHY_RE = re.compile(r"理解|听起来|感受|不容易|辛苦|确实|心里|一定很|这种")
EMPTY_PROFILE = {"personality_traits": [], "interests": [], "style": []}


def compact(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def clean_profile(up: dict) -> dict:
    out = {"personality_traits": [], "interests": [], "style": []}
    if not isinstance(up, dict):
        return out
    for k, vocab in (("personality_traits", TRAITS), ("interests", INTERESTS), ("style", STYLES)):
        vals = up.get(k) or []
        if isinstance(vals, list):
            for v in vals:
                if v in vocab and v not in out[k]:
                    out[k].append(v)
    return out


def load_points(path: Path, source: str) -> list[dict]:
    """展开官方 schema 为预测点样本。"""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line)
            turns = conv.get("turns") or []
            tmap = {t["turn_id"]: t for t in turns}
            for pp in conv.get("prediction_points") or []:
                uid = pp.get("target_user_turn_id")
                tgt = pp.get("target") or {}
                hist = []
                for t in turns:
                    if t.get("turn_id", 0) <= (uid or 0):
                        hist.append({"role": t["role"], "content": t["content"]})
                user_text = tmap.get(uid, {}).get("content", "")
                reply = compact(tgt.get("response_text") or "")
                rows.append(
                    {
                        "source": source,
                        "conversation_id": conv.get("conversation_id"),
                        "target_user_turn_id": uid,
                        "history": hist,
                        "user_text": user_text,
                        "response_text": reply,
                        "emotion_label": tgt.get("emotion_label") or "",
                        "user_profile": clean_profile(tgt.get("user_profile") or {}),
                        "reply_len": len(reply),
                    }
                )
    return rows


def silver_ok(r: dict) -> tuple[bool, str]:
    """银标过滤。"""
    if not r["response_text"] or len(r["response_text"]) < 40:
        return False, "reply_short"
    if r["emotion_label"] not in EMOTIONS:
        return False, "bad_emotion"
    # 纯说教、无共情 → 丢
    if ADVICE_RE.search(r["response_text"]) and not EMPATHY_RE.search(r["response_text"] + r["user_text"]):
        return False, "advice_only"
    # 首轮就建议（官方很少）：用户历史仅 1 轮且说教
    if len(r["history"]) <= 2 and ADVICE_RE.search(r["response_text"]):
        return False, "first_turn_advice"
    return True, ""


def to_sft(r: dict, w_reply: float, w_emo: float, w_prof: float, use_profile: bool) -> dict:
    target_obj = {
        "response_text": r["response_text"],
        "emotion_label": r["emotion_label"],
        "user_profile": r["user_profile"] if use_profile else EMPTY_PROFILE,
        "memory_refs": [],
    }
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in r["history"]:
        messages.append({"role": t["role"], "content": t["content"]})
    return {
        "id": f"{r['source']}:{r['conversation_id']}:p{r['target_user_turn_id']}",
        "source": r["source"],
        "messages": messages,
        "target": target_obj,
        "meta": {
            "emotion_label": r["emotion_label"],
            "reply_len": r["reply_len"],
            "loss_weight": {
                "response_text": w_reply,
                "emotion_label": w_emo,
                "user_profile": w_prof,
            },
            "use_profile_supervision": use_profile,
        },
    }


def emotion_counts(samples: list[dict]) -> collections.Counter:
    return collections.Counter(s["target"]["emotion_label"] for s in samples)


def dump(path: Path, samples: list[dict], extra: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    report = {
        "n": len(samples),
        "by_source": dict(collections.Counter(s["source"] for s in samples)),
        "emotion_dist": dict(emotion_counts(samples).most_common()),
        **extra,
    }
    (path.parent / f"stats_{path.stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(path.name, "n=", len(samples), "sources=", report["by_source"])
    print("  emotion top:", report["emotion_dist"])


def main() -> None:
    print("loading...")
    off = load_points(OFFICIAL, "official")
    dz = load_points(DATAZIP, "datazip")
    sc = load_points(SCUT, "scut15k")
    print(f"official={len(off)} datazip={len(dz)} scut={len(sc)}")

    # ========== A: 仅官方 ==========
    mix_a = [to_sft(r, 1.0, 1.0, 1.0, True) for r in off]
    dump(OUT_DIR / "mix_A.jsonl", mix_a, {"recipe": "A", "note": "official only"})

    # ========== 银标过滤 ==========
    def filter_silver(rows, name):
        kept, drop = [], collections.Counter()
        for r in rows:
            ok, reason = silver_ok(r)
            if ok:
                kept.append(r)
            else:
                drop[reason] += 1
        print(f"silver {name}: kept {len(kept)} / {len(rows)} drop={dict(drop)}")
        return kept, drop

    dz_keep, dz_drop = filter_silver(dz, "datazip")
    sc_keep, sc_drop = filter_silver(sc, "scut15k")

    # ========== B: 官方 + 银标（压 neutral/gratitude） ==========
    # 目标：官方全量；银标合计约与官方点数 1:1
    # 压类：neutral, gratitude 欠采样；relaxed 全收
    CAP_RATIO = {"neutral": 0.02, "gratitude": 0.03, "helplessness": 0.08}
    # 相对银标池的上限比例（占该源银标数）

    def sample_silver(kept: list[dict], target_total: int) -> list[dict]:
        by_emo = collections.defaultdict(list)
        for r in kept:
            by_emo[r["emotion_label"]].append(r)
        out = []
        # relaxed 优先全要
        for emo, items in by_emo.items():
            if emo == "relaxed":
                out.extend(items)
        remain = max(0, target_total - len(out))
        # 其余类按 cap 与均分
        others = [e for e in by_emo if e != "relaxed"]
        quota = {}
        for e in others:
            cap = int(len(kept) * CAP_RATIO.get(e, 0.12))
            quota[e] = max(cap, 8)
        # 按 quota 抽，再补到 target
        for e in others:
            items = by_emo[e]
            take = min(quota[e], len(items))
            out.extend(random.sample(items, take))
        if len(out) > target_total:
            # 优先保留非 neutral
            non_neutral = [x for x in out if x["emotion_label"] != "neutral"]
            neutral = [x for x in out if x["emotion_label"] == "neutral"]
            keep_n = min(len(neutral), max(0, target_total - len(non_neutral)))
            out = non_neutral + (neutral[:keep_n] if keep_n < len(neutral) else neutral)
            if len(out) > target_total:
                out = out[:target_total]
        elif len(out) < target_total:
            pool = [x for x in kept if x not in out]
            if pool:
                out.extend(random.sample(pool, min(len(pool), target_total - len(out))))
        return out

    silver_all = dz_keep + sc_keep
    # B: 银标点数 ≈ 官方点数
    silver_b = sample_silver(silver_all, target_total=len(off))
    mix_b = [to_sft(r, 1.0, 1.0, 1.0, True) for r in off]
    mix_b += [to_sft(r, 0.6, 0.8, 0.0, False) for r in silver_b]
    random.shuffle(mix_b)
    dump(
        OUT_DIR / "mix_B.jsonl",
        mix_b,
        {
            "recipe": "B",
            "note": "official + silver (profile no supervise, down neutral/gratitude)",
            "silver_filter": {"datazip": dict(dz_drop), "scut": dict(sc_drop)},
            "silver_sampled": len(silver_b),
            "loss_weights": {
                "official": {"reply": 1, "emotion": 1, "profile": 1},
                "silver": {"reply": 0.6, "emotion": 0.8, "profile": 0},
            },
        },
    )

    # ========== C: 稀有类略过采样（仍偏官方，不搞完全均匀） ==========
    # 在 B 的银标上，对 joy/care/pride/surprise/disgust 复制 2 份
    RARE = {"joy", "care", "pride", "surprise", "disgust"}
    silver_c = list(silver_b)
    for r in silver_b:
        if r["emotion_label"] in RARE:
            silver_c.append(r)  # x2
            silver_c.append(r)  # x3 total for very rare
    # 对 relaxed 也略加（官方多、银标少）
    for r in silver_b:
        if r["emotion_label"] == "relaxed":
            silver_c.append(r)
    mix_c = [to_sft(r, 1.0, 1.0, 1.0, True) for r in off]
    mix_c += [to_sft(r, 0.6, 0.9, 0.0, False) for r in silver_c]
    random.shuffle(mix_c)
    dump(
        OUT_DIR / "mix_C.jsonl",
        mix_c,
        {
            "recipe": "C",
            "note": "B + rare/relaxed oversample (still not uniform)",
            "rare_oversample": sorted(RARE),
        },
    )

    print("\nDone ->", OUT_DIR)


if __name__ == "__main__":
    main()
