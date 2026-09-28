"""把华南理工 part_4 筛成官方训练集格式。

两段筛选：
1. 规则：去掉过短、寒暄、辱骂、概念讲解，并选出预测点。
2. 模型：DeepSeek / Qwen（OpenAI 兼容接口）判断是否保留，并给情绪和画像伪标签。

回复文本用助手原文，不由模型改写。memory_refs 固定为 []。

示例（PowerShell）：
  $env:DEEPSEEK_API_KEY = "sk-..."
  python 脚本/label_scut_part.py --provider deepseek --stage ping
  python 脚本/label_scut_part.py --provider deepseek --stage rules
  python 脚本/label_scut_part.py --provider deepseek --stage label --max-calls 200
  python 脚本/label_scut_part.py --provider deepseek --stage export

Qwen（阿里云 DashScope 兼容模式）：
  $env:DASHSCOPE_API_KEY = "sk-..."
  python 脚本/label_scut_part.py --provider qwen --model qwen-plus --stage label --max-calls 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "数据集" / "华南理工_分工" / "part_4.json"
DEFAULT_OUT = ROOT / "数据集" / "华南理工_分工" / "part_4_work"

EMOTIONS = [
    "joy", "gratitude", "relaxed", "care", "pride", "neutral", "surprise", "mixed",
    "sadness", "loneliness", "anxiety", "anger", "fear", "disgust", "shame", "helplessness",
]
TRAITS = [
    "extroverted", "introverted", "open", "conservative", "high_conscientiousness",
    "casual", "agreeable", "assertive", "emotionally_stable", "sensitive",
]
INTERESTS = [
    "study_exam", "programming_technology", "reading_writing", "film_animation", "music",
    "games", "sports_fitness", "travel_outdoor", "pets", "social", "career_development", "art_design",
]
STYLES = [
    "brief", "detailed", "colloquial", "formal", "direct", "indirect", "humorous",
    "rational", "high_emotional_expression", "low_emotional_expression", "emoji_user",
]
# 第四份约占全库定额的 1/4。不够就不要凑。
QUOTA = {
    "surprise": 150, "care": 150, "disgust": 150, "joy": 150, "pride": 150,
    "fear": 300, "helplessness": 300, "sadness": 300, "loneliness": 300,
    "shame": 300, "anger": 300, "gratitude": 300, "mixed": 300, "neutral": 300,
    "anxiety": 200, "relaxed": 200,
}
RARE = ["surprise", "care", "disgust", "joy", "pride"]
CONFUSABLE = ["fear", "helplessness", "sadness", "loneliness", "shame", "anger", "gratitude", "mixed", "neutral"]

PRESETS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
}

# 官方训练集已占用 train_conv_000001–008338。补充数据从 100000 起编，格式仍是 train_conv_ 加 6 位数字。
CONVERSATION_ID_RE = re.compile(r"^train_conv_[0-9]{6}$")
ID_OFFSET = 100_000
THANKS_RE = re.compile(
    r"^(好的?|谢谢(你的?(建议|支持|理解|帮助))?|感谢(你的?)?|嗯+|哦+|明白了|知道了|"
    r"我会(试试|尝试|好好实践)|这些(听起来)?都不错|没问题).{0,8}$"
)
LOW_INFO_RE = re.compile(r"^(哦+|嗯+|好的?|谢谢|感谢|我明白了|我知道了|我会)")
FEELING_RE = re.compile(
    r"害怕|担心|焦虑|难过|伤心|无助|生气|愤怒|讨厌|孤独|寂寞|羞耻|丢脸|压力|崩溃|"
    r"委屈|失望|开心|高兴|感激|厌倦|空虚|迷茫|自卑|不自信|内疚|痛苦|烦|累|分手|离开"
)
CONCEPT_RE = re.compile(r"什么是|什么叫|如何定义|的定义|是指什么")
BLOCK_RE = re.compile(r"傻逼|脑残|操你|妈的|滚蛋")
SYSTEM_PROMPT = """你是情感陪伴数据标注员。只输出一个 JSON 对象，不要 Markdown。

判断当前这一轮是否适合作为训练样本，并标注用户情绪与画像。
情绪只看用户自己的话，不看助手的话。助手回复只用来判断口吻。

emotion_label 只能是：
joy, gratitude, relaxed, care, pride, neutral, surprise, mixed,
sadness, loneliness, anxiety, anger, fear, disgust, shame, helplessness

画像标签只能从下面选择；没有原话证据就留空数组。
personality_traits: extroverted, introverted, open, conservative, high_conscientiousness, casual, agreeable, assertive, emotionally_stable, sensitive
interests: study_exam, programming_technology, reading_writing, film_animation, music, games, sports_fitness, travel_outdoor, pets, social, career_development, art_design
style: brief, detailed, colloquial, formal, direct, indirect, humorous, rational, high_emotional_expression, low_emotional_expression, emoji_user

keep=false 的情况：
- 用户不是在讲自己的处境或感受，而是在问概念定义
- 助手主要在列步骤、提要求、讲道理，没有接住用户的感受（advice_dump）
- 情绪无法从用户原话判断

输出字段：
{
  "keep": true,
  "drop_reason": "",
  "reply_style": "empathic",
  "emotion_label": "anxiety",
  "confidence": 0.0,
  "emotion_evidence": "必须是当前用户原话的连续摘录",
  "not_anxiety_reason": "若标成 fear/helplessness/sadness/loneliness/shame/anger/gratitude/mixed/neutral，用一句话说明为什么不是 anxiety；否则空字符串",
  "user_profile": {"personality_traits": [], "interests": [], "style": []},
  "profile_evidence": {}
}
reply_style 只能是 empathic、advice_dump、other。
confidence 取 0 到 1。profile_evidence 的键是画像标签，值是用户原话摘录。"""


def iter_dialogs(path: Path, limit: int | None):
    decoder = json.JSONDecoder()
    buf = ""
    seen = False
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk and not buf.strip():
                break
            buf += chunk
            while True:
                buf = buf.lstrip()
                if not buf:
                    break
                if not seen:
                    if buf[0] != "[":
                        raise ValueError("输入不是 JSON 数组")
                    buf = buf[1:]
                    seen = True
                    continue
                if buf[0] == "]":
                    return
                if buf[0] == ",":
                    buf = buf[1:]
                    continue
                try:
                    obj, end = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                yield obj
                count += 1
                buf = buf[end:]
                if limit is not None and count >= limit:
                    return
            if not chunk:
                break


def conversation_id_for(dialog_id: int) -> str:
    number = int(dialog_id) + ID_OFFSET
    if not 0 <= number <= 999999:
        raise ValueError(f"对话 id {dialog_id} 超出 6 位编号范围")
    conversation_id = f"train_conv_{number:06d}"
    if not CONVERSATION_ID_RE.fullmatch(conversation_id):
        raise ValueError(conversation_id)
    return conversation_id


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def is_thanks(text: str) -> bool:
    folded = re.sub(r"[，。！？、\s~～]", "", text or "")
    if len(folded) <= 18 and bool(THANKS_RE.match(folded)):
        return True
    if FEELING_RE.search(text or ""):
        return False
    return len(folded) <= 40 and bool(LOW_INFO_RE.match(folded))


def is_concept(text: str) -> bool:
    return bool(CONCEPT_RE.search(text or "")) and ("我" not in text) and ("自己" not in text)


def usable_pair(user_text: str, assistant_text: str) -> str | None:
    if BLOCK_RE.search(user_text) or BLOCK_RE.search(assistant_text):
        return "blocked"
    if len(user_text.strip()) < 8:
        return "user_short"
    if len(assistant_text.strip()) < 20:
        return "assistant_short"
    if is_thanks(user_text):
        return "low_info"
    if is_concept(user_text):
        return "concept"
    return None


def build_candidate(dialog: dict) -> list[dict]:
    messages = dialog.get("messages") or []
    turns = []
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        turns.append({"turn_id": len(turns) + 1, "role": role, "content": content})

    pairs = []
    for index in range(len(turns) - 1):
        if turns[index]["role"] != "user" or turns[index + 1]["role"] != "assistant":
            continue
        reason = usable_pair(turns[index]["content"], turns[index + 1]["content"])
        if reason:
            continue
        pairs.append(index)
    if len(pairs) > 1:
        pairs = pairs[1:]
    if len(pairs) > 3:
        pairs = pairs[-3:]

    dialog_id = dialog.get("id")
    found = []
    for index in pairs:
        user_turn = turns[index]
        assistant_turn = turns[index + 1]
        history_users = [turn["content"] for turn in turns[: index + 1] if turn["role"] == "user"]
        found.append({
            "candidate_id": f"{dialog_id}:{user_turn['turn_id']}",
            "dialog_id": dialog_id,
            "topic": dialog.get("topic") or "",
            "target_user_turn_id": user_turn["turn_id"],
            "assistant_response_turn_id": assistant_turn["turn_id"],
            "turns": turns,
            "user_text": user_turn["content"],
            "history_user_text": "\n".join(history_users),
            "assistant_text": assistant_turn["content"],
        })
    return found


def write_jsonl(path: Path, rows: list[dict], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stage_rules(input_path: Path, out_dir: Path, limit: int | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "candidates.jsonl"
    labeled = out_dir / "labeled.jsonl"
    if labeled.exists() and limit is None:
        raise SystemExit(f"{labeled} 已存在。重跑规则会让旧标注对不上。确认后删除该文件，或换一个 --out-dir。")
    dialogs = 0
    kept = 0
    drop_reasons = Counter()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for dialog in iter_dialogs(input_path, limit):
            dialogs += 1
            messages = dialog.get("messages") or []
            for index in range(len(messages) - 1):
                user = messages[index]
                assistant = messages[index + 1]
                if user.get("role") != "user" or assistant.get("role") != "assistant":
                    continue
                reason = usable_pair(user.get("content") or "", assistant.get("content") or "")
                if reason:
                    drop_reasons[reason] += 1
            rows = build_candidate(dialog)
            kept += len(rows)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if dialogs % 5000 == 0:
                print(f"规则筛选已扫描 {dialogs} 段，候选预测点 {kept}", flush=True)
    print(f"对话 {dialogs}，候选预测点 {kept}，写入 {dest}")
    if drop_reasons:
        print("规则丢弃次数：", dict(drop_reasons))
    return dest


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_for(candidate: dict) -> str:
    history = []
    for turn in candidate["turns"]:
        if turn["turn_id"] > candidate["assistant_response_turn_id"]:
            break
        speaker = "用户" if turn["role"] == "user" else "助手"
        history.append(f"{speaker}: {turn['content']}")
    return (
        f"主题: {candidate['topic']}\n"
        f"当前要标注的用户轮次 turn_id={candidate['target_user_turn_id']}\n\n"
        "对话（只到这一轮助手回复）：\n"
        + "\n".join(history)
    )


def parse_model_json(text: str) -> dict:
    body = (text or "").strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("模型输出不是 JSON object")
    return data


def quote_in(haystack: str, quote: str) -> bool:
    needle = compact(quote).strip("\"'「」『』")
    if len(needle) < 4:
        return False
    return needle in compact(haystack)


def validate_label(candidate: dict, label: dict, min_confidence: float) -> str | None:
    if label.get("keep") is not True:
        return label.get("drop_reason") or "model_drop"
    if label.get("reply_style") == "advice_dump":
        return "advice_dump"
    emotion = label.get("emotion_label")
    if emotion not in EMOTIONS:
        return "bad_emotion"
    try:
        confidence = float(label.get("confidence"))
    except (TypeError, ValueError):
        return "bad_confidence"
    if confidence < min_confidence:
        return "low_confidence"
    if not quote_in(candidate["user_text"], str(label.get("emotion_evidence") or "")):
        return "emotion_evidence"
    if emotion in CONFUSABLE and len(compact(str(label.get("not_anxiety_reason") or ""))) < 4:
        return "missing_contrast"
    return None


def clean_profile(candidate: dict, label: dict) -> dict:
    raw = label.get("user_profile") or {}
    evidence = label.get("profile_evidence") or {}
    if not isinstance(raw, dict):
        raw = {}
    if not isinstance(evidence, dict):
        evidence = {}
    allowed = {
        "personality_traits": TRAITS,
        "interests": INTERESTS,
        "style": STYLES,
    }
    cleaned = {}
    for key, vocab in allowed.items():
        values = []
        for item in raw.get(key) or []:
            if item not in vocab or item in values:
                continue
            if quote_in(candidate["history_user_text"], str(evidence.get(item) or "")):
                values.append(item)
        cleaned[key] = values
    return cleaned


def call_chat(base_url: str, api_key: str, model: str, user_prompt: str) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    last_error = None
    for use_json in (True, False):
        body = dict(payload)
        if use_json:
            body["response_format"] = {"type": "json_object"}
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"].get("content") or ""
                if not content:
                    raise RuntimeError(f"空回复: {result}")
                return content
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code == 400 and use_json:
                    break
                if exc.code in (429, 500, 502, 503) and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(last_error) from exc
            except (urllib.error.URLError, TimeoutError, KeyError, RuntimeError) as exc:
                last_error = str(exc)
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(last_error) from exc
    raise RuntimeError(last_error or "请求失败")


def label_candidate(candidate: dict, base_url: str, api_key: str, model: str, min_confidence: float) -> dict:
    content = call_chat(base_url, api_key, model, prompt_for(candidate))
    label = parse_model_json(content)
    reason = validate_label(candidate, label, min_confidence)
    profile = clean_profile(candidate, label) if reason is None else {
        "personality_traits": [], "interests": [], "style": [],
    }
    record = {
        "candidate_id": candidate["candidate_id"],
        "dialog_id": candidate["dialog_id"],
        "topic": candidate["topic"],
        "target_user_turn_id": candidate["target_user_turn_id"],
        "assistant_response_turn_id": candidate["assistant_response_turn_id"],
        "turns": candidate["turns"],
        "user_text": candidate["user_text"],
        "assistant_text": candidate["assistant_text"],
        "accepted": reason is None,
        "reject_reason": reason or "",
        "emotion_label": label.get("emotion_label") if reason is None else "",
        "confidence": label.get("confidence") if reason is None else 0,
        "user_profile": profile,
        "model": model,
        "label": label,
    }
    return record


def stage_ping(base_url: str, api_key: str, model: str) -> None:
    content = call_chat(base_url, api_key, model, "只回复一个 JSON：{\"keep\": false, \"drop_reason\": \"ping\"}")
    print(content[:500])
    print(f"连接成功：{model} @ {base_url}")


def stage_label(out_dir: Path, base_url: str, api_key: str, model: str, max_calls: int, concurrency: int, min_confidence: float, seed: int, stop_when_rare: bool) -> None:
    candidates_path = out_dir / "candidates.jsonl"
    labeled_path = out_dir / "labeled.jsonl"
    if not candidates_path.exists():
        raise SystemExit("还没有 candidates.jsonl，先运行 --stage rules")
    candidates = load_jsonl(candidates_path)
    done = {}
    if labeled_path.exists():
        for row in load_jsonl(labeled_path):
            done[row["candidate_id"]] = row
    order = list(candidates)
    random.Random(seed).shuffle(order)
    pending = [row for row in order if row["candidate_id"] not in done][:max_calls]
    print(f"候选 {len(candidates)}，已标注 {len(done)}，本轮调用 {len(pending)}")
    if not pending:
        return

    accepted_emotions = Counter(
        row["emotion_label"] for row in done.values() if row.get("accepted") and row.get("emotion_label")
    )
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    calls = 0
    with labeled_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), concurrency):
            chunk = pending[start:start + concurrency]
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [
                    pool.submit(label_candidate, candidate, base_url, api_key, model, min_confidence)
                    for candidate in chunk
                ]
                for future in as_completed(futures):
                    try:
                        record = future.result()
                    except Exception as exc:
                        print(f"调用失败，已跳过一条：{exc}", flush=True)
                        continue
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    calls += 1
                    if record["accepted"]:
                        accepted_emotions[record["emotion_label"]] += 1
            print(f"本轮已调用 {calls}，当前接受数 {dict(accepted_emotions)}", flush=True)
            if stop_when_rare and all(accepted_emotions[name] >= QUOTA[name] for name in RARE):
                print("五类稀有情绪都已达到第四份定额，停止调用。")
                break
    print(f"标注写入 {labeled_path}")


def stage_export(out_dir: Path) -> None:
    labeled_path = out_dir / "labeled.jsonl"
    if not labeled_path.exists():
        raise SystemExit("还没有 labeled.jsonl，先运行 --stage label")
    latest = {}
    for row in load_jsonl(labeled_path):
        latest[row["candidate_id"]] = row
    grouped = defaultdict(list)
    for row in latest.values():
        if row.get("accepted") and row.get("emotion_label") in QUOTA:
            grouped[row["emotion_label"]].append(row)
    chosen = []
    summary = {}
    for emotion, quota in QUOTA.items():
        rows = sorted(grouped.get(emotion, []), key=lambda item: float(item.get("confidence") or 0), reverse=True)
        take = rows[:quota]
        chosen.extend(take)
        summary[emotion] = {"accepted": len(rows), "kept": len(take), "quota": quota}

    by_dialog = defaultdict(list)
    for row in chosen:
        by_dialog[row["dialog_id"]].append(row)
    dest = out_dir / "train.jsonl"
    with dest.open("w", encoding="utf-8") as handle:
        for dialog_id in sorted(by_dialog, key=lambda item: (isinstance(item, str), item)):
            points = sorted(by_dialog[dialog_id], key=lambda item: item["target_user_turn_id"])
            record = {
                "conversation_id": conversation_id_for(dialog_id),
                "turns": points[0]["turns"],
                "prediction_points": [
                    {
                        "target_user_turn_id": point["target_user_turn_id"],
                        "assistant_response_turn_id": point["assistant_response_turn_id"],
                        "target": {
                            "response_text": point["assistant_text"],
                            "emotion_label": point["emotion_label"],
                            "user_profile": point["user_profile"],
                            "memory_refs": [],
                        },
                    }
                    for point in points
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"对话 {len(by_dialog)}，预测点 {len(chosen)}，写入 {dest}")
    for emotion in EMOTIONS:
        item = summary[emotion]
        print(f"{emotion:16} 接受 {item['accepted']:4}  留下 {item['kept']:4}  定额 {item['quota']:4}")


def resolve_provider(args) -> tuple[str, str, str]:
    preset = PRESETS.get(args.provider, {})
    base_url = args.base_url or preset.get("base_url")
    model = args.model or preset.get("model")
    key_env = args.api_key_env or preset.get("api_key_env")
    if not base_url or not model or not key_env:
        raise SystemExit("自定义接口需要 --base-url --model --api-key-env")
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise SystemExit(f"环境变量 {key_env} 是空的。不要把密钥写进脚本。")
    return base_url, api_key, model


def main() -> None:
    parser = argparse.ArgumentParser(description="用规则和 DeepSeek/Qwen 筛选华南理工 part_4")
    parser.add_argument("--provider", choices=["deepseek", "qwen", "custom"], default="deepseek")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--stage", choices=["ping", "rules", "label", "export", "all"], required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="只扫描前 N 段，试跑用")
    parser.add_argument("--max-calls", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-when-rare", action="store_true")
    args = parser.parse_args()
    limit = args.limit or None

    if args.stage in ("ping", "label", "all"):
        base_url, api_key, model = resolve_provider(args)
    else:
        base_url = api_key = model = ""

    if args.stage == "ping":
        stage_ping(base_url, api_key, model)
        return
    if args.stage in ("rules", "all"):
        stage_rules(args.input, args.out_dir, limit)
    if args.stage in ("label", "all"):
        stage_label(
            args.out_dir, base_url, api_key, model,
            args.max_calls, args.concurrency, args.min_confidence, args.seed, args.stop_when_rare,
        )
    if args.stage in ("export", "all"):
        stage_export(args.out_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("已中断。再次运行 --stage label 会从上次写出的标注继续。")
