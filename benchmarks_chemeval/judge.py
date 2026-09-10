"""LLM-as-judge（自由記述タスクの採点）。

ChemEval では空所補充・短答・計算・要約生成・提綱生成・合成経路推薦・反応中間体推導
などが LLM 採点（公式 `Textual/LLM evaluate/`）になっている。ここでは同じ役割を
「問題文 + 正解 + エージェントの答え」を渡して 0..10 点を返させる 1 ターンの
問い合わせで実装する。SDK が無い / API キーが無い場合は `available=False` になり、
該当タスクは集計対象外（score=None）になる。

判定結果は results/<label>/judge_cache.json にキャッシュするので、再採点
（`evaluate score`）では API を呼び直さない。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

PROMPT_TEMPLATE = """あなたは化学分野の専門家で、採点者です。以下の問題に対する解答を、
参照解答と比較して 0〜10 点で採点してください。

採点基準:
- 10: 参照解答と科学的に等価。必要な要素（数値・条件・段階・結論）がすべて正しい。
- 7-9: 主要な内容は正しいが、細部（単位・補足条件・一部の段階）に不足や誤りがある。
- 4-6: 部分的に正しいが、重要な要素の誤りや欠落がある。
- 1-3: ほとんど誤り。方向性のみ合っている。
- 0: 完全に誤り、無回答、または問題に答えていない。

表現・言語・書式の違いは減点しないこと（内容のみで判断する）。

# 問題
{query}

# 参照解答
{target}

# 採点対象の解答
{answer}

JSON だけを出力してください: {{"score": <0-10 の数値>, "reason": "<1 文の理由>"}}
"""

SYSTEM_PROMPT = ("あなたは厳格で一貫した化学の採点者です。指示された JSON のみを出力し、"
                 "説明文やコードフェンスを付けないでください。")


def cache_key(task_id: str, query: str, target: str, answer: Any) -> str:
    payload = json.dumps([task_id, query, target, str(answer)], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def parse_judgement(text: str) -> dict | None:
    """判定テキスト → {"score": 0..1, "reason": str}。"""
    if not text:
        return None
    for block in re.findall(r"\{.*?\}", text, flags=re.DOTALL)[::-1]:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "score" in parsed:
            try:
                raw = float(parsed["score"])
            except (TypeError, ValueError):
                continue
            return {"score": max(0.0, min(1.0, raw / 10.0)),
                    "raw_score": raw, "reason": str(parsed.get("reason", ""))[:500]}
    found = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:/\s*10|点)", text)
    if found:
        raw = float(found.group(1))
        return {"score": max(0.0, min(1.0, raw / 10.0)), "raw_score": raw, "reason": ""}
    # 長めの reason を書いている途中で出力が打ち切られると JSON が閉じず、
    # 上のブロック抽出では拾えない。score 自体は先頭に出ているので
    # キー名として拾う（judge が実際に付けた点を回収するだけで、基準は緩めない）。
    found = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if found:
        raw = float(found.group(1))
        reason = re.search(r'"reason"\s*:\s*"(.*)', text, flags=re.DOTALL)
        return {"score": max(0.0, min(1.0, raw / 10.0)), "raw_score": raw,
                "reason": (reason.group(1)[:500] if reason else "") + "（出力が打ち切られたため score のみ回収）"}
    return None


class Judge:
    """claude-agent-sdk（既定）または anthropic SDK 経由の 1 ターン採点。"""

    def __init__(self, provider: str = "auto", model: str | None = None,
                 cache_path: Path | None = None, concurrency: int = 4):
        self.model = model
        self.cache_path = Path(cache_path) if cache_path else None
        self.concurrency = max(1, concurrency)
        self.cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.provider = self._resolve(provider)
        self.unavailable_reason = "" if self.provider else "judge に使える SDK がありません"

    # --- provider 解決 --------------------------------------------------

    @staticmethod
    def _has(module: str) -> bool:
        import importlib.util
        return importlib.util.find_spec(module) is not None

    def _resolve(self, provider: str) -> str | None:
        if provider in ("none", "off", ""):
            return None
        if provider in ("auto", "claude") and self._has("claude_agent_sdk"):
            return "claude"
        if provider in ("auto", "anthropic") and self._has("anthropic"):
            return "anthropic"
        if provider not in ("auto",):
            return None
        return None

    @property
    def available(self) -> bool:
        return self.provider is not None

    # --- 実行 -----------------------------------------------------------

    async def _ask_claude_sdk(self, prompt: str) -> str:
        from claude_agent_sdk import ClaudeAgentOptions, query

        # max_turns=1 だと、複雑な分子で採点理由が長くなったときに SDK が
        # `Reached maximum number of turns (1)` をエラーとして返し、採点が
        # 落ちる（フル評価で実際に 3 件が未採点になった）。ツールは一切
        # 許可していないので 2 にしても往復が増えるだけで暴走はしない。
        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            model=self.model,
            allowed_tools=[],
            max_turns=2,
        )
        text = ""
        try:
            async for message in query(prompt=prompt, options=options):
                kind = type(message).__name__
                if kind == "AssistantMessage":
                    for block in getattr(message, "content", []) or []:
                        if type(block).__name__ == "TextBlock":
                            text += getattr(block, "text", "")
                elif kind == "ResultMessage":
                    text = getattr(message, "result", "") or text
        except Exception:
            # ターン上限などで打ち切られても、それまでに受け取った判定文は捨てない。
            # 判定は先頭に {"score": N} を書くので parse_judgement が回収できる
            # （回収できなければ従来どおり例外として扱う）。
            if not text:
                raise
        return text

    async def _ask_anthropic(self, prompt: str) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic()
        message = await client.messages.create(
            model=self.model or "claude-sonnet-5",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(block, "text", "") for block in message.content)

    async def judge_one(self, task_id: str, query_text: str, target: str, answer: Any) -> dict:
        key = cache_key(task_id, query_text, target, answer)
        if key in self.cache:
            return self.cache[key]
        if not self.available:
            return {"score": None, "reason": self.unavailable_reason}
        prompt = PROMPT_TEMPLATE.format(query=query_text[:8000], target=str(target)[:4000],
                                        answer=str(answer)[:4000])
        try:
            if self.provider == "claude":
                text = await self._ask_claude_sdk(prompt)
            else:
                text = await self._ask_anthropic(prompt)
        except Exception as e:                       # 採点の失敗で評価全体を落とさない
            return {"score": None, "reason": f"judge 失敗: {type(e).__name__}: {e}"}
        result = parse_judgement(text) or {"score": None,
                                           "reason": f"判定を解釈できません: {text[:200]}"}
        self.cache[key] = result
        return result

    async def judge_many(self, requests: list[dict]) -> dict[str, dict]:
        """requests = [{"item_id", "task_id", "query", "target", "answer"}]。"""
        semaphore = asyncio.Semaphore(self.concurrency)

        async def one(request: dict) -> tuple[str, dict]:
            async with semaphore:
                result = await self.judge_one(request["task_id"], request["query"],
                                              request["target"], request["answer"])
            return request["item_id"], result

        done = await asyncio.gather(*(one(r) for r in requests))
        self.save()
        return dict(done)

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
