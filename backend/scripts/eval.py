"""评测脚本：跑固定问题集，量化「答对率 / 命中页 / 延迟」。

用法：
    python scripts/eval.py                      # 默认跑全部，连 http://127.0.0.1:8000
    python scripts/eval.py --base http://127.0.0.1:8010
    python scripts/eval.py --category vocab     # 只跑词条题
    python scripts/eval.py --limit 5            # 只跑前 5 条（快速冒烟）

用例在 backend/eval/cases.json，报告写到 backend/eval/report.md。
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List

BACKEND_DIR = Path(__file__).resolve().parent.parent
CASES_PATH = BACKEND_DIR / "eval" / "cases.json"
REPORT_PATH = BACKEND_DIR / "eval" / "report.md"


def ask(base: str, question: str) -> Dict:
    payload = json.dumps(
        {"question": question, "history": [], "mode": "auto"}, ensure_ascii=False
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base.rstrip('/')}/api/query",
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def evaluate(case: Dict, base: str) -> Dict:
    started = time.perf_counter()
    error = ""
    data: Dict = {}
    try:
        data = ask(base, case["question"])
    except Exception as exc:  # 网络 / 500 / 超时
        error = str(exc)
    elapsed = round(time.perf_counter() - started, 2)

    answer = data.get("answer", "")
    pages = [source.get("page") for source in data.get("sources", [])]
    missing = [
        keyword
        for keyword in case.get("expect_keywords", [])
        if keyword.lower() not in answer.lower()
    ]
    page_ok = True
    if case.get("expect_page"):
        page_ok = case["expect_page"] in pages

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "passed": not error and not missing and page_ok,
        "missing": missing,
        "page_ok": page_ok,
        "expect_page": case.get("expect_page"),
        "pages": pages,
        "seconds": elapsed,
        "error": error,
        "answer": answer[:160].replace("\n", " "),
    }


def write_report(results: List[Dict], base: str) -> None:
    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    latencies = sorted(item["seconds"] for item in results)
    p50 = statistics.median(latencies) if latencies else 0.0

    categories: Dict[str, List[Dict]] = {}
    for item in results:
        categories.setdefault(item["category"], []).append(item)

    lines = [
        "# PersonalRAG 评测报告",
        "",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 服务：{base}",
        f"- 用例：{total}，通过：{passed}，通过率：{passed / total:.0%}" if total else "- 无用例",
        f"- 延迟中位数：{p50:.2f}s",
        "",
        "## 分类",
        "",
        "| 类别 | 通过 / 总数 | 通过率 |",
        "| --- | --- | --- |",
    ]
    for name, items in sorted(categories.items()):
        ok = sum(1 for item in items if item["passed"])
        lines.append(f"| {name} | {ok} / {len(items)} | {ok / len(items):.0%} |")

    lines += [
        "",
        "## 明细",
        "",
        "| 结果 | 用例 | 问题 | 耗时 | 页码 | 备注 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        mark = "✅" if item["passed"] else "❌"
        notes = []
        if item["missing"]:
            notes.append("缺关键词：" + "、".join(item["missing"]))
        if not item["page_ok"]:
            notes.append(f"期望第 {item['expect_page']} 页，实际 {item['pages']}")
        if item["error"]:
            notes.append("错误：" + item["error"][:60])
        question = item["question"].replace("|", "\\|")
        lines.append(
            f"| {mark} | {item['id']} | {question} | {item['seconds']}s | "
            f"{item['pages']} | {'；'.join(notes) or '-'} |"
        )

    content = "\n".join(lines) + "\n"
    try:
        REPORT_PATH.write_text(content, "utf-8")
        print(f"报告：{REPORT_PATH}")
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / "personalrag_eval_report.md"
        fallback.write_text(content, "utf-8")
        print(f"无法写入 {REPORT_PATH}（{exc}），报告已改写到：{fallback}")


def clear_cache(base: str) -> None:
    """清空问答缓存，保证测到的是真实延迟而不是缓存命中。"""
    try:
        request = urllib.request.Request(f"{base.rstrip('/')}/api/cache/clear", method="POST")
        urllib.request.urlopen(request, timeout=10).read()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="PersonalRAG 评测")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="服务地址")
    parser.add_argument("--category", default="", help="只跑某个类别：vocab / knowledge / meta")
    parser.add_argument("--limit", type=int, default=0, help="最多跑几条")
    args = parser.parse_args()

    payload = json.loads(CASES_PATH.read_text("utf-8"))
    cases = payload["cases"]
    if args.category:
        cases = [case for case in cases if case["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("没有匹配的用例")
        return

    print(f"开始评测：{len(cases)} 条，服务 {args.base}\n")
    clear_cache(args.base)
    results = []
    for index, case in enumerate(cases, start=1):
        item = evaluate(case, args.base)
        results.append(item)
        mark = "ok  " if item["passed"] else "FAIL"
        print(
            f"[{index:02d}/{len(cases)}] {mark} {item['id']:<18} "
            f"{item['seconds']:>5.2f}s  {item['question'][:30]}"
        )
        if not item["passed"]:
            if item["missing"]:
                print(f"         缺关键词：{item['missing']}")
            if not item["page_ok"]:
                print(f"         页码不符：期望 {item['expect_page']}，实际 {item['pages']}")
            if item["error"]:
                print(f"         错误：{item['error']}")
            print(f"         回答：{item['answer'][:120]}")

    write_report(results, args.base)
    passed = sum(1 for item in results if item["passed"])
    print(f"\n通过 {passed}/{len(results)}")


if __name__ == "__main__":
    main()
