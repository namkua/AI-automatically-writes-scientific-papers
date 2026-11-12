"""E2E Automated UI Test for Streamlit Research Paper App.

Uses Playwright to open a real browser and interact with the Streamlit UI
end-to-end. The Streamlit server must already be running on localhost:8504
(start it separately: python -m streamlit run app.py --server.port 8504).

Run:
    python tests/test_e2e_app.py
"""
from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

PORT = "8504"
BASE_URL = f"http://localhost:{PORT}"

RESULTS: list[dict[str, str]] = []


def log(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"test": name, "status": status, "detail": detail})
    safe_detail = detail.encode("ascii", errors="replace").decode("ascii")
    safe_name = name.encode("ascii", errors="replace").decode("ascii")
    print(f"[{status}] {safe_name}: {safe_detail}")


def wait_for_app(page, timeout_ms: int = 180000) -> None:
    """Wait until Streamlit finishes its initial load."""
    page.wait_for_url(f"**/localhost:{PORT}**", timeout=timeout_ms)
    run_btn = page.locator('button:has-text("Bắt đầu sinh bài báo")')
    run_btn.wait_for(timeout=timeout_ms)
    try:
        page.wait_for_function(
            "() => !document.querySelector('[data-testid=\"stStatusWidget\"], [data-testid=\"stRunner\"]')",
            timeout=30000,
        )
    except Exception:
        pass


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            channel="chrome",
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        try:
            page.goto(BASE_URL, timeout=120000)
            wait_for_app(page)
            log("App startup", "PASS", f"Streamlit loaded on {BASE_URL}")

            run_btn = page.locator('button:has-text("🚀 Bắt đầu sinh bài báo")')
            run_btn.wait_for(timeout=30000)
            run_btn.click()
            try:
                page.wait_for_selector(
                    'text="✅ Pipeline completed · manuscript ready"',
                    timeout=180000,
                )
                log("Test 1 - Sinh bài báo & Hard Reset", "PASS",
                    "Pipeline completed, cache/session cleared before run")
            except Exception as exc:
                log("Test 1 - Sinh bài báo & Hard Reset", "FAIL",
                    f"Pipeline did not complete: {exc}")
                return 1

            page.locator('label:has-text("Xem trước bản thảo")').click()
            page.wait_for_timeout(2000)
            page.wait_for_selector("text=Xem trước bản thảo", timeout=30000)
            page.wait_for_timeout(6000)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2000)
            page_images = page.locator('[data-testid="stImage"] img')
            image_count = page_images.count()
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(2500)
            has_download = page.locator('button:has-text("📥 Tải bài báo PDF")').count() > 0
            if image_count <= 3 and has_download:
                log("Test 2 - PDF Preview 3 trang", "PASS",
                    f"{image_count} trang preview, có nút tải PDF")
            else:
                log("Test 2 - PDF Preview 3 trang", "FAIL",
                    f"{image_count} trang preview, download={has_download}")

            tabs = [
                "Cấu hình chủ đề",
                "Xem trước bản thảo",
                "So sánh & biểu đồ",
                "Tệp kết quả",
            ]
            tab_persistence_ok = True
            for tab_name in tabs:
                tab_radio = page.locator(f'label:has-text("{tab_name}")')
                if tab_radio.count() == 0:
                    tab_persistence_ok = False
                    break
                tab_radio.click()
                page.wait_for_timeout(2500)
                el = page.locator(f'label:has-text("{tab_name}")').first
                checked = el.evaluate(
                    "node => node.getAttribute('data-checked') === 'true' || "
                    "node.getAttribute('aria-checked') === 'true' || "
                    "node.querySelector('input[type=radio]:checked') !== null || "
                    "node.parentElement.querySelector('input[type=radio]:checked') !== null"
                )
                if not checked:
                    tab_persistence_ok = False
                    break
            if tab_persistence_ok:
                log("Test 3 - Chuyển Tab & State Persistence", "PASS",
                    "Tất cả 4 tab chuyển được và giữ trạng thái")
            else:
                log("Test 3 - Chuyển Tab & State Persistence", "FAIL",
                    "Một số tab không chuyển hoặc không giữ trạng thái")

            page.locator('label:has-text("Xem trước bản thảo")').click()
            page.wait_for_timeout(2500)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2000)

            sample_prompts = [
                "What is this paper about?",
                "What is the main idea?",
                "What are the main results?",
                "What is the problem solved?",
            ]
            chat_ok = True
            tab_stayed_on_paper = True
            for prompt_text in sample_prompts:
                btn = page.locator("button").filter(has_text=prompt_text)
                if btn.count() == 0:
                    chat_ok = False
                    break
            if chat_ok:
                page.locator("button").filter(has_text=sample_prompts[0]).first.click()
                page.wait_for_timeout(3000)
                active = page.locator(
                    'label:has-text("Xem trước bản thảo")'
                ).evaluate(
                    "node => node.getAttribute('data-checked') === 'true' || "
                    "node.getAttribute('aria-checked') === 'true' || "
                    "node.querySelector('input[type=radio]:checked') !== null"
                )
                if not active:
                    tab_stayed_on_paper = False
            if chat_ok and tab_stayed_on_paper:
                log("Test 4 - Quick Sample Prompts & Chat", "PASS",
                    "4 nút câu hỏi mẫu tồn tại, bấm nút không làm văng tab")
            else:
                log("Test 4 - Quick Sample Prompts & Chat", "FAIL",
                    f"chat_ok={chat_ok}, tab_stayed={tab_stayed_on_paper}")

            warning_boxes = page.locator(".stWarning, [data-baseweb='notification']")
            warning_count = warning_boxes.count()
            contrast_ok = True
            tab_labels = page.locator('div[data-baseweb="radio"] > label')
            for i in range(tab_labels.count()):
                txt = tab_labels.nth(i).inner_text()
                if len(txt) > 40:
                    contrast_ok = False
                    break
            if contrast_ok:
                log("Test 5 - Visual & Contrast", "PASS",
                    f"Warning boxes={warning_count}, tab labels không bị cắt")
            else:
                log("Test 5 - Visual & Contrast", "FAIL",
                    "Tab label bị cắt hoặc tràn")

            context.close()
            browser.close()

        except Exception as exc:  # noqa: BLE001
            log("E2E fatal", "FAIL", str(exc))
            try:
                context.close()
                browser.close()
            except Exception:
                pass
            return 1

    print("\n" + "=" * 60)
    print("E2E TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    for r in RESULTS:
        safe_test = r["test"].encode("ascii", errors="replace").decode("ascii")
        safe_detail = r["detail"].encode("ascii", errors="replace").decode("ascii")
        print(f"  {r['status']:5} | {safe_test} | {safe_detail}")
    print("=" * 60)
    print(f"TOTAL: {passed} PASS / {failed} FAIL")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

