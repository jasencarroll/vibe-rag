---
name: e2e-test
description: Run browser-based E2E tests using Playwright MCP
license: MIT
user-invocable: true
allowed-tools:
  - read_file
  - grep
  - glob
  - run_command
  - ask_user_question
  - playwright_browser_navigate
  - playwright_browser_snapshot
  - playwright_browser_click
  - playwright_browser_fill_form
  - playwright_browser_take_screenshot
  - playwright_browser_evaluate
  - playwright_browser_wait_for
  - playwright_browser_press_key
  - playwright_browser_select_option
  - playwright_browser_tabs
  - playwright_browser_close
  - playwright_browser_console_messages
  - playwright_browser_network_requests
---

# E2E Test

Run browser-based end-to-end tests using Playwright.

## Workflow

1. Ask the user what to test if not specified (URL, user flow, expected behavior)
2. Navigate to the target URL with `playwright_browser_navigate`
3. Take a snapshot with `playwright_browser_snapshot` to understand the page
4. Execute the test scenario:
   - Fill forms with `playwright_browser_fill_form`
   - Click buttons/links with `playwright_browser_click`
   - Wait for navigation/loading with `playwright_browser_wait_for`
   - Take screenshots at key steps with `playwright_browser_take_screenshot`
5. Verify results:
   - Check page content via snapshots
   - Check console for errors with `playwright_browser_console_messages`
   - Check network requests with `playwright_browser_network_requests`
6. Report: pass/fail, screenshots taken, any errors found
7. Close browser with `playwright_browser_close`

## Guidelines

- Always snapshot before interacting to understand current page state
- Use accessibility snapshots (not vision) for reliable element targeting
- Screenshot on failures for debugging
- Check console messages for JS errors after each navigation
- Report both what passed AND what failed
