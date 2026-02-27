from playwright.sync_api import sync_playwright, expect

def verify_telemetry_changes():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the telemetry page
        page.goto("http://localhost:3000/telemetry")

        # Wait for the chart to be visible
        chart = page.locator(".recharts-responsive-container")
        expect(chart).to_be_visible()

        # Verify table headers have scope attribute (accessibility check)
        headers = page.locator("th[scope='col']")
        count = headers.count()
        print(f"Found {count} table headers with scope='col'")
        if count < 4:
            print("Error: Expected at least 4 headers with scope='col'")

        # Verify table caption exists
        caption = page.locator("caption")
        expect(caption).to_contain_text("Detailed Telemetry Logs")
        print("Table caption found and verified.")

        # Hover over the chart to trigger tooltip
        # Note: Hovering specific coordinates on a responsive chart can be tricky,
        # but we'll try to hover the center.
        box = chart.bounding_box()
        if box:
            center_x = box['x'] + box['width'] / 2
            center_y = box['y'] + box['height'] / 2
            page.mouse.move(center_x, center_y)
            # Wait for tooltip to appear
            # The custom tooltip has class 'card' and specific text structure
            tooltip = page.locator(".recharts-tooltip-wrapper .card")
            # We might need to wait a bit or move mouse slightly if not immediately triggered
            try:
                expect(tooltip).to_be_visible(timeout=5000)
                print("Tooltip appeared.")

                # Check for enriched data content
                content = tooltip.inner_text()
                print(f"Tooltip content: {content}")

                if "Event:" in content and "Outcome:" in content:
                    print("Tooltip contains enriched data.")
                else:
                    print("Warning: Tooltip might not contain all expected fields.")
            except Exception as e:
                print(f"Tooltip verification failed or timed out: {e}")

        # Take a screenshot
        screenshot_path = "/app/telemetry_verification.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    verify_telemetry_changes()
