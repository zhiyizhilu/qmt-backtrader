import asyncio
from playwright.async_api import async_playwright

async def screenshot():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': 750, 'height': 1000})
        await page.goto('file:///E:/jupyter%20notebook/automatic/qmt_backtrader/promo_ai_research.html')
        await asyncio.sleep(2)
        await page.screenshot(
            path='E:/jupyter notebook/automatic/qmt_backtrader/promo_screenshots/ai_research.png',
            full_page=False,
            clip={'x': 0, 'y': 0, 'width': 750, 'height': 1000}
        )
        await browser.close()
        print('Done: ai_research.png')

asyncio.run(screenshot())
