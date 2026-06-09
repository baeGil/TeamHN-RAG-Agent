# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "beautifulsoup4",
#     "cloudscraper",
#     "playwright",
# ]
# ///

import os
import time
import cloudscraper
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

URL = "https://vinfastauto.com/vn_vi/tai-lieu-o-to"
FOLDER_NAME = "vinfast_tai_lieu/xe_hoi"
os.makedirs(FOLDER_NAME, exist_ok=True)

def download_pdf(scraper, pdf_url, filename):
    try:
        response = scraper.get(pdf_url, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(FOLDER_NAME, filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f" -> [Thành công] Đã tải: {filename}")
        else:
            print(f" -> [Thất bại] Lỗi mã {response.status_code} khi tải {filename}")
    except Exception as e:
        print(f" -> [Lỗi] Không thể tải {filename}: {e}")

def main():
    print("==========================================================")
    print("Khởi chạy công cụ cào tài liệu VinFast sử dụng Playwright")
    print("==========================================================")
    
    scraper = cloudscraper.create_scraper(browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    })

    pdf_links = []

    print("Đang khởi tạo trình duyệt ngầm (Playwright)...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        
        page = context.new_page()
        print(f"Đang truy cập trang web: {URL}...")
        
        try:
            page.goto(URL, timeout=60000, wait_until="domcontentloaded")
            print("Đang đợi trang web xử lý JavaScript và tải danh sách tài liệu...")
            page.wait_for_timeout(5000)
            
            html_content = page.content()
            soup = BeautifulSoup(html_content, "html.parser")
            
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                if href.lower().endswith('.pdf') or '.pdf?' in href.lower():
                    full_url = urljoin(URL, href)
                    if full_url not in pdf_links:
                        pdf_links.append(full_url)
                        
        except Exception as e:
            print(f"Có lỗi xảy ra khi duyệt trang: {e}")
        finally:
            browser.close()

    total_files = len(pdf_links)
    print(f"\nTìm thấy tổng cộng: {total_files} file PDF.")

    if total_files == 0:
        return

    print(f"Bắt đầu tải các file vào thư mục '{FOLDER_NAME}'...")
    for idx, pdf_url in enumerate(pdf_links, 1):
        filename = pdf_url.split('/')[-1]
        if '?' in filename:
            filename = filename.split('?')[0]
            
        print(f"[{idx}/{total_files}] Đang tải: {filename}...")
        download_pdf(scraper, pdf_url, filename)
        time.sleep(2)

    print("\n==========================================================")
    print("Hoàn thành! Toàn bộ file đã được lưu trong thư mục.")
    print("==========================================================")

if __name__ == "__main__":
    main()