"""
Generates a realistic multi-page PDF financial filing (Apple Inc. FY2023 10-K excerpt)
using PyMuPDF for immediate local testing and reproducible benchmarks.
"""

from pathlib import Path
import pymupdf as fitz

OUTPUT_PDF = Path(__file__).resolve().parent / "raw" / "apple_10k_sample.pdf"

PAGES = [
    # Page 1: Title & Overview
    """UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549
FORM 10-K
ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
For the fiscal year ended September 30, 2023
Commission file number: 001-36743

APPLE INC.
(Exact name of registrant as specified in its charter)

Item 1. Business
Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories, and sells a variety of related services.
The Company's fiscal year is the 52- or 53-week period that ends on the last Saturday of September.

Key Business Highlights:
- iPhone sales continue to represent the largest portion of total net sales.
- Services revenue reached an all-time record in fiscal 2023, driven by App Store, Advertising, Cloud, and Payment services.
- Research and Development (R&D) investments increased to support silicon development and artificial intelligence integration.
""",

    # Page 2: Consolidated Statements of Operations (Table)
    """Apple Inc.
CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions, except number of shares which are reflected in thousands and per share amounts)

Years ended September 30, 2023, September 24, 2022, and September 25, 2021:

| Revenue Category | 2023 | 2022 | 2021 |
|---|---|---|---|
| Products Net Sales | $ 298,085 | $ 316,199 | $ 297,392 |
| Services Net Sales | $ 85,200 | $ 78,129 | $ 68,425 |
| Total Net Sales | $ 383,285 | $ 394,328 | $ 365,817 |
| Cost of Sales: Products | $ 180,683 | $ 201,471 | $ 192,266 |
| Cost of Sales: Services | $ 33,454 | $ 22,075 | $ 20,715 |
| Total Cost of Sales | $ 214,137 | $ 223,546 | $ 212,981 |
| Gross Margin | $ 169,148 | $ 170,782 | $ 152,836 |
| Research and Development | $ 29,915 | $ 26,251 | $ 21,914 |
| Selling, General & Administrative | $ 24,932 | $ 25,094 | $ 21,973 |
| Total Operating Expenses | $ 54,847 | $ 51,345 | $ 43,887 |
| Operating Income | $ 114,301 | $ 119,437 | $ 108,949 |
| Net Income | $ 96,995 | $ 99,803 | $ 94,680 |
""",

    # Page 3: Segment & Product Performance Breakdown
    """Apple Inc.
MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS

Net Sales by Product Category:
- iPhone: Net sales were $200,583 million in 2023 compared to $205,489 million in 2022, representing a 2% decrease primarily due to foreign exchange headwinds and lower demand in certain international markets.
- Mac: Net sales were $29,357 million in 2023 compared to $40,177 million in 2022, a 27% decline reflecting challenging market conditions and tough comparison against previous M-series launches.
- iPad: Net sales were $28,300 million in 2023 compared to $29,292 million in 2022, a 3% decrease.
- Wearables, Home and Accessories: Net sales were $39,845 million in 2023 compared to $41,241 million in 2022, a 3% decrease.
- Services: Net sales were $85,200 million in 2023 compared to $78,129 million in 2022, an increase of 9% driven by growth across customer accounts and subscriptions.

Total net sales decreased 3% or $11,043 million during 2023 compared to 2022.
""",

    # Page 4: Risk Factors (Item 1A)
    """Apple Inc.
Item 1A. Risk Factors

Global Supply Chain and Manufacturing Risks:
The Company's business and results of operations can be adversely affected by disruptions to manufacturing, logistics, and supply chain partners located primarily in Asia. Component shortages, customs disruptions, or geopolitical tensions could result in inventory shortages and delayed product delivery.

Competition and Technological Evolution:
The markets for the Company's products and services are highly competitive and characterized by rapid technological change, evolving industry standards, and frequent new product introductions. The Company must continue to invest significantly in Research and Development to maintain competitive advantages.

Foreign Exchange Rate Volatility:
Because the Company derives a substantial portion of its revenues from international operations, fluctuations in the value of foreign currencies relative to the U.S. dollar can adversely affect reported net sales and operating margins.
"""
]


def generate_sample_pdf():
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    for text in PAGES:
        page = doc.new_page()
        # Insert formatted text in page
        rect = fitz.Rect(50, 50, 550, 750)
        page.insert_textbox(rect, text, fontsize=10, fontname="helv", align=fitz.TEXT_ALIGN_LEFT)

    doc.save(str(OUTPUT_PDF))
    doc.close()
    print(f"Sample financial PDF generated: {OUTPUT_PDF.name} ({OUTPUT_PDF.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    generate_sample_pdf()
