import sys
sys.path.insert(0, r'f:\both_investor')

# Simulate fetching data for TCS
from app2 import fetch_deep_stock_data

print("Testing fetch_deep_stock_data with 'TCS'...")
result = fetch_deep_stock_data("TCS")

if result:
    print(f"✓ Success! Fetched data for {result['symbol']}")
    print(f"  - Info keys: {len(result['info'])}")
    print(f"  - History shape: {result['history'].shape}")
    print(f"  - News count: {len(result['news'])}")
    print(f"  - Has income stmt: {result['income_stmt'] is not None}")
    print(f"  - Has balance sheet: {result['balance_sheet'] is not None}")
    print(f"  - Has cash flow: {result['cash_flow'] is not None}")
else:
    print("✗ Failed to fetch data")
