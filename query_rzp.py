import os, httpx, time
from app.config import get_settings
s = get_settings()
print("Querying reference_id=roe_rd_1787726095")
r = httpx.get("https://api.razorpay.com/v1/payment_links/", params={"reference_id": "roe_rd_1787726095"}, auth=(s.razorpay_key_id, s.razorpay_key_secret))
print("Params match:", r.json())
print("\nQuerying all")
r2 = httpx.get("https://api.razorpay.com/v1/payment_links/", params={"count": 5}, auth=(s.razorpay_key_id, s.razorpay_key_secret))
for item in r2.json().get("items", []):
    print("Link id:", item.get("id"), "Ref id:", item.get("reference_id"))
