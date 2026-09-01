"""Quick pre-demo API verification: POST Test Case 2 (before2/after2) to /predict.

Used during final hackathon demo preparation to confirm the running backend
responds correctly without touching any prediction logic. Offline tool only.
"""
import json
import time
import urllib.request

API = "http://127.0.0.1:8000/predict"
BASE = r"c:\Users\Eman Malik\Disaster-Damage-AI\test_images"


def post_pair(before_name: str, after_name: str) -> dict:
    boundary = "----democheckboundary"
    parts = []
    for field, name in (("before", before_name), ("after", after_name)):
        with open(rf"{BASE}\{name}", "rb") as f:
            payload = f.read()
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field}"; filename="{name}"\r\n'
                f"Content-Type: image/jpeg\r\n\r\n"
            ).encode()
            + payload
            + b"\r\n"
        )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        API,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
        elapsed = time.perf_counter() - start
    return data, elapsed


if __name__ == "__main__":
    data, elapsed = post_pair("before2.jpg", "after2.jpg")
    classes = {
        k: f"{v['percentage']:.2f}% ({v['pixels']} px)"
        for k, v in data["classes"].items()
    }
    print("success:          ", data["success"])
    print("dominant_class:   ", data["dominant_class"])
    print("prediction_quality:", data["prediction_quality"], "% (softmax confidence)")
    print("class shares:")
    for k, v in classes.items():
        print(f"  {k}: {v}")
    total = sum(c["percentage"] for c in data["classes"].values())
    print(f"class sum:          {total:.2f}%")
    print("warnings:          ", data["warnings"] or "none")
    print(f"mask base64 length: {len(data['damage_mask'])} chars")
    print(f"API round-trip:     {elapsed:.2f} s")
    # affected/unaffected as the frontend computes them
    c0 = data["classes"]["class_0"]["percentage"]
    print(f"affected area:      {100 - c0:.2f}%  |  unaffected: {c0:.2f}%")
