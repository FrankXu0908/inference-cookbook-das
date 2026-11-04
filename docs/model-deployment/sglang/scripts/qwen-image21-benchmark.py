import argparse
import base64
import concurrent.futures
import json
from pathlib import Path
import statistics
import struct
import threading
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description="Benchmark compatible Qwen-Image-2.1 HTTP requests.")
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen-Image-2.1")
    parser.add_argument("--size", choices=["1280x736", "1920x1088", "2560x1440", "3840x2176", "1024x1024", "1280x768"], default="1280x736")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("qwen-image21-benchmark"))
    args = parser.parse_args()
    if not 1 <= args.concurrency <= args.max_batch_size:
        parser.error("concurrency must be between 1 and the server's configured max-batch-size")
    if args.requests < args.concurrency or args.warmup_rounds < 0:
        parser.error("requests must be >= concurrency; warmup-rounds must be >= 0")
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "prompt": "一只红色陶瓷茶壶放在木桌上，窗边自然光，写实摄影，细节清晰。",
        "size": args.size,
        "n": 1,
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "seed": 42,
        "response_format": "b64_json",
    }
    encoded = json.dumps(payload).encode()
    (args.output / "request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def run_round(count, save_image=False):
        gate = threading.Barrier(args.concurrency + 1)

        def request(index):
            if index < args.concurrency:
                gate.wait()
            start = time.perf_counter()
            try:
                req = urllib.request.Request(
                    args.url.rstrip("/") + "/v1/images/generations",
                    data=encoded,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=3600) as response:
                    status = response.status
                    data = json.load(response)
                elapsed = time.perf_counter() - start
                if status != 200 or len(data.get("data", [])) != 1:
                    raise RuntimeError(f"Unexpected response: {data}")
                item = data["data"][0]
                png = base64.b64decode(item.pop("b64_json"))
                if png[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", png[16:24]) != tuple(map(int, args.size.split("x"))):
                    raise RuntimeError(f"Expected a {args.size} PNG")
                item["image_bytes"] = len(png)
                if save_image and index == 0:
                    (args.output / "output.png").write_bytes(png)
                return {"index": index, "elapsed_s": elapsed, "http_status": status, "response": data}
            except Exception as error:
                return {"index": index, "elapsed_s": time.perf_counter() - start, "error": str(error)}

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(request, index) for index in range(count)]
            start = time.perf_counter()
            gate.wait()
            rows = [future.result() for future in concurrent.futures.as_completed(futures)]
            duration = time.perf_counter() - start
        return sorted(rows, key=lambda row: row["index"]), duration

    warmups = []
    for _ in range(args.warmup_rounds):
        rows, _ = run_round(args.concurrency)
        warmups.extend(rows)
        (args.output / "warmup.json").write_text(json.dumps(warmups, ensure_ascii=False, indent=2))
        if any("error" in row for row in rows):
            raise SystemExit("Warmup failed; inspect warmup.json and the server log")
    rows, duration = run_round(args.requests, save_image=True)
    (args.output / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    successes = [row for row in rows if "error" not in row]
    latencies = [row["elapsed_s"] for row in successes]
    summary = {
        "size": args.size,
        "concurrency": args.concurrency,
        "configured_max_batch_size": args.max_batch_size,
        "requests": args.requests,
        "successes": len(successes),
        "failures": len(rows) - len(successes),
        "duration_s": duration,
        "images_per_s": len(successes) / duration,
        "mean_s": statistics.mean(latencies) if latencies else None,
        "median_s": statistics.median(latencies) if latencies else None,
        "inference_mean_s": statistics.mean(row["response"]["inference_time_s"] for row in successes)
            if successes and all(isinstance(row["response"].get("inference_time_s"), (int, float)) for row in successes) else None,
        "min_s": min(latencies) if latencies else None,
        "max_s": max(latencies) if latencies else None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
