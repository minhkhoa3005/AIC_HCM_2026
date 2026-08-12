import sys
import os
import base64
import csv
import time
from datetime import datetime
from pathlib import Path

# --- 1. Thiết lập đường dẫn hệ thống ---
current_dir = Path.cwd()
while not (current_dir / "backend").exists() and current_dir.parent != current_dir:
    current_dir = current_dir.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from backend.embedding.clip_encoder import encode_text_raw
from backend.embedding.remote_index import search_remote

# --- 2. Bộ Ma Trận Kiểm Thử (4 Cấp Độ) ---
TEST_SUITE = {
    "Level 1: Nhận diện Tĩnh (Basic Scene)": [
        "A wide shot of a flooded city street",
        "A close-up shot of a person's hands",
        "A news anchor sitting at a desk in a studio"
    ],
    "Level 2: Tương tác & Hành động (Action)": [
        "A reporter holding a microphone and talking to a local resident",
        "Medical staff guiding citizens at a hospital",
        "Police officers directing traffic in the rain"
    ],
    "Level 3: Ràng buộc Đa điều kiện (Multi-constraint)": [
        "An elderly woman wearing a conical hat and carrying vegetables on a bicycle",
        "A person wearing a blue shirt typing on a laptop keyboard in a dark room",
        "Exactly two cars crashed into each other on the highway"
    ],
    "Level 4: Trừu tượng & Báo chí (Abstract/Context)": [
        "A crowded press conference with many camera flashes",
        "A graph, pie chart or data visualization on a digital screen",
        "Black and white surveillance CCTV camera footage showing a street robbery"
    ]
}

TOP_K = 5
OUTPUT_HTML = "test_report_aic_advanced.html"
OUTPUT_CSV = "test_results_aic_advanced.csv"

def get_base64_image(image_path: str) -> str:
    """Đọc file ảnh và encode sang Base64."""
    full_path = current_dir / image_path if not os.path.isabs(image_path) else Path(image_path)
    if full_path.exists():
        try:
            with open(full_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception:
            return ""
    return ""

def get_confidence_badge(score: float) -> str:
    """Đánh giá mức độ tin cậy của Score bằng Badge màu sắc."""
    if score >= 0.28:
        return "<span class='badge bg-high'>HIGH</span>"
    elif score >= 0.22:
        return "<span class='badge bg-mid'>MED</span>"
    else:
        return "<span class='badge bg-low'>LOW</span>"

def main():
    print("=" * 70)
    print("🚀 BẮT ĐẦU CHẠY BENCHMARK DỰ ÁN VIDEO SEARCH AGENT")
    print("=" * 70)
    
    start_total_time = time.time()
    csv_data = []
    category_stats = {}
    
    all_latencies = []
    all_top1_scores = []
    all_top5_scores = []

    # --- 3. Thu thập dữ liệu & Đo lường thông số ---
    query_results_data = []

    for cat_name, queries in TEST_SUITE.items():
        print(f"\n📂 [TEST CATEGORY]: {cat_name}")
        cat_latencies = []
        cat_top1_scores = []
        cat_top5_scores = []

        for q in queries:
            # Đo độ trễ từ lúc Encode Query -> Search Qdrant Cloud
            q_start = time.perf_counter()
            try:
                q_vec = encode_text_raw(q)
                results = search_remote(query_vector=q_vec, top_k=TOP_K)
            except Exception as e:
                print(f"   ❌ Lỗi khi query '{q}': {e}")
                results = []
            
            q_latency_ms = (time.perf_counter() - q_start) * 1000.0
            cat_latencies.append(q_latency_ms)
            all_latencies.append(q_latency_ms)

            top1_sc = results[0].get('score', 0.0) if results else 0.0
            top5_sc_avg = (sum(r.get('score', 0.0) for r in results) / len(results)) if results else 0.0

            cat_top1_scores.append(top1_sc)
            cat_top5_scores.append(top5_sc_avg)
            all_top1_scores.append(top1_sc)
            all_top5_scores.append(top5_sc_avg)

            print(f"   ⏱️ '{q}' | Latency: {q_latency_ms:.1f}ms | Top-1 Score: {top1_sc:.4f}")

            query_results_data.append({
                "category": cat_name,
                "query": q,
                "latency_ms": q_latency_ms,
                "results": results
            })

            # Ghi Data vào tập dữ liệu CSV
            for rank, r in enumerate(results):
                pts = float(r.get('pts_time', 0.0))
                m, s = int(pts // 60), int(pts % 60)
                csv_data.append([
                    cat_name, q, f"{q_latency_ms:.2f}", rank + 1, 
                    f"{r.get('score', 0.0):.4f}", r.get('video_id', ''), 
                    r.get('frame_id', ''), f"{pts:.2f}", f"{m:02d}:{s:02d}"
                ])

        category_stats[cat_name] = {
            "count": len(queries),
            "avg_latency": sum(cat_latencies) / max(len(cat_latencies), 1),
            "avg_top1": sum(cat_top1_scores) / max(len(cat_top1_scores), 1),
            "avg_top5": sum(cat_top5_scores) / max(len(cat_top5_scores), 1)
        }

    total_duration = time.time() - start_total_time
    avg_global_latency = sum(all_latencies) / max(len(all_latencies), 1)
    avg_global_top1 = sum(all_top1_scores) / max(len(all_top1_scores), 1)
    avg_global_top5 = sum(all_top5_scores) / max(len(all_top5_scores), 1)

    # --- 4. Tạo Báo Cáo HTML Giao Diện Dashboard Tối Tân ---
    html = f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="utf-8">
        <title>AIC 2026 - Model Evaluation Report</title>
        <style>
            :root {{
                --primary: #1e293b;
                --accent: #0284c7;
                --bg: #f8fafc;
                --card-bg: #ffffff;
            }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: var(--bg); color: #334155; margin: 0; padding: 30px; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            
            /* Header & Dashboard Cards */
            .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; }}
            .header h1 {{ margin: 0; color: var(--primary); font-size: 26px; }}
            .time-stamp {{ font-size: 13px; color: #64748b; }}
            
            .dashboard {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
            .card {{ background: var(--card-bg); padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; text-align: center; }}
            .card .val {{ font-size: 28px; font-weight: bold; color: var(--accent); margin-top: 5px; }}
            .card .title {{ font-size: 13px; color: #64748b; font-weight: 600; text-transform: uppercase; }}

            /* Tables */
            table {{ width: 100%; border-collapse: collapse; background: var(--card-bg); border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 30px; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #f1f5f9; font-size: 14px; }}
            th {{ background: var(--primary); color: white; font-weight: 600; text-transform: uppercase; font-size: 12px; letter-spacing: 0.5px; }}
            
            .cat-header {{ background: #e0f2fe; color: #0369a1; font-weight: bold; font-size: 15px; }}
            
            /* Image Matrix Grid */
            .result-cell {{ text-align: center; width: 18%; vertical-align: top; background: #fafafa; border-radius: 6px; padding: 8px; border: 1px solid #f1f5f9; }}
            .result-cell img {{ width: 100%; height: 110px; object-fit: cover; border-radius: 4px; border: 1px solid #cbd5e1; }}
            
            /* Badges & Text Formatting */
            .badge {{ padding: 3px 7px; border-radius: 4px; font-size: 10px; font-weight: bold; color: white; display: inline-block; }}
            .bg-high {{ background-color: #10b981; }}
            .bg-mid {{ background-color: #f59e0b; }}
            .bg-low {{ background-color: #ef4444; }}
            
            .metric-tag {{ font-size: 11px; color: #475569; margin-top: 4px; background: #e2e8f0; padding: 2px 5px; border-radius: 3px; display: inline-block; }}
            .query-title {{ font-weight: 600; color: #0f172a; display: block; margin-bottom: 5px; }}
            .latency-tag {{ font-size: 11px; color: #0284c7; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>📊 BÁO CÁO BENCHMARK ĐÁNH GIÁ MÔ HÌNH AI SEARCH</h1>
                    <span class="time-stamp">Thời gian thực thi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total Execution Time: {total_duration:.2f}s</span>
                </div>
            </div>

            <!-- DASHBOARD TỔNG QUAN -->
            <div class="dashboard">
                <div class="card">
                    <div class="title">Tổng Số Queries Test</div>
                    <div class="val">{len(all_latencies)}</div>
                </div>
                <div class="card">
                    <div class="title">Độ Trễ Trung Bình (ms)</div>
                    <div class="val">{avg_global_latency:.1f} <span style="font-size:14px;">ms</span></div>
                </div>
                <div class="card">
                    <div class="title">Top-1 Score Trung Bình</div>
                    <div class="val">{avg_global_top1:.4f}</div>
                </div>
                <div class="card">
                    <div class="title">Top-5 Score Trung Bình</div>
                    <div class="val">{avg_global_top5:.4f}</div>
                </div>
            </div>

            <!-- BẢNG TỔNG HỢP CHỈ SỐ THEO MỨC ĐỘ -->
            <h2>📈 Thống Kê Theo Cấp Độ (Category Metrics)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Cấp Độ Kiểm Thử (Level Category)</th>
                        <th>Số Lượng Query</th>
                        <th>Độ Trễ TB (Latency)</th>
                        <th>Avg Top-1 Similarity Score</th>
                        <th>Avg Top-5 Similarity Score</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for cat_name, stats in category_stats.items():
        html += f"""
        <tr>
            <td><b>{cat_name}</b></td>
            <td>{stats['count']}</td>
            <td><code>{stats['avg_latency']:.1f} ms</code></td>
            <td><b>{stats['avg_top1']:.4f}</b></td>
            <td>{stats['avg_top5']:.4f}</td>
        </tr>
        """
        
    html += """
                </tbody>
            </table>

            <!-- BẢNG CHI TIẾT KẾT QUẢ KÈM HÌNH ẢNH & METADATA -->
            <h2>🔍 Bảng Trực Quan Kết Quả Chi Tiết (Detailed Matrix)</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 20%;">Query & Thông Số Trả Về</th>
                        <th style="width: 16%;">Rank #1 (Top Result)</th>
                        <th style="width: 16%;">Rank #2</th>
                        <th style="width: 16%;">Rank #3</th>
                        <th style="width: 16%;">Rank #4</th>
                        <th style="width: 16%;">Rank #5</th>
                    </tr>
                </thead>
                <tbody>
    """

    curr_cat = ""
    for item in query_results_data:
        if item['category'] != curr_cat:
            curr_cat = item['category']
            html += f"<tr><td colspan='6' class='cat-header'>📂 {curr_cat}</td></tr>"

        html += f"""
        <tr>
            <td>
                <span class="query-title">"{item['query']}"</span>
                <span class="latency-tag">⚡ Latency: {item['latency_ms']:.1f} ms</span>
            </td>
        """

        results = item['results']
        for r in results:
            sc = r.get('score', 0.0)
            vid = r.get('video_id', 'N/A')
            fid = r.get('frame_id', 'N/A')
            pts = float(r.get('pts_time', 0.0))
            m, s = int(pts // 60), int(pts % 60)
            img_b64 = get_base64_image(str(r.get('path', '')))
            
            badge = get_confidence_badge(sc)
            img_tag = f"<img src='data:image/jpeg;base64,{img_b64}'/>" if img_b64 else "<div style='height:110px;background:#e2e8f0;display:flex;align-items:center;justify-content:center;font-size:11px;color:#64748b;'>N/A Image</div>"

            html += f"""
            <td class="result-cell">
                {img_tag}
                <div style="margin-top:5px;">{badge} <b>Score: {sc:.4f}</b></div>
                <div class="metric-tag">📹 {vid} | 🖼️ {fid}</div>
                <div class="metric-tag">⏱️ {m:02d}:{s:02d} ({pts:.1f}s)</div>
            </td>
            """

        # Trám ô trống nếu số kết quả trả về nhỏ hơn TOP_K
        for _ in range(TOP_K - len(results)):
            html += "<td class='result-cell' style='color:#ccc;'>N/A</td>"

        html += "</tr>"

    html += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

    # --- 5. Đóng gói & Lưu File Output ---
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Level_Category", "Query", "Latency_ms", "Rank", "Similarity_Score", "Video_ID", "Frame_ID", "PTS_Seconds", "Formatted_Time"])
        writer.writerows(csv_data)

    print("\n" + "=" * 70)
    print("✅ HOÀN TẤT CHẠY BENCHMARK DỮ LIỆU AI SEARCH!")
    print(f"📊 Dashboard Báo Cáo HTML: {Path(OUTPUT_HTML).resolve()}")
    print(f"📁 Tập Dữ Liệu Thô CSV:    {Path(OUTPUT_CSV).resolve()}")
    print("=" * 70)

if __name__ == "__main__":
    main()