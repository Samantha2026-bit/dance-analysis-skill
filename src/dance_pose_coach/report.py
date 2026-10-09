from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment

from .models import AnalysisResult

REPORT_TEMPLATE = """
<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>舞蹈对比分析报告 · 深度版</title><style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#242338;background:#f5f3ff;margin:0;padding:26px}.page{max-width:1000px;margin:auto}.card{background:white;border-radius:17px;padding:22px;margin:14px 0;box-shadow:0 5px 20px #423d7814}
h1,h2{margin-top:0}.score{font-size:48px;font-weight:760;color:#6557e8}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:11px}.metric{background:#f0eeff;border-radius:13px;padding:14px}.metric b{font-size:24px;display:block;color:#584bd5;margin-top:5px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;border-bottom:1px solid #eceaf7;vertical-align:top}.bar{height:8px;background:#ebe9fa;border-radius:8px;overflow:hidden}.fill{height:100%;background:#6d5ef5}.level{display:inline-block;padding:3px 9px;border-radius:999px;font-size:12px;font-weight:700}.high{background:#ffe5e5;color:#a52b2b}.medium{background:#fff0d2;color:#855500}.note{font-size:13px;color:#666}.warn{border-left:4px solid #efab42}li{margin:.55em 0}</style></head><body><main class="page">
<section class="card"><h1>舞蹈对比分析报告 · 深度版</h1><div class="score">{{ result.overall_score }}<small>/100</small></div><p>姿态可信度 {{ (result.confidence*100)|round|int }}% · {{ "已自动采用镜像对齐" if result.mirror_used else "采用原方向对齐" }}</p></section>
<section class="grid"><div class="metric">姿态还原<b>{{ result.pose_score }}</b></div><div class="metric">节奏卡点<b>{{ result.rhythm_score }}</b></div><div class="metric">连接流畅<b>{{ result.flow_score }}</b></div><div class="metric">能量中位比<b>{{ result.metadata.get("power_energy_ratio","-") }}</b></div></section>
<section class="card"><h2>1. 优先问题与训练动作</h2><table><thead><tr><th>优先级</th><th>项目</th><th>发现与证据</th><th>重点时段</th><th>怎么练</th></tr></thead><tbody>
{% for issue in result.detailed_issues %}<tr><td><span class="level {{ 'high' if issue.level == '高' else 'medium' }}">{{ issue.level }}</span></td><td><b>{{ issue.category }}</b><br>{{ issue.title }}</td><td>{{ issue.evidence }}</td><td>{% if issue.intervals %}{% for a,b in issue.intervals %}{{ a }}–{{ b }}s{% if not loop.last %}、{% endif %}{% endfor %}{% else %}全局{% endif %}</td><td>{{ issue.suggestion }}</td></tr>{% endfor %}
</tbody></table></section>
<section class="card"><h2>2. 胳膊、框架、髋部与腿部</h2><table><thead><tr><th>部位</th><th>得分</th><th>平均角度偏差</th><th>95 分位偏差</th><th>重点时段</th></tr></thead><tbody>
{% for item in result.regions|sort(attribute="score") %}<tr><td>{{ item.name }}</td><td>{{ item.score }}<div class="bar"><div class="fill" style="width:{{ item.score }}%"></div></div></td><td>{{ item.mean_angle_error }}°</td><td>{{ item.max_angle_error }}°</td><td>{% if item.weak_intervals %}{% for a,b in item.weak_intervals %}{{ a }}–{{ b }}s{% if not loop.last %}、{% endif %}{% endfor %}{% else %}无持续高偏差{% endif %}</td></tr>{% endfor %}
</tbody></table></section>
<section class="card"><h2>3. 发力表现与动作能量</h2><p>能量中位比 {{ result.metadata.get("power_energy_ratio","-") }}；能量曲线相关 {{ result.metadata.get("power_correlation","-") }}。高差异时段：{% for a,b in result.metadata.get("power_mismatch_intervals",[]) %}{{ a }}–{{ b }}s{% if not loop.last %}、{% endif %}{% else %}无{% endfor %}。</p><p class="note">发力由关键点速度、加速度和动作重音估算，不代表真实肌肉力量或关节负荷。</p></section>
<section class="card"><h2>4. 节奏与律动</h2><p>用户/参考速度比 {{ result.tempo_ratio }}；重音相对参考偏移 {{ "%+.3f"|format(result.timing_offset_seconds) }} 秒。正值表示晚于参考，负值表示早于参考。</p><ul>{% for item in result.rhythm_issues %}<li>{{ item }}</li>{% endfor %}</ul></section>
<section class="card"><h2>5. 每 4 秒能量对比</h2><table><thead><tr><th>时段</th><th>你的能量</th><th>参考能量</th><th>归一化差异</th></tr></thead><tbody>{% for row in result.metadata.get("comparison_segments",[]) %}<tr><td>{{ row.start }}–{{ row.end }}s</td><td>{{ row.user_energy }}</td><td>{{ row.reference_energy }}</td><td>{{ row.energy_difference }}</td></tr>{% endfor %}</tbody></table></section>
<section class="card"><h2>6. 建议训练顺序</h2><ol>{% for item in result.recommendations %}<li>{{ item }}</li>{% endfor %}</ol></section>
<section class="card warn"><h2>解读边界</h2><p>这是基于 2D/弱 3D 关键点和音频节拍的训练辅助结果。多人、遮挡、运镜、不同编舞版本和不同裁剪区间会降低可比性。不是医学、康复或专业评审结论。</p><p class="note">版本 {{ result.metadata.get("version","unknown") }} · 参考来源 {{ result.metadata.get("reference_source","本地文件") }} · 后端 {{ result.metadata.get("user_pose_backend","unknown") }}/{{ result.metadata.get("reference_pose_backend","unknown") }}</p></section>
</main></body></html>
"""


def render_html(result: AnalysisResult) -> str:
    return Environment(autoescape=True).from_string(REPORT_TEMPLATE).render(result=result)


def _spans(values: list[tuple[float, float]]) -> str:
    return "、".join(f"{a:.1f}–{b:.1f}s" for a, b in values) or "全局"


def render_markdown(result: AnalysisResult) -> str:
    lines = [
        "# 舞蹈对比分析报告 · 深度版",
        "",
        f"- 总分：**{result.overall_score}/100**",
        f"- 姿态 / 节奏 / 流畅度：{result.pose_score} / {result.rhythm_score} / {result.flow_score}",
        f"- 动作重音相对参考偏移：{result.timing_offset_seconds:+.3f}s",
        f"- 姿态可信度：{result.confidence * 100:.0f}%",
        f"- 动作能量中位比：{result.metadata.get('power_energy_ratio','-')}",
        f"- 动作能量曲线相关：{result.metadata.get('power_correlation','-')}",
        "",
        "## 优先问题",
        "",
    ]
    for issue in result.detailed_issues:
        lines.extend(
            [
                f"### [{issue['level']}] {issue['category']}：{issue['title']}",
                f"- 证据：{issue['evidence']}",
                f"- 重点时段：{_spans(issue['intervals'])}",
                f"- 建议：{issue['suggestion']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 身体分区",
            "",
            "| 部位 | 得分 | 平均角度偏差 | 95 分位偏差 | 重点时段 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in sorted(result.regions, key=lambda metric: metric.score):
        lines.append(
            f"| {item.name} | {item.score} | {item.mean_angle_error}° | "
            f"{item.max_angle_error}° | {_spans(item.weak_intervals)} |"
        )
    lines.extend(["", "## 建议训练顺序", ""])
    lines.extend(f"{index}. {item}" for index, item in enumerate(result.recommendations, 1))
    lines.extend(
        [
            "",
            "> 发力指标是速度/加速度代理，不是真实肌肉力量；报告不是医学或专业评审结论。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(result: AnalysisResult, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "html": output_dir / "report.html",
        "markdown": output_dir / "report.md",
        "json": output_dir / "report.json",
    }
    paths["html"].write_text(render_html(result), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(result), encoding="utf-8")
    paths["json"].write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths
