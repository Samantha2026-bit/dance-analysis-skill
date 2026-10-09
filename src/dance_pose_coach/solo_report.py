from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment

from .models import SoloAnalysisResult

SOLO_TEMPLATE = """
<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>单视频舞蹈深度分析</title><style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f5f3ff;color:#242338;margin:0;padding:24px}main{max-width:980px;margin:auto}
.card{background:white;padding:22px;border-radius:17px;margin:14px 0;box-shadow:0 5px 20px #423d7814}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:11px}
.metric{background:#f0eeff;border-radius:13px;padding:14px}.metric b{display:block;font-size:24px;color:#5e50dc;margin-top:5px}.level{display:inline-block;padding:3px 9px;border-radius:999px;font-size:12px;font-weight:700}
.high{background:#ffe5e5;color:#a52b2b}.medium{background:#fff0d2;color:#855500}.observe{background:#e9efff;color:#3552a2}table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
h1,h2,h3{margin-top:0}li{margin:.55em 0}.note{color:#68637b;font-size:14px}.warn{border-left:4px solid #efab42}.hero{background:linear-gradient(135deg,#fff,#ece8ff)}.issue{border-left:5px solid #7567e8}.issue h3{margin-bottom:7px}.impact{background:#f7f5ff;padding:10px 13px;border-radius:10px}.steps{padding-left:22px}.time{color:#5e50dc;font-weight:650}details{margin:12px 0}summary{cursor:pointer;font-weight:700;color:#5146bf}</style></head><body><main>
<section class="card hero"><h1>单视频舞蹈教练报告</h1><h2>先看结论</h2><p>{{ result.coach_summary }}</p><p class="note">没有参考示范，因此不提供姿态正确性或编舞还原度；下面只分析这段视频自身的律动、框架、节奏和动作质量线索。</p></section>
<section><h2>最需要先改的问题</h2>
{% for issue in result.issues[:6] %}<article class="card issue"><span class="level {{ 'high' if issue.level == '高' else 'medium' if issue.level == '中' else 'observe' }}">{{ issue.level }}</span> <b>{{ issue.category }}</b><h3>{{ loop.index }}. {{ issue.title }}</h3>
{% if issue.impact %}<p class="impact"><b>跳出来会是什么感觉：</b>{{ issue.impact }}</p>{% endif %}
<p><b>重点时段：</b><span class="time">{% if issue.intervals %}{% for a,b in issue.intervals %}{{ a }}–{{ b }}s{% if not loop.last %}、{% endif %}{% endfor %}{% else %}贯穿整段或需结合画面确认{% endif %}</span></p>
<p><b>判断依据：</b>{{ issue.evidence }}</p><p><b>训练目标：</b>{{ issue.suggestion }}</p>
<p><b>具体练法：</b></p><ol class="steps">{% for step in issue.drill_steps %}<li>{{ step }}</li>{% endfor %}</ol></article>{% endfor %}</section>
<section class="card"><h2>建议训练顺序</h2><ol><li>先练身体的上下弹性和胸腔收放，不加手臂。</li><li>再练动作终点和手臂框架，保证路线完整。</li><li>最后回到音乐，按“脚下重心 → 胸髋 → 手臂”的顺序逐层加动作。</li><li>每个问题练 3 轮：0.75× 慢练、原速定格、录像复查。</li></ol></section>
<section class="card"><h2>逐段观察</h2><table><thead><tr><th>时段</th><th>上下律动</th><th>胸腔收放</th><th>手臂框架</th><th>节奏状态</th></tr></thead><tbody>{% for row in result.segments %}<tr><td>{{ row.start }}–{{ row.end }}s</td><td>{{ '偏平' if row.vertical_bounce < 0.12 else '有变化' }}</td><td>{{ '不明显' if row.chest_pulse < 0.12 else '有起伏' }}</td><td>{{ '偏收' if row.arm_frame < 0.85 else '较打开' }}</td><td>{{ '较松' if row.median_beat_error > 0.15 else '较集中' }}</td></tr>{% endfor %}</tbody></table></section>
<section class="card"><h2>分析依据（想看数据时再展开）</h2><details><summary>律动、胸髋与框架数据</summary><div class="grid"><div class="metric">上下律动范围<b>{{ result.groove_metrics.vertical_bounce_range_torso_units }}</b></div><div class="metric">胸腔收放范围<b>{{ result.groove_metrics.chest_pulse_range_torso_units }}</b></div><div class="metric">肩髋错动范围<b>{{ result.groove_metrics.shoulder_hip_opposition_range }}</b></div><div class="metric">手臂框架 P75<b>{{ result.groove_metrics.arm_frame_p75_torso_units }}</b></div></div></details>
<details><summary>节奏、动作能量与识别质量</summary><div class="grid"><div class="metric">音乐速度<b>{{ result.estimated_tempo_bpm }} BPM</b></div><div class="metric">识别可信度<b>{{ (result.pose_confidence*100)|round|int }}%</b></div><div class="metric">紧贴节拍比例<b>{{ (result.accents_within_120ms*100)|round|int }}%</b></div><div class="metric">快速变化区间<b>{{ result.abrupt_transition_intervals|length }}</b></div></div><p class="note">“紧贴节拍”指动作重音落在音乐节拍前后约 0.12 秒内。技术数字只用于支持判断，不需要用户自行解读。</p></details>
<details><summary>左右平衡与能量分布</summary><p>手腕幅度平衡 {{ result.arm_metrics.wrist_range_balance }}/100；肘角变化平衡 {{ result.arm_metrics.elbow_range_balance }}/100；膝屈伸平衡 {{ result.lower_body_metrics.knee_range_balance }}/100。</p><p>手臂/核心髋/腿部动作能量占比：{{ (result.power_metrics.arm_motion_share*100)|round|int }}% / {{ (result.power_metrics.center_motion_share*100)|round|int }}% / {{ (result.power_metrics.leg_motion_share*100)|round|int }}%。</p></details></section>
<section class="card warn"><h2>解读边界</h2><p>单视频报告能发现平衡、节奏、幅度和连续性线索，但没有标准示范时不能判断编舞还原度。2D 关键点也不能测量真实肌肉力量、关节负荷或伤病风险。</p><p class="note">版本 {{ result.metadata.get("version","unknown") }} · {{ result.pose_backend }} · 本地处理</p></section>
</main></body></html>
"""


def render_solo_html(result: SoloAnalysisResult) -> str:
    return Environment(autoescape=True).from_string(SOLO_TEMPLATE).render(result=result)


def _spans(values: list[tuple[float, float]]) -> str:
    return "、".join(f"{a:.1f}–{b:.1f}s" for a, b in values) or "无"


def render_solo_markdown(result: SoloAnalysisResult) -> str:
    lines = [
        "# 单视频舞蹈教练报告",
        "",
        "> 无参考示范，不提供姿态正确性或还原总分。",
        "",
        "## 先看结论",
        "",
        result.coach_summary,
        "",
        "## 最需要先改的问题",
        "",
    ]
    for issue in result.issues:
        lines.extend(
            [
                f"### [{issue['level']}] {issue['category']}：{issue['title']}",
                f"- 重点时段：{_spans(issue['intervals'])}",
                f"- 跳出来会是什么感觉：{issue.get('impact') or '需要结合画面确认'}",
                f"- 判断依据：{issue['evidence']}",
                f"- 训练目标：{issue['suggestion']}",
                "- 具体练法：",
                *[f"  {index}. {step}" for index, step in enumerate(issue.get('drill_steps', []), 1)],
                "",
            ]
        )
    lines.extend(
        [
            "## 建议训练顺序",
            "",
            "1. 先练身体的上下弹性和胸腔收放，不加手臂。",
            "2. 再练动作终点和手臂框架，保证路线完整。",
            "3. 最后按脚下重心、胸髋、手臂的顺序逐层加回音乐。",
            "4. 每个问题练 3 轮：0.75× 慢练、原速定格、录像复查。",
            "",
            "## 逐段观察",
            "",
        ]
    )
    for row in result.segments:
        lines.append(
            f"- {row['start']:.1f}–{row['end']:.1f}s："
            f"上下律动{'偏平' if row['vertical_bounce'] < 0.12 else '有变化'}；"
            f"胸腔收放{'不明显' if row['chest_pulse'] < 0.12 else '有起伏'}；"
            f"手臂框架{'偏收' if row['arm_frame'] < 0.85 else '较打开'}；"
            f"节奏{'较松' if row['median_beat_error'] > 0.15 else '较集中'}。"
        )
    lines.extend(
        [
            "",
            "## 分析依据",
            "",
            f"- 姿态识别可信度：{result.pose_confidence * 100:.0f}%",
            f"- 音乐速度：{result.estimated_tempo_bpm} BPM",
            f"- 紧贴节拍的动作重音：{result.accents_within_120ms * 100:.0f}%",
            f"- 上下律动范围：{result.groove_metrics['vertical_bounce_range_torso_units']} 个躯干长度",
            f"- 胸腔收放范围：{result.groove_metrics['chest_pulse_range_torso_units']} 个躯干长度",
            f"- 快速变化时段：{_spans(result.abrupt_transition_intervals)}",
            "",
            "> 发力指标是运动学代理，不是真实肌肉力量或医学结论。",
            "",
        ]
    )
    return "\n".join(lines)


def write_solo_reports(
    result: SoloAnalysisResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "html": output_dir / "report.html",
        "markdown": output_dir / "report.md",
        "json": output_dir / "report.json",
    }
    paths["html"].write_text(render_solo_html(result), encoding="utf-8")
    paths["markdown"].write_text(render_solo_markdown(result), encoding="utf-8")
    paths["json"].write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths
