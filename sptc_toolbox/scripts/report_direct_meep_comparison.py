"""Render the completed direct-Meep comparison without inventing pending results.

Reads measured comparisons and timings only.  Incomplete/stale inputs do not
create or replace report.md.  This script never launches a solver.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "sptc_toolbox/results/direct_meep_comparison_20261018"
NAMES = ("direct_h2", "direct_h1", "zero")
SHANGHAI = timezone(timedelta(hours=8), "Asia/Shanghai")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, digits=6):
    if value is None:
        return "未定义"
    if not math.isfinite(float(value)):
        raise ValueError("Non-finite report value")
    return f"{value:.{digits}g}"


def percent(value):
    return "未定义" if value is None else f"{100 * value:.5g}%"


def harmonic(rows, m):
    matches = [row for row in rows if row["harmonic"] == m]
    if len(matches) != 1:
        raise ValueError(f"Missing or duplicate harmonic {m}")
    return matches[0]


def first_pair(rows, field="complex_vector_relative_difference"):
    return "／".join(percent(harmonic(rows, m)[field]) for m in (-1, 1))


def verdict(rows, target):
    failures = [m for m in (-1, 1)
                if harmonic(rows, m)["complex_vector_relative_difference"] is None
                or harmonic(rows, m)["complex_vector_relative_difference"] > target]
    return "均达到目标" if not failures else "未达到目标：" + "、".join(f"m={m:+d}" for m in failures)


def verify_inputs(folder):
    paths = [folder / "comparison_summary.json", folder / "backend_timing.json",
             folder / "update_probe/summary.json", folder / "paper_provenance.md"]
    paths += [folder / f"{name}_{sector}/summary.json"
              for name in NAMES for sector in ("even", "odd")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return None, {"completed": False, "reason": "missing inputs", "files": missing}
    comparison, timing, probe = [read_json(path) for path in paths[:3]]
    raw = {str(path.relative_to(folder)): read_json(path) for path in paths[4:]}
    pending = [label for label, data in [("comparison", comparison), ("timing", timing),
                                       ("update_probe", probe), *raw.items()]
               if data.get("completed") is not True]
    pending += [name for name in NAMES if not comparison.get("comparisons", {}).get(name, {}).get("completed")]
    if pending:
        return None, {"completed": False, "reason": "pending measurements", "pending": pending}
    provenance = comparison["provenance"]
    for relative in [*raw, "backend_timing.json"]:
        recorded = provenance.get(relative, provenance.get(relative.replace("\\", "/")))
        if recorded is None:
            recorded = provenance.get(relative.replace("/", "\\"))
        if recorded != sha(folder / relative):
            return None, {"completed": False, "reason": "stale comparison; rerun analyzer", "file": relative}
    for name in NAMES:
        result = comparison["comparisons"][name]
        if len(result["comparisons"]) < 2 or not result["final_adjacent_window_comparison"]:
            raise ValueError(f"No independent adjacent-window control for {name}")
        end = result["comparisons"][-1]["time"]
        for sector in ("even", "odd"):
            source = raw[str(Path(f"{name}_{sector}/summary.json"))]
            if end != source["rows"][-1]["time"] or end != source["identity"]["checkpoint_times"][-1]:
                raise ValueError(f"Final checkpoint mismatch for {name}_{sector}")
    material = comparison["material_update_refinement"]
    if material is None or any(material["time"] != comparison["comparisons"][name]["comparisons"][-1]["time"]
                               for name in NAMES[:2]):
        raise ValueError("Material refinement does not use both final records")
    paper = paths[3].read_text(encoding="utf-8-sig")
    paper_rows = [line for line in paper.splitlines() if line.startswith("| ") and "---" not in line]
    if len(paper_rows) < 4:
        raise ValueError("Paper provenance table missing")
    date = datetime.fromtimestamp(max(path.stat().st_mtime for path in paths[4:]), SHANGHAI)
    return (comparison, timing, probe, raw, paper_rows, date), None


def render(folder, data):
    comparison, timing, probe, raw, paper_rows, date = data
    cases = comparison["comparisons"]
    fine, coarse, zero = (cases[name] for name in ("direct_h1", "direct_h2", "zero"))
    last = fine["comparisons"][-1]
    target = comparison["first_sideband_predeclared_relative_target"]
    rows, h2_rows, zero_rows = (case["comparisons"][-1]["bottom"] for case in (fine, coarse, zero))
    both_pass = all(harmonic(case["comparisons"][-1][side], m)["complex_vector_relative_difference"] <= target
                    for case in (fine, coarse) for side in ("bottom", "top") for m in (-1, 1))
    opening = ("两档材料更新时间、上下两种入射下，一级复出射均达到预先声明的目标。"
               if both_pass else "一级复出射尚未在两档材料更新时间下全部达到预先声明的目标；保留失败结果，不放宽阈值。")
    identity = fine["identity"]
    lines = ["# 直接时变 Meep 与静态 Green＋Floquet 对照", "",
             f"数据文件最后更新：{date:%Y-%m-%d %H:%M:%S}（Asia/Shanghai；取六份最终运行摘要的文件时间，不充当批次计时）。", "",
             f"**{opening}** 目标为一级四通道复振幅向量相对差 ≤{percent(target)}，分母为 Floquet 参考向量范数。", "",
             f"两条路线保持同一几何、r{identity['resolution']} Yee 网格、原生亚像素材料、Courant={identity['Courant']:g}、PML、参考平面和调制相位。"
             f"载频 fa/c={identity['carrier']:g}，调制频率 Ωa/(2πc)={identity['drive']:g}，介电调制绝对振幅={identity['amplitude']:g}。"
             "没有拟合频率、幅度或相位。偶／奇对称源的实际结果合成为下侧和上侧 x 偏振入射。", "",
             "## 最终边带与控制", "",
             f"主表为下侧单位载频入射、材料更新间隔 h={identity['material_interval']:g} a/c，"
             f"独立记录窗口 [{number(last['window'][0])}, {number(last['window'][1])}] a/c。功率求和包含上下侧 x/y 四个输出通道。", "",
             "| 边带 m | Green＋Floquet 功率 | 直接 Meep 功率 | 复幅向量相对差 | 功率相对差 | 主通道相位差 |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['harmonic']:+d} | {number(row['reference_power'])} | {number(row['observed_power'])} | "
                     f"{percent(row['complex_vector_relative_difference'])} | {percent(row['power_relative_difference'])} | "
                     f"{number(row['dominant_phase_difference_degrees'], 5)}° |")
    lines += ["", "相位列仅为参考中最强通道的相位差；完整复幅判据同时比较所有四个通道。", "",
              "| 检查 | m=−1／+1 的复幅相对变化 | 判断／分母 |", "|---|---:|---|"]
    for name, case in (("h=2", coarse), ("h=1", fine)):
        current = case["comparisons"][-1]["bottom"]
        lines.append(f"| {name} 对 Floquet | {first_pair(current)} | {verdict(current, target)}；Floquet 范数 |")
    lines.append(f"| 材料更新 h=2→1 | {first_pair(comparison['material_update_refinement']['bottom'])} | h=1 范数；Maxwell dt 未改变 |")
    for name, case in (("h=2", coarse), ("h=1", fine)):
        lines.append(f"| {name} 最后两个独立窗口 | {first_pair(case['final_adjacent_window_comparison'])} | 前一窗口范数 |")
    lines += ["", f"上侧入射 h=1 的一级复幅差为 {first_pair(last['top'])}，{verdict(last['top'], target)}。"
              "以上分别报告动态后端一致性、材料保持间隔和有限时间窗敏感性，不能互相替代。", "",
              "| m | 零调制伪侧带功率 | 相对有调制 Meep 功率 |", "|---:|---:|---:|"]
    for m in (-2, -1, 1, 2):
        value = harmonic(zero_rows, m)["observed_power"]
        denominator = harmonic(rows, m)["observed_power"]
        ratio = None if denominator == 0 else value / denominator
        lines.append(f"| {m:+d} | {number(value)} | {percent(ratio)} |")
    auxiliary = comparison.get("zero_background_subtraction_diagnostic")
    if auxiliary is not None:
        if auxiliary.get("primary_raw_results_unchanged") is not True or auxiliary.get("fitted_parameters") != []:
            raise ValueError("Unexpected fitted or primary-changing auxiliary diagnostic")
        aux_fine = auxiliary["comparisons"]["direct_h1"]
        aux_last = aux_fine["windows"][-1]
        if aux_last["window"] != last["window"]:
            raise ValueError("Auxiliary final window differs from primary result")
        aux_rows = aux_last["bottom_sidebands"]
        aux_adjacent = aux_fine["final_adjacent_window_sideband_comparison"]
        def pair_at(values, orders):
            return "／".join(percent(harmonic(values, m)["complex_vector_relative_difference"]) for m in orders)
        lines += ["", "辅助诊断另外比较同源、同窗口的有调制场与零调制场之差：先在各奇偶扇区相减未归一化原始复场，"
                  "再统一除以该有调制运行的载频入射振幅，最后按固定关系合成为物理入射。没有对后端结果拟合。", "",
                  "| h=1 最终下侧入射 | 一级 −1／+1 复幅差 | 二级 −2／+2 复幅差 |", "|---|---:|---:|",
                  f"| 原始主结果对 Floquet | {first_pair(rows)} | {pair_at(rows, (-2, 2))} |",
                  f"| 同源零调制差分对 Floquet | {first_pair(aux_rows)} | {pair_at(aux_rows, (-2, 2))} |",
                  f"| 差分结果末两独立窗口变化 | {first_pair(aux_adjacent)} | {pair_at(aux_adjacent, (-2, 2))} |", "",
                  "原始结果及其判据保持为主结果；差分仅帮助辨认共同开源暂态的影响，仍保留依赖调制的暂态，"
                  "不是额外的稳态仿真或完整收敛证书。m=0 只保存调制引起的有限窗差值，不拼接静态 S0 来构造载频结果。"]
    zero_window = zero["comparisons"][-1]["window"]
    lines += ["", f"零调制采用自身最终窗口 [{number(zero_window[0])}, {number(zero_window[1])}] a/c；"
              "伪侧带在原始主结果中保留，没有从主结果中相减；上述差分仅作单独辅助诊断。其来源可能包含未衰尽静态响应及有限记录泄漏，不能全部解释成数值噪声。", "",
              f"材料接口短测的 K_base／K_plus 与历史材料矩阵差分别为 {number(probe['K_base_relative_difference_from_historical'])}／"
              f"{number(probe['K_plus_relative_difference_from_historical'])}；下一 Maxwell 步 E=K(t)D 的相对残差为 "
              f"{number(probe['next_step_E_equals_updated_KD_relative_residual'])}。这只检查接口与离散本构，不单独证明长期动态正确。", "",
              "## 受迫响应的稳定性范围", ""]
    for label, case in (("h=2", coarse), ("h=1", fine)):
        changes = [item["final_adjacent_cycle_rms_relative_change"] for item in case["probe_stability"]]
        lines.append(f"- {label}：末两调制周期的有限探针 RMS 相对变化，偶／奇扇区为 {percent(changes[0])}／{percent(changes[1])}。")
    lines += ["", "这些数据只检验当前 Γ 点、当前弱调制、载频受迫响应及有限观察时间。"
              "不据此宣布所有未激发模式稳定，也没有测得完整 Floquet 复本征谱、参量阈值或近 BIC 长寿命模式的全局稳定性。"
              "本对照在同一 r32 空间离散下进行，不消除此前更换空间网格时的效率敏感性。", "",
              "## 耗时：首次计算与缓存复用分开", "",
              "| 直接 Meep 实际运行 | 偶＋奇运行墙钟和 | 偶＋奇 CPU 时间和 | 其中材料重建墙钟和 |",
              "|---|---:|---:|---:|"]
    for name, case in (("h=2", coarse), ("h=1", fine), ("零调制控制", zero)):
        clocks = case["timings"]
        lines.append(f"| {name} | {number(clocks['summed_run_wall_seconds'])} s | {number(clocks['summed_run_cpu_seconds'])} s | "
                     f"{number(clocks['summed_material_update_wall_seconds'])} s |")
    stages = {row["stage"]: row["wall_seconds"] for row in timing["timings"]}
    single = sum(stages[name] for name in ("load_portable_first_call", "select_single_bottom_input",
                 "modulation_for_first_call", "solve_single_bottom_carrier"))
    all_inputs = sum(stages[name] for name in ("load_portable_first_call", "modulation_for_first_call", "solve_all_ten_inputs"))
    history = timing["historical_offline_costs"]
    lines += ["", "直接 Meep 的计时从初始化前开始，包含真实时间推进、材料重建、监测与检查点输出；"
              "偶＋奇两次运行同时提供上下两种载频入射。表内为进程时间之和，不是并发任务的日历耗时。"
              "零调制与另一档更新时间是验证成本，未冒充每个生产工况都必须重复。", "",
              "| 使用情形 | 已测时间 | 未包含／解释 |", "|---|---:|---|",
              f"| 单工况从零：直接 Meep h=1 | {number(fine['timings']['summed_run_wall_seconds'])} s（双扇区之和） | 本次实际端口与探针输出；不依赖 Green 库 |",
              f"| 静态 Green 建库 | {number(history['sum_measured_run_elapsed_seconds'])} s（{history['spatial_sources']} 个源进程计时之和） | "
              f"{history['actual_frequency_responses']} 个实际频率响应；全流程／并发日历时间未完整记录 |",
              f"| 已有库：从磁盘加载并完整计算一个下侧载频输入 | {number(single)} s | 包含材料投影、内部场重建和残差／源覆盖检查 |",
              f"| 已有库：从磁盘加载并完整计算 10 个输入 | {number(all_inputs)} s | 20×10 S 与全部内部场／检查 |",
              f"| 数据已加载：10 输入重复计算 | {number(stages['solve_all_ten_inputs_warm_repeat'])} s | 不含建库和本次加载 |",
              f"| 数据已加载：改变全局调制相位后单输入完整重算 | {number(stages['modulation_for_phase_pi_over_3'] + stages['solve_single_bottom_carrier_phase_pi_over_3'])} s | "
              "只复用确实支持的相位变化，不能推到任意幅度或频率 |", "",
              f"单独 85×85 小矩阵求解为 {number(stages['small_algebra_linear_solve_only'])} s，"
              "它不包含材料投影、场重建、建库及检查，不能用它作为完整 Floquet 用时与 Meep 比速度。", "",
              "当前已测记录中，完整 Green 库的静态采样进程时间总和为 "
              f"{number(history['sum_measured_run_elapsed_seconds']/fine['timings']['summed_run_wall_seconds'], 4)} 倍的本次 h=1 双扇区运行时间和。"
              "两者产物不同（全库覆盖五频多源，直接运行覆盖当前载频），历史并发与采样配置也不同，因此这是工作量记录，不是同机同资源加速比。", "",
              ("对当前单一弱转换工况，已有测量支持直接 Meep 避免完整 Green 建库的优势；若仅需少量工况，应优先评估这条简单路线。"
               if history['sum_measured_run_elapsed_seconds'] > fine['timings']['summed_run_wall_seconds']
               else "当前进程计时没有显示直接 Meep 相对所记录静态采样工作量的优势；完整首次成本仍未记录，不能据此作普遍排序。"),
              "在确实复用现有空间、波矢、频率采样和充分源空间时，缓存后的 Floquet 重算较短；"
              "改变几何、k、所需实频点或源空间充分性可能重新建库。不能从现有计时推算任意参数扫描的回本次数。"
              "完整原始离线成本缺项包括 Legume 种子／基底、部分初始化／材料数据、尾部提取验证、失败控制和归档绘图。"
              "本次后端计时还与其他 Meep 运行并发，报告当前实现的实测值，不作峰值性能结论。", "",
              "## 此前论文是否全部直接使用 Meep", "",
              "此前确实有真实时变 Meep，尤其 Nature Photonics 的局部增长／分裂和后续保留极化记忆的 Lorentz 损耗对照；"
              "但没有一篇论文全部目标面板由纯 Meep 完成。LPR／Science 的最新主复现仍是 Legume＋自建 Floquet；"
              "Nature 的后来 d/g 最小奇异值图属于独立 Floquet–Mie/T-matrix，不能归入 Meep。"
              "j 虽有真实 Lorentz–Meep，工作频率与完整放大宽度仍未对应原面板。", "",
              "| 历史基线 | 已完成的实际路线 | 全部目标图是否纯 Meep |", "|---|---|---|",
              "| LPR Fig.3 周期阵列 | 最新 Legume＋自建 Floquet；旧 Meep 仅静态校准 | 否 |",
              "| Science Fig.4 | 最新 Legume＋自建 Floquet；旧 Meep 仅静态与单模控制 | 否 |",
              "| Nature Fig.4 | 静态和部分真实时变 Meep；c 为 Mie，后来 d/g 为 T-matrix；j 为未完整对应论文的 Lorentz–Meep | 否；已有直接动态机制证据 |", "",
              "[逐面板来源、实际脚本与限制](paper_provenance.md)。本报告已读取该来源表，共 "
              f"{len(paper_rows)-1} 个数据行；详细边界以其中的原始报告为准。", "",
              "[完整比较数据](comparison_summary.json) · [计时明细](backend_timing.json) · [材料更新短测](update_probe/summary.json)"]
    if (folder / "direct_comparison.png").is_file():
        lines += ["", "![同网格直接 Meep 与 Floquet 对照](direct_comparison.png)"]
    lines += ["", f"输入 comparison_summary.json SHA256：`{sha(folder/'comparison_summary.json')}`。"
              f"报告脚本 SHA256：`{sha(__file__)}`。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=DEFAULT)
    args = parser.parse_args()
    folder = args.folder.resolve()
    data, pending = verify_inputs(folder)
    if pending is not None:
        print(json.dumps({**pending, "report_written": False}, ensure_ascii=False))
        return
    text = render(folder, data)
    destination = folder / "report.md"
    temporary = folder / "report.md.tmp"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"completed": True, "report_written": True, "report": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
