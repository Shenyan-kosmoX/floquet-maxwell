"""Compare actual COMSOL static PIN states and insert them in the paper array model.

Consumes completed mesh-4/mesh-6 sweeps, never partial solver output. The
assembled array response remains the source's slow-modulation, uncoupled-cell
physical-optics approximation. It is not dynamic full-wave array scattering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from reproduce_zhang2018_original_farfield import coefficients, farfield_at, farfield_grid, digest, ORDERS, PUBLISHED_DB

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/zhang2018_original_alignment_20261019"
SOURCE=ROOT/"literature/zhang2018_original_alignment_20261019/parameters"
CASES=["ON_R7p8","OFF_literal_28pF","OFF_control_0p028pF"]
LABELS={CASES[0]:"ON: R = 7.8 ohm",CASES[1]:"OFF: printed 28 pF",CASES[2]:"OFF: 0.028 pF control"}


def load_complete(path,expected):
    data=json.loads(path.read_text(encoding="utf8"))
    if data.get("completed") is not True:raise RuntimeError(f"Sweep is incomplete: {path}")
    rows=data["rows"]
    if len(rows)!=3*expected:raise RuntimeError(f"Expected {3*expected} complete rows, got {len(rows)}: {path}")
    by={c:sorted([r for r in rows if r["case"]==c],key=lambda r:r["frequency_GHz"])for c in CASES}
    frequencies=np.array([r["frequency_GHz"]for r in by[CASES[0]]])
    for c in CASES:
        if len(by[c])!=expected or not np.array_equal(frequencies,[r["frequency_GHz"]for r in by[c]]):
            raise RuntimeError("Cases must share the complete frequency grid")
    if not np.any(frequencies==10):raise RuntimeError("10 GHz state must be actually solved, not interpolated")
    return {c:{"f":frequencies,"r":np.array([complex(*r["surface_reflection"])for r in by[c]])}for c in CASES}


def compare_static(fine,coarse,out,extra=None):
    source=SOURCE/"Fig6fg_static_reflection.png"
    shutil.copy2(source,out/"paper_Fig6fg_original_crop.png")
    # Remove only the strip of the unrelated photograph above panels f/g.
    # Their panel labels, axes, curves and legends remain intact.
    source_image=plt.imread(source)[32:-20]
    plt.imsave(out/"paper_Fig6fg.png",source_image)
    fig=plt.figure(figsize=(13.6,10.4))
    gs=fig.add_gridspec(2,2,left=.075,right=.97,bottom=.12,top=.9,hspace=.29,wspace=.22)
    original=fig.add_subplot(gs[0,:]);original.imshow(source_image);original.axis("off")
    original.set_title("Published Fig. 6f,g",fontsize=14,pad=8)
    phase=fig.add_subplot(gs[1,0]);amplitude=fig.add_subplot(gs[1,1])
    colors_phase={CASES[0]:"#8b509e",CASES[1]:"#5e6265",CASES[2]:"#078845"}
    colors_amp={CASES[0]:"#164c99",CASES[1]:"#5e6265",CASES[2]:"#d92027"}
    for c in CASES:
        f=fine[c]["f"];r=fine[c]["r"]; fc=coarse[c]["f"];rc=coarse[c]["r"]
        phase.plot(f,np.angle(r,deg=True),color=colors_phase[c],lw=1.8,label=LABELS[c],marker="o",ms=3)
        amplitude.plot(f,abs(r),color=colors_amp[c],lw=1.8,label=LABELS[c],marker="o",ms=3)
        phase.plot(fc,np.angle(rc,deg=True),linestyle="none",marker="x",ms=5,color=colors_phase[c])
        amplitude.plot(fc,abs(rc),linestyle="none",marker="x",ms=5,color=colors_amp[c])
        if extra is not None:
            phase.plot(extra[c]["f"],np.angle(extra[c]["r"],deg=True),linestyle="none",marker="D",ms=5,
                       markerfacecolor="none",markeredgecolor=colors_phase[c])
            amplitude.plot(extra[c]["f"],abs(extra[c]["r"]),linestyle="none",marker="D",ms=5,
                           markerfacecolor="none",markeredgecolor=colors_amp[c])
    difference=np.angle(fine[CASES[0]]["r"],deg=True)-np.angle(fine[CASES[2]]["r"],deg=True)
    phase.plot(fine[CASES[0]]["f"],difference,"--",color="#df8f21",lw=1.4,label="Phase ON - OFF (0.028 pF)")
    for ax in [phase,amplitude]:
        ax.axvspan(9.94,10.06,color=".82",zorder=-1)
        ax.set_xlim(8,11);ax.set_xlabel("Frequency (GHz)");ax.set_xticks(np.arange(8,11.1,.5));ax.grid(alpha=.12)
        ax.legend(fontsize=8.2,frameon=False,loc="lower left")
    phase.set(ylim=(-200,300),ylabel="Surface reflection phase (deg)",title="COMSOL static unit cell: phase")
    phase.set_yticks([-180,-90,0,90,180,270])
    amplitude.set(ylim=(.83,1.025),ylabel="Reflection amplitude",title="COMSOL static unit cell: amplitude")
    amplitude.set_yticks([.85,.9,.95,1.])
    fig.suptitle("Original metal/PIN geometry: actual static full-wave comparison",fontsize=17,y=.98)
    extra_text=" Diamonds: targeted mesh 2."if extra is not None else ""
    fig.text(.5,.052,"Solid circles/lines: mesh 4; crosses: mesh 6."+extra_text+" Reference plane: z = 2 mm.\n"
             "No bias lines; zero-thickness PEC metal and PEC via. 0.028 pF is an explicit input-control hypothesis, not a confirmed correction.",
             ha="center",fontsize=10,linespacing=1.5)
    for ext in ["png","pdf","svg"]:fig.savefig(out/f"paper_vs_static_reflection.{ext}",dpi=220,facecolor="white")
    plt.close(fig)


def compute_scattering(fine,out):
    code_file=ROOT/"literature/general_spatiotemporal_20261019/zhang2018_code.json"
    code=json.loads(code_file.read_text(encoding="utf8"))
    bits=np.array([[[int(s)for s in word]for word in row]for row in code["codes_y_x"]]);ideal=1-2*bits
    a=coefficients(ideal,ORDERS)
    axis=np.linspace(-1,1,401);u,v=np.meshgrid(axis,axis);radius=np.sqrt(u*u+v*v);visible=radius<=1
    scale=np.sin(np.pi*radius/2)/np.maximum(radius,1e-15)
    f_ideal=farfield_grid(a,axis)
    display_ideal=farfield_at(a,v*scale,u*scale)
    uniform=np.ones((1,8,8),complex);uniform_field=farfield_grid(uniform,axis)[0]
    coeffs=[a];fields=[f_ideal];displays=[display_ideal]
    on=fine[CASES[0]]["r"][np.where(fine[CASES[0]]["f"]==10)[0][0]]
    reports=[]
    index0=int(np.where(ORDERS==0)[0][0]);nonzero=ORDERS!=0
    base_peaks=json.loads((BASE/"farfield/metrics.json").read_text(encoding="utf8"))["peak_comparison"]
    for c in CASES[1:]:
        off=fine[c]["r"][np.where(fine[c]["f"]==10)[0][0]]
        gamma=off*(1-bits)+on*bits
        actual_a=coefficients(gamma,ORDERS)
        kappa=(off-on)/2;average=(off+on)/2
        predicted_a=kappa*a.copy();predicted_a[index0]+=average
        actual_f=farfield_grid(actual_a,axis)
        expected_f=kappa*f_ideal.copy();expected_f[index0]+=average*uniform_field
        actual_display=farfield_at(actual_a,v*scale,u*scale)
        peakdb=20*np.log10(abs(actual_f[:,visible]).max(axis=1))
        for i,order in enumerate(ORDERS):
            if order!=0:peakdb[i]=base_peaks[i]["computed_max_dB"]+20*np.log10(abs(kappa))
        reports.append({"case":c,"on_complex":[on.real,on.imag],"off_complex":[off.real,off.imag],
                        "ON_amplitude":float(abs(on)),"OFF_amplitude":float(abs(off)),
                        "ON_phase_deg":float(np.angle(on,deg=True)),"OFF_phase_deg":float(np.angle(off,deg=True)),
                        "unwrapped_ON_minus_OFF_deg":float(np.angle(on,deg=True)-np.angle(off,deg=True)),
                        "contrast_kappa":[kappa.real,kappa.imag],"average":[average.real,average.imag],
                        "all_nonzero_harmonic_amplitude_ratio":float(abs(kappa)),
                        "all_nonzero_harmonic_power_ratio":float(abs(kappa)**2),
                        "all_nonzero_harmonic_offset_dB":float(20*np.log10(abs(kappa))),
                        "direct_code_vs_contrast_coefficient_max_error":float(abs(actual_a-predicted_a).max()),
                        "direct_spatial_sum_vs_contrast_max_error":float(abs(actual_f-expected_f).max()),
                        "orders":ORDERS.tolist(),"peak_dB":peakdb.tolist()})
        coeffs.append(actual_a);fields.append(actual_f);displays.append(actual_display)
    data={"orders":ORDERS,"bits_y_x_slot":bits,"axis":axis,"coefficient":np.array(coeffs),
          "F_physical_uv":np.array(fields),"F_display_hypothesis":np.array(displays),"visible":visible}
    np.savez_compressed(out/"actual_state_array_data.npz",**data)
    return data,reports


def farfield_plot(data,reports,out,static_mesh=4):
    published=plt.imread(BASE/"farfield/paper_SFig3a.png")
    palette=ListedColormap(published[np.rint(np.linspace(635,275,256)).astype(int),1590,:3])
    selected=[0,1,2]
    # Actual printed panel crops, with their own printed peak labels.
    crops={0:published[49:443,1198:1553],1:published[451:846,82:436],2:published[451:846,453:809]}
    fig,axes=plt.subplots(4,3,figsize=(10.6,12.6))
    fig.subplots_adjust(left=.16,right=.91,top=.90,bottom=.105,wspace=.13,hspace=.21)
    rownames=["Published SI Fig. 3a","Ideal reflection +1 / -1","COMSOL states: OFF 28 pF","COMSOL states: OFF 0.028 pF"]
    for col,m in enumerate(selected):
        axes[0,col].imshow(crops[m]);axes[0,col].axis("off")
        axes[0,col].set_title("Carrier"if m==0 else f"Harmonic +{m}",fontsize=13,pad=10)
    for row in range(1,4):
        for col,m in enumerate(selected):
            mi=int(np.where(ORDERS==m)[0][0]);field=data["F_display_hypothesis"][row-1,mi]
            values=20*np.log10(np.maximum(abs(field),1e-15));values[~data["visible"]]=-10
            im=axes[row,col].imshow(values,origin="lower",extent=(-1,1,-1,1),cmap=palette,vmin=-10,vmax=20)
            maximum=(20*np.log10(abs(data["F_physical_uv"][0,mi][data["visible"]]).max())if row==1 else reports[row-2]["peak_dB"][mi])
            axes[row,col].set_title(f"Maximum: {maximum:.3f} dB",fontsize=10,pad=5)
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
    for row,name in enumerate(rownames):
        y=(axes[row,0].get_position().y0+axes[row,0].get_position().y1)/2
        fig.text(.025,y,name,rotation=90,ha="center",va="center",fontsize=11)
    cax=fig.add_axes([.93,.24,.017,.35]);fig.colorbar(im,cax=cax,ticks=np.arange(-10,21,5)).ax.set_title("dB",pad=9)
    fig.suptitle("Actual two-state COMSOL reflection inserted into the original 64 codes",fontsize=15,y=.98)
    fig.text(.5,.949,"Static full-wave unit cell + the paper's slow-modulation, uncoupled-cell array model",ha="center",fontsize=11)
    fig.text(.5,.047,"All panels share the original -10 to 20 dB scale; values above 20 dB saturate, with peaks stated.\n"
             "Display hypothesis: angular radius rho = 2 theta/pi and exchanged display axes; physical u,v fields are saved.\n"
             f"10 GHz, mesh {static_mesh}; no bias lines, PEC metal/via. The 0.028 pF case is a control, not a confirmed source correction.",
             ha="center",fontsize=9.3,linespacing=1.4)
    for ext in ["png","pdf","svg"]:fig.savefig(out/f"paper_ideal_actual_scattering.{ext}",dpi=220,facecolor="white")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.9));x=np.arange(len(ORDERS))
    ideal_peaks=np.array([q["computed_max_dB"]for q in json.loads((BASE/"farfield/metrics.json").read_text())["peak_comparison"]])
    axes[0].plot(ORDERS,PUBLISHED_DB,"ks",label="Published SI3")
    axes[0].plot(ORDERS,ideal_peaks,"k--",label="Ideal Eq7")
    for report,color in zip(reports,["#686868","#dd4b39"]):
        axes[0].plot(ORDERS,report["peak_dB"],"o-",color=color,label=LABELS[report["case"]])
    axes[0].set(xlabel="Harmonic m",ylabel="Maximum 20 log10 |F_m| (dB)",title="Absolute harmonic peak comparison",xticks=ORDERS)
    ratios=[r["all_nonzero_harmonic_power_ratio"]for r in reports]
    axes[1].bar([0,1],ratios,color=["#686868","#dd4b39"],width=.55)
    axes[1].set(yscale="log",ylim=(1e-4,3),ylabel=r"$|\kappa|^2$ = nonzero-harmonic power / ideal",title="Common contrast factor for all m != 0",xticks=[0,1],xticklabels=["OFF 28 pF","OFF 0.028 pF\n(control)"])
    for i,r in enumerate(reports):axes[1].text(i,ratios[i]*1.25,f"{ratios[i]:.5g}\n{r['all_nonzero_harmonic_offset_dB']:+.2f} dB",ha="center",fontsize=10)
    axes[0].legend(fontsize=8,frameon=False);axes[0].grid(alpha=.2);axes[1].grid(axis="y",alpha=.2)
    fig.suptitle("10 GHz: original fixed 64-cell random code, actual static reflection states",fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(out/"harmonic_peak_comparison.png",dpi=220);fig.savefig(out/"harmonic_peak_comparison.pdf");plt.close(fig)


def sensitivity(fine,coarse):
    result={}
    for c in CASES:
        pairs=[(fine[c]["r"][np.where(fine[c]["f"]==f)[0][0]],r)for f,r in zip(coarse[c]["f"],coarse[c]["r"])if np.any(fine[c]["f"]==f)]
        result[c]={"common_frequencies":int(len(pairs)),"max_complex_reflection_change":float(max(abs(x-y)for x,y in pairs)),
                   "max_amplitude_change":float(max(abs(abs(x)-abs(y))for x,y in pairs)),
                   "max_phase_change_deg":float(max(abs(np.angle(x/y,deg=True))for x,y in pairs))}
    return result


def states_at10(data):
    value={c:data[c]["r"][np.where(data[c]["f"]==10)[0][0]]for c in CASES}
    result={c:{"reflection":[float(r.real),float(r.imag)],"amplitude":float(abs(r)),"phase_deg":float(np.angle(r,deg=True))}for c,r in value.items()}
    for c in CASES[1:]:
        k=(value[c]-value[CASES[0]])/2
        result[c]["phase_ON_minus_OFF_deg"]=float(np.angle(value[CASES[0]],deg=True)-np.angle(value[c],deg=True))
        result[c]["principal_phase_separation_deg"]=float(abs(np.angle(value[CASES[0]]/value[c],deg=True)))
        result[c]["contrast_power_ratio"]=float(abs(k)**2)
        result[c]["contrast_offset_dB"]=float(20*np.log10(abs(k)))
    return result


def relative_phase_sensitivity(fine,coarse):
    result={}
    for c in CASES[1:]:
        rows=[]
        for j,f in enumerate(coarse[c]["f"]):
            selected=np.where(fine[c]["f"]==f)[0]
            if not len(selected):continue
            k=int(selected[0]);rf=fine[CASES[0]]["r"][k]/fine[c]["r"][k]
            rc=coarse[CASES[0]]["r"][j]/coarse[c]["r"][j]
            kf=(fine[c]["r"][k]-fine[CASES[0]]["r"][k])/2
            kc=(coarse[c]["r"][j]-coarse[CASES[0]]["r"][j])/2
            rows.append({"frequency_GHz":float(f),"phase_gap_change_deg":float(np.angle(rf/rc,deg=True)),
                         "nonzero_harmonic_contrast_change_dB":float(20*np.log10(abs(kf/kc)))})
        result[c]={"rows":rows,"max_abs_phase_gap_change_deg":max(abs(r["phase_gap_change_deg"])for r in rows),
                   "max_abs_contrast_change_dB":max(abs(r["nonzero_harmonic_contrast_change_dB"])for r in rows)}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=BASE/"unitcell_plots")
    parser.add_argument("--include-additional",action="store_true",help="Require completed targeted mesh2 and continuous-bias-line controls")
    args=parser.parse_args();out=args.output
    finepath=BASE/"unitcell/sweep_mesh4.json";coarsepath=BASE/"unitcell/sweep_mesh6.json"
    fine=load_complete(finepath,12);coarse=load_complete(coarsepath,6)
    extra=None;additional={}
    if args.include_additional:
        extra=load_complete(BASE/"unitcell/sweep_mesh2.json",3)
        bias=load_complete(BASE/"unitcell/bias_line/sweep_mesh4.json",1)
        additional={"mesh2_vs_mesh4":sensitivity(extra,fine),"mesh2_vs_mesh4_relative_phase":relative_phase_sensitivity(extra,fine),"mesh2_at_10GHz":states_at10(extra),
                    "bias_line_vs_biasfree_mesh4":sensitivity(bias,fine),"bias_line_at_10GHz":states_at10(bias),
                    "bias_scope":"Continuous 0.2mm bias-line geometry control; no unspecified-position RF choke. Not full bias-network reproduction.",
                    "mesh_scope":"Targeted 9.8/10/10.2GHz refinement; this does not establish full 8-11GHz convergence."}
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"svg.fonttype":"none","font.size":10})
    compare_static(fine,coarse,out,extra)
    scattering_input=extra if extra is not None else fine
    static_mesh=2 if extra is not None else 4
    data,results=compute_scattering(scattering_input,out);farfield_plot(data,results,out,static_mesh)
    mesh=sensitivity(fine,coarse)
    nulls={str(k):json.loads((BASE/f"unitcell/grounded_slab_null_mesh{k}.json").read_text())for k in [6,4,2]if (BASE/f"unitcell/grounded_slab_null_mesh{k}.json").exists()}
    if args.include_additional:
        nulls["bias_line_mesh4"]=json.loads((BASE/"unitcell/bias_line/grounded_slab_null_mesh4.json").read_text())
    metadata={"doi":"10.1038/s41467-018-06802-0","source_panels":["Main Fig6f,g","SI Fig3a"],
              "static_scope":"COMSOL frequency-domain original-dimension unit cell, normal incidence E_x, periodic boundaries; bias-free control; PEC zero-thickness metal and PEC via, source lossy F4B.",
              "dynamic_scope":"Actual two complex static reflection states inserted in paper Eq6+7; adiabatic switching, isotropic Epq=1, no unequal-state mutual coupling, switching transient, bias network or dynamic full-wave array.",
              "capacitance":"Printed article 28pF preserved. 0.028pF is a source-motivated control only, not confirmed erratum or fitted value.",
              "reference_plane":"Surface z=2mm, propagation sign checked independently by grounded dielectric analytic response", "mesh_sensitivity":mesh,"mesh4_vs_mesh6_relative_phase":relative_phase_sensitivity(fine,coarse),"grounded_slab_checks":nulls,
              "at_10GHz":results,"farfield_static_input_mesh":static_mesh,"sweep_frequencies_GHz":fine[CASES[0]]["f"].tolist(),"additional_controls":additional,
              "input_hashes":{str(p):digest(p)for p in [finepath,coarsepath,ROOT/"scripts/comsol_zhang2018_unitcell.py",SOURCE/"source_contract.json",Path(__file__)]}}
    (out/"metrics.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    if args.include_additional:
        for p in [BASE/"unitcell/sweep_mesh2.json",BASE/"unitcell/bias_line/sweep_mesh4.json"]:metadata["input_hashes"][str(p)]=digest(p)
        (out/"metrics.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    table="\n".join(f"| {LABELS[r['case']]} | {r['OFF_amplitude']:.6f} | {r['OFF_phase_deg']:.3f} | {r['all_nonzero_harmonic_power_ratio']:.6g} | {r['all_nonzero_harmonic_offset_dB']:+.4f} |"for r in results)
    rows=[]
    for i,m in enumerate(ORDERS):rows.append(f"| {m:+d} | {PUBLISHED_DB[i]:.4f} | {results[0]['peak_dB'][i]:.4f} | {results[1]['peak_dB'][i]:.4f} |")
    extra_report=""
    if args.include_additional:
        extra_rows=[]
        for model,label in [(states_at10(fine),"无bias mesh4"),(additional["mesh2_at_10GHz"],"无bias mesh2"),(additional["bias_line_at_10GHz"],"连续bias线 mesh4")]:
            for c in CASES[1:]:
                p=model[c];extra_rows.append(f"| {label} | {LABELS[c]} | {model[CASES[0]]['amplitude']:.6f} | {p['amplitude']:.6f} | {p['phase_ON_minus_OFF_deg']:.3f} | {p['contrast_offset_dB']:+.4f} |")
        gap_rows=[]
        for c in CASES[1:]:
            for r in additional["mesh2_vs_mesh4_relative_phase"][c]["rows"]:
                gap_rows.append(f"| {LABELS[c]} | {r['frequency_GHz']:.1f} | {r['phase_gap_change_deg']:+.4f} | {r['nonzero_harmonic_contrast_change_dB']:+.5f} |")
        extra_report="\n## 新增针对性网格与偏置线控制\n\n| 控制 | OFF输入 | ON幅度 | OFF幅度 | ON−OFF相位差/° | 非零谐波变化/dB |\n|---|---|---:|---:|---:|---:|\n"+"\n".join(extra_rows)+"\n\nmesh2只在9.8、10、10.2GHz计算，图中以菱形叠加；宽频主曲线仍来自mesh4，10GHz下游远场使用更细mesh2。未据此宣称完整8–11GHz收敛。连续0.2mm偏置线控制只在10GHz检查几何敏感性，没有插入原文位置未确定的2nH choke，不能替代完整偏置网络。两态相位差消去共同参考面相位，因此比单个绝对相位更适合本机制判断。\n\n| OFF输入 | 频率/GHz | mesh4→2的相位差变化/° | 非零谐波对比度变化/dB |\n|---|---:|---:|---:|\n"+"\n".join(gap_rows)+"\n"
    report=f"""# 原金属/PIN单元的真实静态全波与随机码远场

论文：[Zhang et al., Space-time-coding digital metasurfaces](https://doi.org/10.1038/s41467-018-06802-0)。比较正文 Fig.6f,g 与 SI Fig.3。两部分各自的物理模型边界明确保留。

![论文与COMSOL静态反射](paper_vs_static_reflection.png)

## 实际计算与输入

读取已完整结束的 COMSOL mesh4（每态12频点）和 mesh6（每态6频点），全部三态使用同一几何：15mm周期，7.6×6mm贴片，2mm厚F4B（ε=2.65，tanδ=0.001），原尺寸stub、0.3mm PIN间隙、via焊盘。ON为7.8Ω串联30pH；OFF为28pF串联30pH。另独立运行OFF=0.028pF控制。法向x偏振；x/y周期边界。当前为SI9支持比较的无偏置线控制，金属和via用PEC，金属零厚度；未计算完整偏置网络、实金属损耗及焊料。

论文正文确实印28pF，厂商手册的典型电容量级约0.025pF。这里没有宣布文章存在已确认的单位错误：28pF保留为字面输入；0.028pF单列为解释控制，没有扫描电容拟合曲线。

所有相位参考同一z=2mm贴片平面。端口传播因子的符号已通过相同端口与网格的解析PEC接地介质薄膜检查。原图的精确端口参考面未公开，相位未为贴图平移。

## 10GHz两态对比

以下真实状态与下游远场使用 mesh{static_mesh} 的10GHz解。ON反射幅度 {results[0]['ON_amplitude']:.6f}、相位 {results[0]['ON_phase_deg']:.3f}°。

| OFF输入 | OFF幅度 | OFF相位/° | 非零谐波功率/理想 | 非零谐波变化/dB |
|---|---:|---:|---:|---:|
{table}

28pF字面输入下，ON/OFF非常接近，调制对比度明显弱。0.028pF控制得到相近于原文的两态相位分离机制，频谱位置与幅度仍有差异，未补偿或拟合。原文声称10GHz附近两态幅度大于0.95，当前ON约{results[0]['ON_amplitude']:.3f}，该差别保留。

## 把真实反射代入原64时序

Γ(t)=OFF·(1-bit)+ON·bit。记理想码s=1−2bit，κ=(OFF−ON)/2，Γ平均=(OFF+ON)/2，则Γ(t)=Γ平均+κs(t)。因此对所有m≠0，原Eq7远场严格满足 F_actual,m=κF_ideal,m；方向图形状不变而强度统一缩放。m=0另有均匀阵列项，不能只乘κ。直接从全部时槽重新计算Fourier和空间叠加，已独立核对这个恒等式，最大复场误差{max(r['direct_spatial_sum_vs_contrast_max_error']for r in results):.6g}。

![原图、理想码、真实两态远场](paper_ideal_actual_scattering.png)

以上沿用原图公共−10到20dB色条，超过20dB的载频峰值会饱和显示，真实峰值写在标题。显示视图显式采用之前诊断的角半径+轴交换假设，原作者绘图代码尚未确认；完整物理uv复场保存在数据文件中。没有逐图归一化。

| m | 原SI3峰值/dB | COMSOL字面28pF/dB | COMSOL0.028pF控制/dB |
|---|---:|---:|---:|
{chr(10).join(rows)}

![各谐波峰值与对比度](harmonic_peak_comparison.png)

这里完成的是实际静态单元全波→慢调制阵列模型的连接。原论文本来就忽略单元互耦并使用Epq=1；本轮没有验证相邻单元异态时的互耦、开关瞬态或完整8×8动态全波。也不把原金属/PIN结果称为Legume介质薄膜复现。

## 网格敏感性和复算

mesh6→mesh4的共同频点最大复反射变化：{json.dumps({k:v['max_complex_reflection_change']for k,v in mesh.items()},ensure_ascii=False)}。这是当前两档网格敏感性，不能代替所有几何/电路近似的检查。

{extra_report}

复算后处理：`.venv/Scripts/python.exe sptc_toolbox/scripts/plot_zhang2018_unitcell_scattering.py{' --include-additional' if args.include_additional else ''}`。脚本拒绝completed=false或缺少频点的输入。模型脚本为`scripts/comsol_zhang2018_unitcell.py`。

数据：`actual_state_array_data.npz`；指标、完整模型范围和输入SHA256：`metrics.json`。
"""
    (out/"report.md").write_text(report,encoding="utf8")
    print(json.dumps({"output":str(out),"at_10GHz":results,"mesh_sensitivity":mesh},ensure_ascii=True))


if __name__=="__main__":main()
