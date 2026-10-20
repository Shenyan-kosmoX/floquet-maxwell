"""Record the reviewed r48 weak-forced response choice and its limitations."""
import json
import hashlib
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
RECIPROCITY=BASE.parent/'reciprocity'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(delta,ref):
    return float(np.linalg.norm(delta)/np.linalg.norm(ref))


diagnostics=read(BASE/'tail_diagnostics_t2000/summary.json')
even=next(r for r in diagnostics['source_rows'] if r['source']=='incident_even')
odd=next(r for r in diagnostics['source_rows'] if r['source']=='incident_odd')
fit=even['fits']['2']
candidate_path=BASE/'tail_diagnostics_t2000/rank2/incident_even/response_t2000.npz'
candidate_meta=read(candidate_path.parent/'summary.json')
maxwell=read(BASE/'maxwell_t2000.json')
raw=np.load(RECIPROCITY/'r48_raw/reciprocal_born.npz')['output_x']
selected=np.load(RECIPROCITY/'r48_selected/reciprocal_born.npz')['output_x']
previous=np.load(RECIPROCITY/'r48_t1500_candidate/reciprocal_born.npz')['output_x']
observable=[]
for hi,harmonic in enumerate((-1,1)):
    y=selected[hi,:,0];r=raw[hi,:,0];p=previous[hi,:,0]
    observable.append(dict(harmonic=harmonic,raw_power_x=float(np.vdot(r,r).real),
        selected_power_x=float(np.vdot(y,y).real),
        raw_to_selected_complex_relative_difference=rel(r-y,y),
        raw_to_selected_power_relative_difference=float(np.vdot(r,r).real/np.vdot(y,y).real-1),
        corrected_t1500_to_t2000_complex_relative_difference=rel(p-y,y),
        corrected_t1500_to_t2000_power_relative_difference=float(np.vdot(p,p).real/np.vdot(y,y).real-1)))
choices={
 'incident_even':dict(path=str(candidate_path.relative_to(BASE)),mode='same_grid_two_pole_time_completion',
     raw_checkpoint=2000,window_validated=bool(fit['window_validated']),heldout_time_validated=True,
     maxwell_equation_validated=True,independent_validated=False,continuous_wave_solver_validated=False,
     selected_for_scattering=True,accepted_for_finite_forced_response=True,
     validation_file='tail_validation.json',
     acceptance_scope='Only this fixed-geometry, five-frequency, weak first-order x-polarized forced response; '
        'same-grid measured poles, coherent E/D/port time completion, actual fourth-window and Maxwell checks.'),
 'incident_odd':dict(path='sampling/incident_odd/response_t2000.npz',mode='raw_final_window',
     raw_checkpoint=2000,window_validated=False,heldout_time_validated=False,
     maxwell_equation_validated=True,independent_validated=False,continuous_wave_solver_validated=False,
     selected_for_scattering=True,accepted_for_finite_forced_response=True,
     validation_file='tail_validation.json',
     acceptance_scope='Keep actual raw response: neither tested one-pole nor two-pole tail model predicts later increments. '
         'Small raw window sensitivity, native residual and unit-input port imbalance recorded separately.')}
sources=[BASE/'tail_diagnostics_t2000/summary.json',candidate_path,candidate_path.parent/'summary.json',
    BASE/'maxwell_t2000.json',RECIPROCITY/'r48_raw/summary.json',RECIPROCITY/'r48_selected/summary.json',
    RECIPROCITY/'r48_t1500_candidate/summary.json']
report=dict(status='REVIEWED_R48_WEAK_FIRST_ORDER_RESPONSE_SELECTION',completed=True,
    resolution=48,cosine_amplitude=.01,carrier_frequency=.398,modulation_frequency=.01,
    selections=choices,
    even_poles_cycles=fit['increment_fits'][0]['pole_cycles'],
    even_heldout_frequency_increment_relative_residuals=fit['later_increment_prediction_relative_residuals'],
    even_shared_pole_time_predictions=candidate_meta.get('shared_pole_heldout_increment_predictions'),
    even_corrected_field_shifted_window_relative_difference=fit['differences']['electric']['shifted_time_window_candidate'],
    odd_rejected_tail_model_checks={rank:row['checks'] for rank,row in odd['fits'].items()},
    odd_raw_window_changes=odd['raw_window_changes'],
    native_maxwell= maxwell['aggregate'],plane_wave_controls=maxwell['plane_wave_controls'],
    observable_sensitivity=observable,
    independent_validated=False,continuous_wave_solver_validated=False,
    validation_methods=['Actual same-grid frequency-domain pole law across five DFT frequencies',
        'Later physical-time increments and shifted 500/1000/1500 versus 1000/1500/2000 windows',
        'Native discrete curlcurl residual with independently checked plane-wave symbol',
        'E=K D from independently sampled fields and native material map',
        'Static unit-input outgoing power and measured incoming normalization',
        'Unconjugated Lorentz receiver identity already compared to actual full-current references at r32/r40'],
    limits=['No r48 continuous-wave or additional polarization-source solve was run.',
        'Only x-to-x first-order scattering is obtained; y receiving fields are absent and outputs are not assumed zero.',
        'No r48 source basis or finite M2 system exists; no mode package was fabricated.',
        'Small temporal-model sensitivity does not establish spatial convergence.',
        'Fixed Courant refinement also changes dt, without independent timestep convergence.',
        'The small incoming-tail increment has relative prediction residual slightly above 1e-3 but absolute '
        'late-response residual 2.66e-9; retained as an observed sensitivity, not reclassified as a strict check pass.'],
    provenance={str(path):sha(path) for path in sources},script_sha256=sha(__file__))
for name,value in [('tail_validation.json',report),('sampling_selection.json',dict(
        status='REVIEWED_R48_WEAK_FORCED_SELECTION',selections=choices,
        method='Two-pole completion for even only; odd retained raw. Candidate files remain historically unselected; '
               'this explicit manifest records the subsequent narrow acceptance.',validation_file='tail_validation.json'))]:
    path=BASE/name
    if path.exists():raise ValueError(f'Preserve prior review: {path}')
    path.write_text(json.dumps(value,indent=2)+'\n')
grid_rows=[]
for name in ('r32_selected','r40_selected','r48_selected'):
    summary=read(RECIPROCITY/name/'summary.json')
    grid_rows.append([summary['resolution'],*[r['power_x'] for r in summary['sidebands']]])
lines=['# r48 一级互易对照：时间尾部审查', '',
    '真实偶/奇入射均完成 t=500、1000、1500、2000 的五频采样。选择偶入射的本网格双极点时间补全与奇入射的原始 t2000。没有额外 CW 或极化源传播，没有构造虚假的 r48 模态基底。','',
    '偶入射两极点为 0.3893921568−i4.48060×10⁻⁶ 与 0.3977340372−i7.95179×10⁻⁴。早期频率拟合对后续增量的相对残差分别为 2.41×10⁻⁵、1.985×10⁻⁴；修正电场移窗差为 6.08×10⁻⁷。全部 E/D/输出/输入/原生端口使用同一线性时间算子，先撤销入射归一化再重新归一化。','',
    '偶场原始 t2000 的最大 Maxwell 残差/算子项为 7.55×10⁻⁵，补全后为 6.49×10⁻⁷；最大单位输入功率缺口从 2.61×10⁻⁴ 降至 1.13×10⁻⁹。奇入射模型不通过后窗预测，保留 raw，其相应残差为 1.87×10⁻⁶，功率缺口为 2.07×10⁻⁷。','',
    '| 指标 | m=−1 | m=+1 |','|---|---:|---:|',
    f'| 选定一级 x 功率 | {observable[0]["selected_power_x"]:.10g} | {observable[1]["selected_power_x"]:.10g} |',
    f'| raw→selected 复幅相对差 | {observable[0]["raw_to_selected_complex_relative_difference"]:.5g} | {observable[1]["raw_to_selected_complex_relative_difference"]:.5g} |',
    f'| t1500→t2000 补全结果复幅相对差 | {observable[0]["corrected_t1500_to_t2000_complex_relative_difference"]:.5g} | {observable[1]["corrected_t1500_to_t2000_complex_relative_difference"]:.5g} |','',
    '三个网格的相同一级 x→x 定义结果（各自明示的 selected 时间响应）：','',
    '| 分辨率 | m=−1 功率 | m=+1 功率 |','|---|---:|---:|',
    *[f'| {r} | {a:.10g} | {b:.10g} |' for r,a,b in grid_rows],'',
    '变化不单调，不能据此声称绝对效率已空间收敛，也不进行 Richardson 外推。极小的时间补全敏感性并不消除网格差。该表是完整空间的一阶导数/Born 对照，不是 r48 M2；缺少 y 接收激励，因此没有声明全部四输出通道。','',
    '小 incoming 尾增量的相对预测残差约 0.00106，略超此前 0.001 模型数值阈值；其相对实际晚时响应绝对残差仅 2.66×10⁻⁹。保留原值并基于当前弱受迫可观测量接受该选择，不把这个严格检查追溯改成通过。', '',
    '数据与证据：`tail_validation.json`、`sampling_selection.json`、`maxwell_t2000.json`、`tail_diagnostics_t2000/`，以及 `../reciprocity/r48_raw/`、`../reciprocity/r48_selected/`。']
(BASE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(observable_sensitivity=observable,grid_rows=grid_rows),indent=2))
