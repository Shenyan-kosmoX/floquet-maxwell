# Extended native Lorentz static q survey

q=.1,.2,.3,.4,.5,.6,.65,.7,1.0, r16/Courant.25, gamma=0, unchanged physical Lorentz parameters, source center .26516/width.12 and two150-unit windows. All raw poles and three-probe field traces are preserved.

| q | pole counts by window/probe | unique selected f | candidate frequencies |
|---:|---|---:|---|
| 0.1 | [[1, 1, 1], [1, 1, 1]] | 0.2858133958774289 | [0.2858133958774289] |
| 0.2 | [[1, 1, 1], [1, 1, 1]] | 0.28679740535823284 | [0.28679740535823284] |
| 0.3 | [[2, 2, 2], [2, 2, 2]] | unselected | [0.20773482496809995, 0.2077671833513503, 0.28944669661737554, 0.2894809980784997] |
| 0.4 | [[4, 4, 4], [2, 2, 3]] | unselected | [0.24362452060583775, 0.254815541123742, 0.25483920015419464, 0.2549863770436078, 0.2781737916984671, 0.29800378231378133] |
| 0.5 | [[2, 2, 1], [1, 1, 1]] | unselected | [0.25938909848860797, 0.3162678347027259, 0.3168886264230484] |
| 0.6 | [[1, 1, 1], [1, 1, 1]] | 0.2634722083324177 | [0.2634722083324177] |
| 0.65 | [[1, 1, 1], [1, 1, 1]] | 0.2642879826747937 | [0.2642879826747937] |
| 0.7 | [[1, 1, 1], [1, 1, 1]] | 0.26473173803326894 | [0.26473173803326894] |
| 1.0 | [[1, 1, 1], [1, 1, 1]] | 0.26519160247606094 | [0.26519160247606094] |

{
  "lower": {
    "target_frequency": 0.2645714835348636,
    "sign_change_brackets": [
      {
        "q_bracket": [
          0.65,
          0.7
        ],
        "frequency_bracket": [
          0.2642879826747937,
          0.26473173803326894
        ],
        "identity_status": "Source-visible single-pole candidates; no full spatial field identity across this interval yet"
      }
    ],
    "rejected_unconnected_candidate_sign_changes": [
      {
        "q_bracket": [
          0.2,
          0.6
        ],
        "frequency_bracket": [
          0.28679740535823284,
          0.2634722083324177
        ],
        "identity_status": "Source-visible single-pole candidates; no full spatial field identity across this interval yet",
        "rejection": "Do not bridge ambiguous intermediate q samples or assume a single branch",
        "ambiguous_q": [
          0.3,
          0.4,
          0.5
        ]
      }
    ],
    "status": "SOURCE_VISIBLE_SIGN_CHANGE_FOUND"
  },
  "upper": {
    "target_frequency": 0.26574826788359995,
    "sign_change_brackets": [],
    "rejected_unconnected_candidate_sign_changes": [
      {
        "q_bracket": [
          0.2,
          0.6
        ],
        "frequency_bracket": [
          0.28679740535823284,
          0.2634722083324177
        ],
        "identity_status": "Source-visible single-pole candidates; no full spatial field identity across this interval yet",
        "rejection": "Do not bridge ambiguous intermediate q samples or assume a single branch",
        "ambiguous_q": [
          0.3,
          0.4,
          0.5
        ]
      }
    ],
    "status": "NO_SIGN_CHANGE_IN_UNAMBIGUOUS_SAMPLED_POINTS"
  }
}

{
  "frequency": 0.26519160247606094,
  "inside_predicted_weak_tongue": true,
  "interpretation": "The unique source-visible M-point frequency lies between the first-order static targets; this predicts the weak tongue reaches the Gamma-M segment endpoint if it is the same physical branch. No upper exit within q<=1 is thereby demonstrated."
}

- q<=1 is the Gamma-M segment; a missing upper crossing must not be replaced by an extrapolated band edge.
- Source-visible one-pole consistency is not a complete eigenvalue or spatial-field branch certificate.
- 600 THz is the explicitly cyclic material interpretation; these frequencies are not the paper196THz half-pump.
- No fitted decay is assigned a physical radiation Q.
- Some broad-source static candidates have positive fitted decay despite the passive gamma=0 model. This is unresolved numerical/extraction evidence, not physical gain; real-frequency probe consistency does not validate these imaginary parts.
- Static target frequencies use central first-order sigma derivative, not directly solved dynamic gap edges.
