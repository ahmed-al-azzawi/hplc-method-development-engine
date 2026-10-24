import streamlit as st

def generate_mermaid_flowchart() -> str:
    """Reads session state run history arrays and generates a syntax-safe Mermaid flowchart string."""
    peak_counts = st.session_state.get('all_peak_counts', [])
    resolutions = st.session_state.get('all_resolutions_final', [])
    runtimes = st.session_state.get('all_run_times', [])
    flow_rates = st.session_state.get('all_flow_rates', [])
    lengths = st.session_state.get('all_column_lengths', [])
    particle_sizes = st.session_state.get('all_particle_sizes', [])
    pct_bs = st.session_state.get('all_pct_bs', [])

    peakspec = st.session_state.get('peakspec', 0)
    res_spec = st.session_state.get('res_spec', 1.4)
    runspec = st.session_state.get('runspec', 15.0)

    total_runs = len(peak_counts)
    if total_runs == 0:
        mermaid_code = "graph TD\n    Start[Start Optimization] --> Wait[No Run Data Available]"
        st.session_state['current_mermaid'] = mermaid_code
        return mermaid_code

    lines = [
        "graph TD",
        "    classDef runNode fill:#2563eb,color:#fff,stroke:#1e40af,stroke-width:2px;",
        "    classDef decisionNode fill:#f59e0b,color:#000,stroke:#d97706,stroke-width:1px;",
        "    classDef outcomeNode fill:#10b981,color:#fff,stroke:#047857,stroke-width:2px;",
        "    classDef escalateNode fill:#ef4444,color:#fff,stroke:#b91c1c,stroke-width:2px;"
    ]

    prev_node_id = "Start[Start Optimization]"

    for i in range(total_runs):
        run_id = f"R{i+1}"
        dec_id = f"D{i+1}"
        out_id = f"O{i+1}"

        peaks = peak_counts[i] if i < len(peak_counts) else "N/A"
        rs = resolutions[i] if i < len(resolutions) else 0.0
        tR = runtimes[i] if i < len(runtimes) else 0.0
        F = flow_rates[i] if i < len(flow_rates) else "N/A"
        L = lengths[i] if i < len(lengths) else "N/A"
        dp = particle_sizes[i] if i < len(particle_sizes) else "N/A"
        pct_b = pct_bs[i] if i < len(pct_bs) else "N/A"

        # Construct clean node definitions without embedded quotes or unescaped style separators
        run_label = f"Run {i+1} Conditions<br/>F: {F} mL/min | L: {L} mm<br/>dp: {dp} um | %B: {pct_b}%"
        metric_label = f"Run {i+1} Metrics<br/>Peaks: {peaks}/{peakspec}<br/>Rs: {rs:.2f} (Spec >= {res_spec})<br/>tR: {tR:.2f} min (Spec <= {runspec})"

        lines.append(f'    {run_id}["{run_label}"]')
        lines.append(f'    class {run_id} runNode;')
        lines.append(f'    {dec_id}{{"{metric_label}"}}')
        lines.append(f'    class {dec_id} decisionNode;')

        lines.append(f'    {prev_node_id} --> {run_id}')
        lines.append(f'    {run_id} --> {dec_id}')

        # Evaluation Matrix Decision Nodes
        if isinstance(peaks, (int, float)) and peaks == peakspec:
            pass_res = rs >= res_spec
            tR_far_below = tR < (0.5 * runspec)
            tR_below = (tR < runspec) and not tR_far_below
            tR_above = tR > runspec
            rs_below_floor = rs < 0.5

            if pass_res:
                if tR_far_below or tR_below:
                    lines.append(f'    {dec_id} -->|Peaks == Spec, Rs >= Spec, tR <= Spec| {out_id}["Method MET!"]')
                    lines.append(f'    class {out_id} outcomeNode;')
                else:
                    lines.append(f'    {dec_id} -->|Peaks == Spec, Rs >= Spec, tR > Spec| {out_id}["Efficiency (N): Increase F"]')
                    lines.append(f'    class {out_id} decisionNode;')
            else:
                if rs_below_floor:
                    lines.append(f'    {dec_id} -->|Rs Below Floor (< 0.5)| {out_id}["Escalate: Selectivity (alpha)"]')
                    lines.append(f'    class {out_id} escalateNode;')
                elif tR_far_below:
                    lines.append(f'    {dec_id} -->|tR << Spec, Floor < Rs < Spec| {out_id}["Efficiency (N): Increase L or Decrease F/dp/SPP"]')
                    lines.append(f'    class {out_id} decisionNode;')
                elif tR_below:
                    lines.append(f'    {dec_id} -->|tR < Spec, Floor < Rs < Spec| {out_id}["Efficiency (N): Decrease F/dp or switch to SPP"]')
                    lines.append(f'    class {out_id} decisionNode;')
                else:
                    lines.append(f'    {dec_id} -->|tR > Spec, Floor < Rs < Spec| {out_id}["Efficiency (N): Decrease dp, adjust F"]')
                    lines.append(f'    class {out_id} decisionNode;')
        elif isinstance(peaks, (int, float)) and peaks < peakspec:
            lines.append(f'    {dec_id} -->|Peaks < Spec| {out_id}["Escalate: Retention/Selectivity Cycle"]')
            lines.append(f'    class {out_id} escalateNode;')
        else:
            lines.append(f'    {dec_id} -->|Evaluated| {out_id}["Next Strategy Pending"]')
            lines.append(f'    class {out_id} decisionNode;')

        prev_node_id = out_id

    mermaid_code = "\n".join(lines)
    st.session_state['current_mermaid'] = mermaid_code
    return mermaid_code

st.title("🗺️ Method Optimization Decision Flowchart")

# Generate or update graph string
mermaid_str = generate_mermaid_flowchart()

# Render natively in Streamlit
st.mermaid_chart(mermaid_str)
