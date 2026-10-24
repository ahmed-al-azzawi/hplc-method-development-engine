import streamlit as st


import streamlit as st


def get_default_column_inventory() -> list:
    """Generates an extensive default inventory reflecting Restek's commercial product catalog.

    Includes Raptor (SPP), Force (FPP), Ultra, and Pinnacle DB configurations
    across standard lengths, IDs, particle sizes, and USLC chemistries.
    """
    inventory = []

    # Restek Core Chemistries (USLC & Traditional)
    ligands = [
        "C18",
        "C8",
        "Biphenyl",
        "FluoroPhenyl",
        "PFP",
        "ARC-18",
        "AQ-C18",
        "Cyano",
        "Phenyl-Hexyl",
        "Silica",
    ]

    # Standard Internal Diameters (mm)
    ids = [2.1, 3.0, 4.6]

    # --- 1. RAPTOR CORE-SHELL (SPP) COLUMNS ---
    # Common Particle Sizes: 1.8 µm, 2.7 µm, 5.0 µm
    spp_configs = [
        # UHPLC / High-Speed (1.8 µm)
        (30.0, 1.8, [2.1, 3.0]),
        (50.0, 1.8, [2.1, 3.0]),
        (100.0, 1.8, [2.1, 3.0]),
        # Rapid Resolution & Standard (2.7 µm)
        (50.0, 2.7, [2.1, 3.0, 4.6]),
        (100.0, 2.7, [2.1, 3.0, 4.6]),
        (150.0, 2.7, [2.1, 3.0, 4.6]),
        # High-Efficiency HPLC (5.0 µm)
        (50.0, 5.0, [2.1, 3.0, 4.6]),
        (100.0, 5.0, [2.1, 3.0, 4.6]),
        (150.0, 5.0, [3.0, 4.6]),
        (250.0, 5.0, [4.6]),
    ]

    for length, dp, id_list in spp_configs:
        for inner_id in id_list:
            for ligand in ["C18", "Biphenyl", "FluoroPhenyl", "ARC-18", "C8"]:
                inventory.append({
                    "length": float(length),
                    "dp": float(dp),
                    "is_spp": True,
                    "ligand": ligand,
                    "id": float(inner_id),
                })

    # --- 2. FORCE & ULTRA FULLY POROUS (FPP) COLUMNS ---
    # Common Particle Sizes: 1.7 µm, 1.8 µm, 1.9 µm, 3.0 µm, 5.0 µm
    fpp_configs = [
        # UHPLC Fully Porous (1.7 µm / 1.8 µm / 1.9 µm)
        (30.0, 1.8, [2.1]),
        (50.0, 1.8, [2.1, 3.0]),
        (100.0, 1.8, [2.1, 3.0]),
        (150.0, 1.8, [2.1]),
        # Intermediate / Scalable HPLC (3.0 µm)
        (50.0, 3.0, [2.1, 3.0, 4.6]),
        (75.0, 3.0, [2.1, 3.0, 4.6]),
        (100.0, 3.0, [2.1, 3.0, 4.6]),
        (150.0, 3.0, [2.1, 3.0, 4.6]),
        # Traditional Analytical HPLC (5.0 µm)
        (50.0, 5.0, [4.6]),
        (100.0, 5.0, [3.0, 4.6]),
        (150.0, 5.0, [2.1, 3.0, 4.6]),
        (250.0, 5.0, [3.0, 4.6]),
    ]

    for length, dp, id_list in fpp_configs:
        for inner_id in id_list:
            for ligand in ligands:
                inventory.append({
                    "length": float(length),
                    "dp": float(dp),
                    "is_spp": False,
                    "ligand": ligand,
                    "id": float(inner_id),
                })

    # Deduplicate configurations
    unique_inventory = []
    seen = set()
    for item in inventory:
        key = (
            item["length"],
            item["dp"],
            item["is_spp"],
            item["ligand"],
            item["id"],
        )
        if key not in seen:
            seen.add(key)
            unique_inventory.append(item)

    return unique_inventory


def initialize_column_inventory():
    """Ensures available_columns list exists in session state and is populated once."""
    if 'inventory_initialized' not in st.session_state:
        st.session_state.available_columns = get_default_column_inventory()
        st.session_state.inventory_initialized = True

    if 'available_columns' not in st.session_state:
        st.session_state.available_columns = []

    if 'available_ligands' not in st.session_state:
        st.session_state.available_ligands = [
            'C18', 'C8', 'Biphenyl', 'FluoroPhenyl', 'PFP', 
            'ARC-18', 'AQ-C18', 'Cyano', 'Phenyl-Hexyl', 'Silica'
        ]


def render_inventory_manager_page():
    """Renders a clean, user-friendly Column Hardware & Inventory Manager page."""
    initialize_column_inventory()

    st.title("📦 Column Hardware & Inventory Manager")
    st.caption("Manage physical in-stock laboratory columns and stationary phase chemistries.")
    st.divider()

    tab1, tab2 = st.tabs(["🧪 Physical Column Inventory", "🏷️ Stationary Phase Chemistries"])

    # --- TAB 1: PHYSICAL COLUMNS IN POSSESSION ---
    with tab1:
        st.subheader("Add Column to Inventory")

        with st.form(key="add_column_inventory_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                ligand = st.selectbox("Ligand / Chemistry", options=st.session_state.available_ligands, index=0)
                length = st.number_input("Length (mm)", min_value=10.0, max_value=300.0, value=100.0, step=10.0)

            with col2:
                particle_size = st.number_input("Particle Size (µm)", min_value=1.0, max_value=10.0, value=2.7, step=0.1)
                internal_diameter = st.selectbox("Internal Diameter (mm)", options=[2.1, 3.0, 4.6], index=1)

            with col3:
                tech_type = st.radio("Particle Technology", options=["Core-Shell (SPP)", "Fully Porous (FPP)"], index=0)
                is_spp = (tech_type == "Core-Shell (SPP)")

            submitted = st.form_submit_button("➕ Add Column to Inventory", type="primary")

            if submitted:
                new_col = {
                    'length': float(length),
                    'dp': float(particle_size),
                    'is_spp': is_spp,
                    'ligand': ligand,
                    'id': float(internal_diameter)
                }
                
                # Prevent exact duplicates
                if not any(c == new_col for c in st.session_state.available_columns):
                    st.session_state.available_columns.append(new_col)
                    st.success(f"Added **{length:.0f} x {internal_diameter:.1f} mm, {particle_size:.1f} µm {ligand} ({tech_type})**.")
                    st.rerun()
                else:
                    st.warning("This column configuration is already in your inventory.")

        st.divider()

        # Header controls with both Reset and Clear All buttons
        col_hdr, col_actions = st.columns([0.55, 0.45])
        with col_hdr:
            st.markdown(f"#### In-Stock Inventory ({len(st.session_state.available_columns)} Columns)")
        with col_actions:
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("🔄 Reset Defaults", help="Load full Restek commercial inventory"):
                    st.session_state.available_columns = get_default_column_inventory()
                    st.session_state.inventory_initialized = True
                    st.success("Reset inventory to commercial defaults.")
                    st.rerun()
            with btn_col2:
                if st.button("🗑️ Clear All", type="secondary", help="Remove all columns from inventory"):
                    st.session_state.available_columns = []
                    st.session_state.inventory_initialized = True  # Prevents initializer from re-populating defaults
                    st.success("Cleared all columns from inventory.")
                    st.rerun()

        if not st.session_state.available_columns:
            st.info("No columns currently in stock. All optimization candidates will be tagged as `[PURCHASE NEEDED]`.")
        else:
            # Display cards for existing inventory
            for idx, c in enumerate(st.session_state.available_columns):
                with st.container(border=True):
                    c_info, c_btn = st.columns([0.82, 0.18])
                    tech_str = "Core-Shell (SPP)" if c['is_spp'] else "Fully Porous (FPP)"
                    
                    with c_info:
                        st.markdown(f"**Column #{idx+1}: {c['length']:.0f} x {c['id']:.1f} mm, {c['dp']:.1f} µm {c['ligand']}**")
                        st.caption(f"Technology: `{tech_str}` | Internal Diameter: `{c['id']:.1f} mm`")

                    with c_btn:
                        if st.button("🗑️ Remove", key=f"del_col_{idx}"):
                            st.session_state.available_columns.pop(idx)
                            st.rerun()

    # --- TAB 2: LIGANDS / CHEMISTRIES ---
    with tab2:
        st.subheader("Manage Chemistries")
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.markdown("#### Add New Ligand")
            new_lig = st.text_input("Ligand Name", placeholder="e.g., C30, Cyano", key="input_new_lig")
            if st.button("➕ Add Ligand", key="btn_add_lig", type="primary"):
                if new_lig and new_lig not in st.session_state.available_ligands:
                    st.session_state.available_ligands.append(new_lig)
                    st.success(f"Added **{new_lig}**.")
                    st.rerun()

        with col_b:
            st.markdown("#### Current Ligands")
            for lig in st.session_state.available_ligands:
                st.write(f"- `{lig}`")


if __name__ == "__main__":
    render_inventory_manager_page()
