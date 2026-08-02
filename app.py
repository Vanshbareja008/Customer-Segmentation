import gradio as gr
import joblib
import numpy as np
import pandas as pd

# ==========================
# Load Model Package
# ==========================

try:
    package = joblib.load("customer_segmentation_kprototypes.pkl")
    model = package["model"]
    scaler = package["scaler"]
    numerical_columns = package["numerical_columns"]
    categorical_columns = package["categorical_columns"]
    categorical_indices = package["categorical_indices"]
except Exception as e:
    print(f"Warning: Model file loading failed. ({e})")

# ==========================================
# Cluster Metadata & Persona Mappings
# ==========================================

CLUSTER_PERSONAS = {
    0: {
        "title": "💎 High-Value Loyalists",
        "subtitle": "Tier 1 Premium Segment",
        "description": "High-income customers with premium coverage policies. Highly engaged with minimal churn risk.",
        "badge_color": "#10B981",
        "bg_color": "rgba(16, 185, 129, 0.08)",
        "border_color": "rgba(16, 185, 129, 0.4)",
        "recommendation": "Cross-sell premium wealth management products and offer exclusive VIP loyalty rewards.",
        "retention_score": "96%",
        "value_tier": "High ($$$)",
    },
    1: {
        "title": "🎓 Young Digital Budget Seekers",
        "subtitle": "Growth & Mobile-First Segment",
        "description": "Younger demographic looking for cost-effective basic coverage with digital-first interaction preferences.",
        "badge_color": "#0284C7",
        "bg_color": "rgba(2, 132, 199, 0.08)",
        "border_color": "rgba(2, 132, 199, 0.4)",
        "recommendation": "Offer flexible pay-as-you-go insurance options via mobile app and SMS notifications.",
        "retention_score": "74%",
        "value_tier": "Moderate ($)",
    },
    2: {
        "title": "🛡️ Family Protection Focused",
        "subtitle": "Core Life & Health Segment",
        "description": "Mid-age married individuals prioritizing comprehensive health, vehicle, and life coverage for dependents.",
        "badge_color": "#8B5CF6",
        "bg_color": "rgba(139, 92, 246, 0.08)",
        "border_color": "rgba(139, 92, 246, 0.4)",
        "recommendation": "Promote bundled family coverage packages and long-term savings plans.",
        "retention_score": "88%",
        "value_tier": "Medium-High ($$)",
    },
    3: {
        "title": "⚠️ High-Service Demand Segment",
        "subtitle": "Attention Required / Retention Risk",
        "description": "Frequent customer service interactions with moderate policy spend and higher retention sensitivity.",
        "badge_color": "#EF4444",
        "bg_color": "rgba(239, 68, 68, 0.08)",
        "border_color": "rgba(239, 68, 68, 0.4)",
        "recommendation": "Assign dedicated customer support reps to resolve inquiries quickly and offer targeted renewal discounts.",
        "retention_score": "52%",
        "value_tier": "Sensitive ($$)",
    },
}

# ==========================
# Real-Time KPI Function
# ==========================


def update_live_metrics(income, coverage, premium, interaction):
    loss_ratio = (
        round((premium / coverage) * 100, 2) if coverage and coverage > 0 else 0
    )

    risk_label = "🟢 Low Risk"
    risk_color = "#34D399"
    if interaction == "High" or loss_ratio > 8:
        risk_label = "🔴 High Attention"
        risk_color = "#FCA5A5"
    elif interaction == "Medium" or loss_ratio > 4:
        risk_label = "🟡 Moderate Attention"
        risk_color = "#FCD34D"

    summary_html = f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 5px;">
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); padding: 12px 10px; border-radius: 12px; text-align: center;">
            <div style="color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Premium / Coverage Ratio</div>
            <div style="color: #38BDF8; font-size: 20px; font-weight: 700; margin-top: 4px;">{loss_ratio}%</div>
        </div>
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); padding: 12px 10px; border-radius: 12px; text-align: center;">
            <div style="color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Service Touch level</div>
            <div style="color: #FBBF24; font-size: 18px; font-weight: 700; margin-top: 5px;">{interaction}</div>
        </div>
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); padding: 12px 10px; border-radius: 12px; text-align: center;">
            <div style="color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Retention Radar</div>
            <div style="color: {risk_color}; font-size: 15px; font-weight: 700; margin-top: 6px;">{risk_label}</div>
        </div>
    </div>
    """
    return summary_html


# ==========================
# Main Prediction Function
# ==========================


def predict_customer(
    age,
    income,
    coverage,
    premium,
    purchase_year,
    gender,
    marital_status,
    education,
    geographic,
    occupation,
    behavioral,
    interaction,
    insurance_products,
    policy_type,
    customer_preference,
    communication_channel,
    contact_time,
    language,
):
    try:
        # Prepare Numerical Features
        numeric_df = pd.DataFrame(
            {
                "Age": [age],
                "Income Level": [income],
                "Coverage Amount": [coverage],
                "Premium Amount": [premium],
                "Purchase Year": [purchase_year],
            }
        )
        scaled_numeric = scaler.transform(numeric_df)

        # Prepare Categorical Features
        categorical_df = pd.DataFrame(
            {
                "Gender": [gender],
                "Marital Status": [marital_status],
                "Education Level": [education],
                "Geographic Information": [geographic],
                "Occupation": [occupation],
                "Behavioral Data": [behavioral],
                "Interactions with Customer Service": [interaction],
                "Insurance Products Owned": [insurance_products],
                "Policy Type": [policy_type],
                "Customer Preferences": [customer_preference],
                "Preferred Communication Channel": [communication_channel],
                "Preferred Contact Time": [contact_time],
                "Preferred Language": [language],
            }
        )

        # Merge Numeric + Categorical
        X = np.concatenate([scaled_numeric, categorical_df.values], axis=1)

        # Predict Cluster Index
        cluster_id = int(model.predict(X, categorical=categorical_indices)[0])

        # Retrieve Persona Profile
        persona = CLUSTER_PERSONAS.get(
            cluster_id,
            {
                "title": f"Cluster {cluster_id + 1} Segment",
                "subtitle": "Standard Classification",
                "description": "Standard profile matching default cluster characteristics.",
                "badge_color": "#3B82F6",
                "bg_color": "rgba(59, 130, 246, 0.08)",
                "border_color": "rgba(59, 130, 246, 0.4)",
                "recommendation": "Apply standard customer engagement and marketing strategy.",
                "retention_score": "80%",
                "value_tier": "Standard",
            },
        )

        # Dashboard Persona Card Output HTML
        html_output = f"""
        <div style="
            background: {persona['bg_color']}; 
            border: 1.5px solid {persona['border_color']}; 
            border-radius: 16px; 
            padding: 24px; 
            margin-top: 10px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.4);
            backdrop-filter: blur(12px);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        ">
            <!-- Top Status Bar -->
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
                <span style="
                    background-color: {persona['badge_color']}; 
                    color: #FFFFFF; 
                    font-size: 12px; 
                    font-weight: 800; 
                    letter-spacing: 0.6px;
                    padding: 6px 14px; 
                    border-radius: 20px;
                    text-transform: uppercase;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
                ">
                    Cluster {cluster_id + 1} Assigned
                </span>
                <span style="color: #94A3B8; font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 6px;">
                    <span style="height: 8px; width: 8px; background-color: #10B981; border-radius: 50%; display: inline-block;"></span>
                    Scoring Engine Active
                </span>
            </div>

            <!-- Header Info -->
            <h2 style="color: #FFFFFF; font-size: 26px; font-weight: 700; margin: 0 0 4px 0; letter-spacing: -0.5px;">
                {persona['title']}
            </h2>
            <div style="color: {persona['badge_color']}; font-size: 14px; font-weight: 600; margin-bottom: 14px;">
                {persona['subtitle']}
            </div>
            
            <p style="color: #CBD5E1; font-size: 14px; margin: 0 0 20px 0; line-height: 1.6;">
                {persona['description']}
            </p>

            <!-- Key Persona Metrics Grid -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px;">
                <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255,255,255,0.06); padding: 14px; border-radius: 12px;">
                    <span style="color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; display: block; margin-bottom: 4px;">Predicted Retention Rate</span>
                    <span style="color: #FFFFFF; font-size: 20px; font-weight: 700;">{persona['retention_score']}</span>
                </div>
                <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255,255,255,0.06); padding: 14px; border-radius: 12px;">
                    <span style="color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; display: block; margin-bottom: 4px;">Segment Profitability Tier</span>
                    <span style="color: #FFFFFF; font-size: 20px; font-weight: 700;">{persona['value_tier']}</span>
                </div>
            </div>

            <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 18px 0;">

            <!-- Strategy Recommendation -->
            <div style="display: flex; gap: 14px; align-items: flex-start; background: rgba(15, 23, 42, 0.6); padding: 16px; border-radius: 12px; border-left: 4px solid {persona['badge_color']};">
                <span style="font-size: 22px; line-height: 1;">💡</span>
                <div>
                    <strong style="color: #FFFFFF; font-size: 14px; display: block; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px;">Recommended Action Plan:</strong>
                    <p style="color: #94A3B8; font-size: 14px; margin: 0; line-height: 1.5;">{persona['recommendation']}</p>
                </div>
            </div>
        </div>
        """
        return html_output

    except Exception as e:
        return f"""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #EF4444; color: #FCA5A5; padding: 16px; border-radius: 12px; font-family: sans-serif;">
            ⚠️ <strong>Model Execution Notice:</strong> {e}
        </div>
        """


# ==========================================
# Custom High-End Styling CSS
# ==========================================

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global Reset & Dark Mode Theme */
body, .gradio-container {
    background-color: #090D16 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #F3F4F6 !important;
}

/* Glassmorphism Header */
.main-header {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 24px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
}

.main-header h1 {
    color: #FFFFFF !important;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
}

.main-header p {
    color: #94A3B8 !important;
    font-size: 14px;
    margin: 0 0 16px 0;
}

/* Developer Pill Badge */
.developer-card {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 30px;
    padding: 6px 18px;
    display: inline-flex;
    gap: 14px;
    align-items: center;
}

.dev-info {
    color: #CBD5E1;
    font-size: 13px;
    font-weight: 500;
}

.dev-info span {
    color: #10B981;
    font-weight: 700;
}

/* Form Container Style */
.form-card {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    padding: 18px !important;
    margin-bottom: 16px !important;
}

.card-title {
    color: #10B981;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Custom Input Field Integration */
.gr-box, .gr-input, label {
    background-color: transparent !important;
    border: none !important;
}

input, select, textarea {
    background-color: #0F172A !important;
    border: 1px solid #334155 !important;
    color: #F8FAFC !important;
    border-radius: 8px !important;
}

input:focus, select:focus {
    border-color: #10B981 !important;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2) !important;
}

label span {
    color: #94A3B8 !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Action Predict Button */
button.primary-btn {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 14px !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    width: 100% !important;
}

button.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
}
"""

# ==========================
# Gradio Interface Setup
# ==========================

with gr.Blocks(css=css, title="Customer Analytics AI Engine") as demo:

    # Main Dashboard Header
    gr.HTML("""
    <div class="main-header">
        <h1>🛡️ Customer Segmentation Intelligence</h1>
        <p>Enterprise Machine Learning Dashboard for Real-Time Persona Scoring</p>
        <div class="developer-card">
            <div class="dev-info">Developer: <span>Vansh Bareja</span></div>
            <div class="dev-info">•</div>
            <div class="dev-info">Roll No: <span>241047</span></div>
        </div>
    </div>
    """)

    # 2-Column Master Layout
    with gr.Row(equal_height=False):

        # Left Column: Form Inputs (Width: 3/5)
        with gr.Column(scale=3):

            # Financial & Policy Section
            with gr.Group(elem_classes=["form-card"]):
                gr.HTML(
                    '<div class="card-title">💵 Financial Profile & Coverage</div>'
                )
                with gr.Row():
                    age = gr.Slider(
                        label="Age", minimum=18, maximum=80, value=35, step=1
                    )
                    income = gr.Number(label="Income Level ($)", value=60000)
                with gr.Row():
                    coverage = gr.Number(
                        label="Coverage Amount ($)", value=250000
                    )
                    premium = gr.Number(label="Premium Amount ($)", value=15000)

            # Demographics Section
            with gr.Group(elem_classes=["form-card"]):
                gr.HTML(
                    '<div class="card-title">👤 Demographics & Background</div>'
                )
                with gr.Row():
                    gender = gr.Dropdown(
                        ["Male", "Female"], value="Male", label="Gender"
                    )
                    marital_status = gr.Dropdown(
                        ["Single", "Married", "Divorced", "Widowed"],
                        value="Single",
                        label="Marital Status",
                    )
                with gr.Row():
                    education = gr.Dropdown(
                        ["High School", "Bachelor", "Master", "PhD"],
                        value="Bachelor",
                        label="Education Level",
                    )
                    geographic = gr.Dropdown(
                        ["Urban", "Suburban", "Rural"],
                        value="Urban",
                        label="Geography",
                    )

            # Behavioral & Account Profile
            with gr.Group(elem_classes=["form-card"]):
                gr.HTML(
                    '<div class="card-title">⚙️ Behavioral & Policy Details</div>'
                )
                with gr.Row():
                    occupation = gr.Dropdown(
                        [
                            "Employed",
                            "Business",
                            "Self-Employed",
                            "Student",
                            "Retired",
                            "Unemployed",
                        ],
                        value="Employed",
                        label="Occupation",
                    )
                    policy_type = gr.Dropdown(
                        ["Health", "Life", "Vehicle", "Home", "Travel"],
                        value="Health",
                        label="Policy Type",
                    )
                with gr.Row():
                    behavioral = gr.Dropdown(
                        ["Low", "Medium", "High"],
                        value="Medium",
                        label="Activity Score",
                    )
                    interaction = gr.Dropdown(
                        ["Low", "Medium", "High"],
                        value="Medium",
                        label="Service Frequency",
                    )
                    insurance_products = gr.Dropdown(
                        ["1", "2", "3", "4", "5"],
                        value="2",
                        label="Products Owned",
                    )

            # Preferences Section
            with gr.Group(elem_classes=["form-card"]):
                gr.HTML(
                    '<div class="card-title">📱 Channel & Interaction Preferences</div>'
                )
                with gr.Row():
                    customer_preference = gr.Dropdown(
                        ["Price", "Coverage", "Service", "Benefits"],
                        value="Coverage",
                        label="Primary Value Priority",
                    )
                    communication_channel = gr.Dropdown(
                        ["Email", "Phone", "SMS", "Mobile App"],
                        value="Email",
                        label="Preferred Channel",
                    )
                with gr.Row():
                    contact_time = gr.Dropdown(
                        ["Morning", "Afternoon", "Evening"],
                        value="Morning",
                        label="Preferred Time",
                    )
                    language = gr.Dropdown(
                        ["English", "Hindi", "Spanish", "French"],
                        value="English",
                        label="Language",
                    )

            # Hidden Constant Input
            purchase_year = gr.Number(value=2024, visible=False)

        # Right Column: Output & Real-Time Analytics Panel (Width: 2/5)
        with gr.Column(scale=2):

            # Real-Time Telemetry Bar
            with gr.Group(elem_classes=["form-card"]):
                gr.HTML(
                    '<div class="card-title">📊 Live Telemetry Bar</div>'
                )
                live_kpi_display = gr.HTML()

            # Execute Primary Prediction Button
            predict_btn = gr.Button(
                "⚡ Score Customer Segment", elem_classes=["primary-btn"]
            )

            # High-Impact Persona Display Output
            with gr.Group(elem_classes=["form-card"]):
                gr.HTML(
                    '<div class="card-title">🎯 Cluster Persona Insights</div>'
                )
                output = gr.HTML()

    # ==========================
    # Dynamic Event Listeners
    # ==========================

    # Real-time Telemetry Listener Setup
    kpi_inputs = [income, coverage, premium, interaction]
    for inp in kpi_inputs:
        inp.change(
            fn=update_live_metrics, inputs=kpi_inputs, outputs=live_kpi_display
        )

    # Initial Telemetry Render on Load
    demo.load(
        fn=update_live_metrics, inputs=kpi_inputs, outputs=live_kpi_display
    )

    # Prediction Action Listener
    predict_btn.click(
        fn=predict_customer,
        inputs=[
            age,
            income,
            coverage,
            premium,
            purchase_year,
            gender,
            marital_status,
            education,
            geographic,
            occupation,
            behavioral,
            interaction,
            insurance_products,
            policy_type,
            customer_preference,
            communication_channel,
            contact_time,
            language,
        ],
        outputs=output,
    )

# Launch Dashboard Application
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
