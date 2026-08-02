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
        "description": "High-income customers with premium coverage policies. Highly engaged with minimal churn risk.",
        "badge_color": "#059669",  # Emerald Green
        "bg_color": "#ECFDF5",
        "recommendation": "Cross-sell premium wealth management products and offer exclusive VIP loyalty rewards.",
    },
    1: {
        "title": "🎓 Young Digital Budget Seekers",
        "description": "Younger demographic looking for cost-effective basic coverage with digital-first interaction preferences.",
        "badge_color": "#0284C7",  # Sky Blue
        "bg_color": "#F0F9FF",
        "recommendation": "Offer flexible pay-as-you-go insurance options via mobile app and SMS notifications.",
    },
    2: {
        "title": "🛡️ Family Protection Focused",
        "description": "Mid-age married individuals prioritizing comprehensive health, vehicle, and life coverage for dependents.",
        "badge_color": "#7C3AED",  # Purple
        "bg_color": "#F5F3FF",
        "recommendation": "Promote bundled family coverage packages and long-term savings plans.",
    },
    3: {
        "title": "⚠️ High-Service Demand Segment",
        "description": "Frequent customer service interactions with moderate policy spend and higher retention sensitivity.",
        "badge_color": "#DC2626",  # Red
        "bg_color": "#FEF2F2",
        "recommendation": "Assign dedicated customer support reps to resolve inquiries quickly and offer targeted renewal discounts.",
    },
}

# ==========================
# Prediction Function
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
                "description": "Standard profile matching default cluster characteristics.",
                "badge_color": "#2563EB",
                "bg_color": "#EFF6FF",
                "recommendation": "Apply standard customer engagement and marketing strategy.",
            },
        )

        # Generate Rich HTML Persona Card Output
        html_output = f"""
        <div style="
            background-color: {persona['bg_color']}; 
            border: 2px solid {persona['badge_color']}; 
            border-radius: 12px; 
            padding: 22px; 
            margin-top: 10px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            font-family: 'Segoe UI', Tahoma, sans-serif;
        ">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                <span style="
                    background-color: {persona['badge_color']}; 
                    color: white; 
                    font-size: 14px; 
                    font-weight: 700; 
                    padding: 5px 14px; 
                    border-radius: 20px;
                ">
                    Cluster {cluster_id + 1}
                </span>
                <span style="color: #6B7280; font-size: 13px; font-weight: 600;">Prediction Complete</span>
            </div>

            <h2 style="color: #111827; font-size: 24px; font-weight: 700; margin: 0 0 8px 0;">
                {persona['title']}
            </h2>
            
            <p style="color: #374151; font-size: 15px; margin: 0 0 16px 0; line-height: 1.5;">
                {persona['description']}
            </p>

            <hr style="border: none; border-top: 1px solid rgba(0,0,0,0.12); margin: 14px 0;">

            <div style="display: flex; gap: 10px; align-items: flex-start;">
                <span style="font-size: 20px;">💡</span>
                <div>
                    <strong style="color: #111827; font-size: 15px;">Target Business Strategy:</strong>
                    <p style="color: #4B5563; font-size: 14px; margin: 4px 0 0 0; line-height: 1.4;">{persona['recommendation']}</p>
                </div>
            </div>
        </div>
        """
        return html_output

    except Exception as e:
        return f"<div style='color:red; font-weight:bold; padding:10px;'>Error processing prediction: {e}</div>"


# ==========================
# Custom Dashboard CSS
# ==========================

css = """
body, .gradio-container {
    background-color: #2F363D !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

/* Main Dashboard Header */
.main-header {
    background: #1B2127;
    padding: 24px 20px;
    border-radius: 14px;
    text-align: center;
    border: 1.5px solid #10B981;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.main-header h1 {
    color: #FFFFFF !important;
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 6px 0;
}

.main-header p {
    color: #9CA3AF !important;
    font-size: 15px;
    margin: 0 0 16px 0;
}

/* Developer Info Card */
.developer-card {
    background: #252D35;
    border-radius: 10px;
    padding: 10px 20px;
    max-width: 320px;
    margin: 0 auto;
    border: 1px solid #374151;
}

.dev-title {
    color: #D1D5DB;
    font-size: 14px;
    font-weight: 600;
}

.dev-name {
    color: #10B981;
    font-size: 20px;
    font-weight: 700;
}

.dev-roll {
    color: #FBBF24;
    font-size: 16px;
    font-weight: 700;
}

/* Input Form Sections */
.form-card {
    background: #FFFFFF !important;
    padding: 18px !important;
    border-radius: 12px !important;
    margin-bottom: 15px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}

label span {
    color: #1F2937 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* Submit Button */
button.primary-btn {
    background: #374151 !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    padding: 12px !important;
    border: 1px solid #4B5563 !important;
    transition: all 0.3s ease;
}

button.primary-btn:hover {
    background: #10B981 !important;
    border-color: #10B981 !important;
}
"""

# ==========================
# Gradio Interface Setup
# ==========================

with gr.Blocks(css=css, title="Customer Segmentation Analytics") as demo:

    # Header & Developer Card
    gr.HTML("""
    <div class="main-header">
        <h1>🛡️ Customer Segmentation Analytics Dashboard</h1>
        <p>Intelligent Customer Cluster Prediction using K-Prototypes</p>
        <div class="developer-card">
            <div class="dev-title">Developer Card</div>
            <div class="dev-name">👨‍💻 Developed By: Vansh Bareja</div>
            <div class="dev-roll">Roll No: 241047</div>
        </div>
    </div>
    """)

    # Group 1: Numerical Values
    with gr.Group(elem_classes=["form-card"]):
        with gr.Row():
            age = gr.Number(label="Age", value=35)
            income = gr.Number(label="Income Level", value=60000)
            coverage = gr.Number(label="Coverage Amount", value=250000)
            premium = gr.Number(label="Premium Amount", value=15000)

    # Group 2: Demographics
    with gr.Group(elem_classes=["form-card"]):
        with gr.Row():
            gender = gr.Dropdown(
                ["Male", "Female"], value="Male", label="Gender"
            )
            marital_status = gr.Dropdown(
                ["Single", "Married", "Divorced", "Widowed"],
                value="Single",
                label="Marital Status",
            )
            education = gr.Dropdown(
                ["High School", "Bachelor", "Master", "PhD"],
                value="Bachelor",
                label="Education Level",
            )
            geographic = gr.Dropdown(
                ["Urban", "Suburban", "Rural"],
                value="Urban",
                label="Geographic Information",
            )

    # Group 3: Behavior & Preferences
    with gr.Group(elem_classes=["form-card"]):
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
            behavioral = gr.Dropdown(
                ["Low", "Medium", "High"],
                value="Medium",
                label="Behavioral Data",
            )

        with gr.Row():
            interaction = gr.Dropdown(
                ["Low", "Medium", "High"],
                value="Medium",
                label="Interactions with Customer Service",
            )
            insurance_products = gr.Dropdown(
                ["1", "2", "3", "4", "5"],
                value="2",
                label="Insurance Products Owned",
            )

        with gr.Row():
            policy_type = gr.Dropdown(
                ["Health", "Life", "Vehicle", "Home", "Travel"],
                value="Health",
                label="Policy Type",
            )
            customer_preference = gr.Dropdown(
                ["Price", "Coverage", "Service", "Benefits"],
                value="Coverage",
                label="Customer Preferences",
            )

        with gr.Row():
            communication_channel = gr.Dropdown(
                ["Email", "Phone", "SMS", "Mobile App"],
                value="Email",
                label="Preferred Communication Channel",
            )
            contact_time = gr.Dropdown(
                ["Morning", "Afternoon", "Evening"],
                value="Morning",
                label="Preferred Contact Time",
            )
            language = gr.Dropdown(
                ["English", "Hindi", "Spanish", "French"],
                value="English",
                label="Preferred Language",
            )

    # Hidden Constant Input
    purchase_year = gr.Number(value=2024, visible=False)

    # Prediction Action
    predict_btn = gr.Button(
        "Predict Customer Segment", elem_classes=["primary-btn"]
    )

    # Output Persona Display
    output = gr.HTML(label="Prediction Result")

    # Event Listener
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

# Launch Server
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
