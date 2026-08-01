import gradio as gr
import joblib
import pandas as pd
import numpy as np

# ==========================
# Load Model Package
# ==========================

package = joblib.load("customer_segmentation_kprototypes.pkl")

model = package["model"]
scaler = package["scaler"]
numerical_columns = package["numerical_columns"]
categorical_columns = package["categorical_columns"]
categorical_indices = package["categorical_indices"]


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
    language
):

    try:

        # Numerical Features
        numeric_df = pd.DataFrame({
            "Age":[age],
            "Income Level":[income],
            "Coverage Amount":[coverage],
            "Premium Amount":[premium],
            "Purchase Year":[purchase_year]
        })

        # Scale Numerical Features
        scaled_numeric = scaler.transform(numeric_df)

        # Categorical Features
        categorical_df = pd.DataFrame({
            "Gender":[gender],
            "Marital Status":[marital_status],
            "Education Level":[education],
            "Geographic Information":[geographic],
            "Occupation":[occupation],
            "Behavioral Data":[behavioral],
            "Interactions with Customer Service":[interaction],
            "Insurance Products Owned":[insurance_products],
            "Policy Type":[policy_type],
            "Customer Preferences":[customer_preference],
            "Preferred Communication Channel":[communication_channel],
            "Preferred Contact Time":[contact_time],
            "Preferred Language":[language]
        })

        # Merge Numeric + Categorical
        X = np.concatenate(
            [
                scaled_numeric,
                categorical_df.values
            ],
            axis=1
        )

        # Predict Cluster
        cluster = model.predict(
            X,
            categorical=categorical_indices
        )[0]

        return f"🎯 Predicted Customer Segment : Cluster {cluster + 1}"

    except Exception as e:
        return f"Error : {e}"

# ==========================
# Gradio Interface
# ==========================

css = """
/* =========================
   Professional Business Theme
========================= */

body{
    background:#F5F5F5;
}

.gradio-container{
    background:#F5F5F5;
    font-family:"Segoe UI",Arial,sans-serif;
    color:#1F2937;
}

/* =========================
   Header
========================= */

.main-header{
    background:#2C2C2C;
    color:white;
    padding:28px;
    border-radius:14px;
    text-align:center;
    margin-bottom:22px;
    border-left:6px solid #16A34A;
    box-shadow:0 5px 15px rgba(0,0,0,.08);
}

.main-header h1{
    margin:0;
    font-size:34px;
    font-weight:700;
}

.main-header p{
    margin-top:10px;
    color:#D1D5DB;
}

/* =========================
   Cards
========================= */

.section{
    background:white;
    border-radius:12px;
    padding:22px;
    margin:18px 0;
    border:1px solid #E5E7EB;
    box-shadow:0 3px 10px rgba(0,0,0,.05);
}

/* =========================
   Inputs
========================= */

.gr-textbox textarea,
.gr-number input,
.gr-dropdown{
    border-radius:8px !important;
    border:1px solid #D1D5DB !important;
}

.gr-textbox textarea:focus,
.gr-number input:focus{
    border-color:#16A34A !important;
    box-shadow:0 0 0 2px rgba(22,163,74,.15) !important;
}

/* =========================
   Button
========================= */

button.primary{
    background:#2C2C2C !important;
    color:white !important;
    border:none !important;
    border-radius:10px !important;
    font-size:16px;
    font-weight:600;
    padding:12px 20px;
    transition:.25s;
}

button.primary:hover{
    background:#16A34A !important;
}

/* =========================
   Result
========================= */

.result-box textarea{
    background:#F0FDF4 !important;
    color:#166534 !important;
    font-size:20px !important;
    font-weight:700 !important;
    border:1px solid #BBF7D0 !important;
}

/* =========================
   Footer
========================= */

.footer{
    text-align:center;
    color:#6B7280;
    margin-top:25px;
}

/* =========================
   Scrollbar
========================= */

::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-thumb{
    background:#BDBDBD;
    border-radius:20px;
}
"""

with gr.Blocks(
 theme = gr.themes.Monochrome(
    primary_hue="gray",
    secondary_hue="gray",
    neutral_hue="gray",
),
    with gr.Blocks(theme=theme, css=css) as demo:

    gr.Markdown("""
<div class="main-header">

# 🛡️ Customer Segmentation Analytics Dashboard

### Intelligent Customer Cluster Prediction using K-Prototypes

Developed by **Vansh Bareja**

Roll No. **241047**

</div>
""")

    with gr.Row():

        age = gr.Number(
            label="Age",
            value=35
        )

        income = gr.Number(
            label="Income Level",
            value=60000
        )

    with gr.Row():

        coverage = gr.Number(
            label="Coverage Amount",
            value=250000
        )

        premium = gr.Number(
            label="Premium Amount",
            value=15000
        )

    purchase_year = gr.Number(
        label="Purchase Year",
        value=2024
    )

    gr.Markdown("## 📝 Customer Information")

    with gr.Row():

        gender = gr.Dropdown(
            ["Male", "Female"],
            value="Male",
            label="Gender"
        )

        marital_status = gr.Dropdown(
            ["Single", "Married", "Divorced", "Widowed"],
            value="Single",
            label="Marital Status"
        )

    with gr.Row():

        education = gr.Dropdown(
            [
                "High School",
                "Bachelor",
                "Master",
                "PhD"
            ],
            value="Bachelor",
            label="Education Level"
        )

        geographic = gr.Dropdown(
            [
                "Urban",
                "Suburban",
                "Rural"
            ],
            value="Urban",
            label="Geographic Information"
        )

    occupation = gr.Dropdown(
        [
            "Employed",
            "Business",
            "Self-Employed",
            "Student",
            "Retired",
            "Unemployed"
        ],
        value="Employed",
        label="Occupation"
    )

    behavioral = gr.Dropdown(
        [
            "Low",
            "Medium",
            "High"
        ],
        value="Medium",
        label="Behavioral Data"
    )

    interaction = gr.Dropdown(
        [
            "Low",
            "Medium",
            "High"
        ],
        value="Medium",
        label="Interactions with Customer Service"
    )

    insurance_products = gr.Dropdown(
        [
            "1",
            "2",
            "3",
            "4",
            "5"
        ],
        value="2",
        label="Insurance Products Owned"
    )

    policy_type = gr.Dropdown(
        [
            "Health",
            "Life",
            "Vehicle",
            "Home",
            "Travel"
        ],
        value="Health",
        label="Policy Type"
    )

    customer_preference = gr.Dropdown(
        [
            "Price",
            "Coverage",
            "Service",
            "Benefits"
        ],
        value="Coverage",
        label="Customer Preferences"
    )

    communication_channel = gr.Dropdown(
        [
            "Email",
            "Phone",
            "SMS",
            "Mobile App"
        ],
        value="Email",
        label="Preferred Communication Channel"
    )

    contact_time = gr.Dropdown(
        [
            "Morning",
            "Afternoon",
            "Evening"
        ],
        value="Morning",
        label="Preferred Contact Time"
    )

    language = gr.Dropdown(
        [
            "English",
            "Hindi",
            "Spanish",
            "French"
        ],
        value="English",
        label="Preferred Language"
    )

    predict_btn = gr.Button(
        "Predict Customer Segment",
        variant="primary"
    )

    output = gr.Textbox(
        label="Prediction Result"
    )
# ==========================
# Button Action
# ==========================

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
            language
        ],
        outputs=output
    )

    # ==========================
    # Example Values
    # ==========================

    gr.Examples(
        examples=[
            [
                35,
                65000,
                300000,
                18000,
                2023,
                "Male",
                "Married",
                "Bachelor",
                "Urban",
                "Employed",
                "High",
                "Medium",
                "2",
                "Health",
                "Coverage",
                "Email",
                "Morning",
                "English"
            ],
            [
                48,
                90000,
                600000,
                32000,
                2022,
                "Female",
                "Married",
                "Master",
                "Suburban",
                "Business",
                "Medium",
                "High",
                "3",
                "Life",
                "Benefits",
                "Phone",
                "Evening",
                "Hindi"
            ],
            [
                27,
                45000,
                150000,
                9000,
                2024,
                "Male",
                "Single",
                "Bachelor",
                "Urban",
                "Self-Employed",
                "Low",
                "Low",
                "1",
                "Vehicle",
                "Price",
                "SMS",
                "Afternoon",
                "English"
            ]
        ],
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
            language
        ]
    )

    gr.Markdown(
        """
        ---
        ### 📌 Instructions
        - Enter customer information.
        - Click **Predict Customer Segment**.
        - The trained K-Prototypes model will predict the most suitable cluster.

        **Developed by Vansh Bareja**  
        **Roll No. 241047**
        """
    )


# ==========================
# Launch App
# ==========================

demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)
