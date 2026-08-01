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
/* ========= Global ========= */

body{
    background:#F3F6FA;
}

.gradio-container{
    background:#F3F6FA;
    font-family: "Segoe UI", Arial, sans-serif;
}

/* ========= Header ========= */

.main-header{
    background:#1E3A8A;
    color:white;
    padding:28px;
    border-radius:14px;
    text-align:center;
    margin-bottom:25px;
    box-shadow:0 4px 15px rgba(0,0,0,0.08);
}

.main-header h1{
    margin-bottom:8px;
}

.main-header p{
    color:#D6E4FF;
}

/* ========= Cards ========= */

.section{
    background:white;
    border-radius:12px;
    padding:22px;
    margin-top:18px;
    margin-bottom:18px;
    border:1px solid #E5E7EB;
    box-shadow:0 2px 10px rgba(0,0,0,0.05);
}

/* ========= Inputs ========= */

.gr-textbox,
.gr-dropdown,
.gr-number{
    border-radius:8px !important;
}

.gr-textbox textarea,
.gr-number input,
.gr-dropdown{
    border:1px solid #CBD5E1 !important;
}

.gr-textbox textarea:focus,
.gr-number input:focus{
    border-color:#2563EB !important;
    box-shadow:0 0 0 2px rgba(37,99,235,.15) !important;
}

/* ========= Buttons ========= */

button.primary{
    background:#2563EB !important;
    border:none !important;
    color:white !important;
    font-weight:600;
    border-radius:10px !important;
    padding:12px 24px !important;
    transition:.2s;
}

button.primary:hover{
    background:#1D4ED8 !important;
}

/* ========= Prediction Result ========= */

.result-box textarea{
    background:#ECFDF5 !important;
    color:#065F46 !important;
    font-size:20px !important;
    font-weight:700 !important;
    border:1px solid #A7F3D0 !important;
    border-radius:10px !important;
}

/* ========= Example Table ========= */

.gr-samples{
    border-radius:10px;
    overflow:hidden;
    border:1px solid #E5E7EB;
}

/* ========= Footer ========= */

.footer{
    text-align:center;
    color:#6B7280;
    font-size:14px;
    padding:25px;
}

/* ========= Scrollbar ========= */

::-webkit-scrollbar{
    width:10px;
}

::-webkit-scrollbar-thumb{
    background:#CBD5E1;
    border-radius:20px;
}

::-webkit-scrollbar-thumb:hover{
    background:#94A3B8;
}
"""

with gr.Blocks(
 theme = gr.themes.Monochrome(
    primary_hue="slate",
    secondary_hue="gray",
    neutral_hue="gray",
),
    css=css,
    title="Customer Segmentation System"
) as demo:

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
