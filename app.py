import gradio as gr
import joblib
import numpy as np
import pandas as pd

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
    language,
):
    try:
        # Numerical Features
        numeric_df = pd.DataFrame(
            {
                "Age": [age],
                "Income Level": [income],
                "Coverage Amount": [coverage],
                "Premium Amount": [premium],
                "Purchase Year": [purchase_year],
            }
        )

        # Scale Numerical Features
        scaled_numeric = scaler.transform(numeric_df)

        # Categorical Features
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
        X = np.concatenate(
            [scaled_numeric, categorical_df.values], axis=1
        )

        # Predict Cluster
        cluster = model.predict(X, categorical=categorical_indices)[0]

        return f"🎯 Predicted Customer Segment : Cluster {cluster + 1}"

    except Exception as e:
        return f"Error : {e}"


# ==========================
# Gradio Custom Styling
# ==========================

css = """
body {
    background-color: #ECECEC;
}

.gradio-container {
    background-color: #ECECEC !important;
    font-family: 'Segoe UI', Arial, sans-serif;
}

/* Main Header Card */
.main-header {
    background-color: #2B2B2B;
    padding: 30px 20px;
    border-radius: 16px;
    text-align: center;
    border-left: 8px solid #1B8A5A;
    box-shadow: 0 8px 18px rgba(0,0,0,0.12);
    margin-bottom: 20px;
}

.main-header h1 {
    color: #FFFFFF !important;
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 8px;
}

.main-header h3 {
    color: #E8E8E8 !important;
    font-weight: 400;
    margin-bottom: 20px;
}

/* Inner Developer Card */
.developer-card {
    background: #3B3B3B;
    border-radius: 12px;
    padding: 16px;
    max-width: 320px;
    margin: 0 auto;
    border: 1px solid #5A5A5A;
    text-align: center;
}

.dev-label {
    color: #FFFFFF !important;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 4px;
}

.dev-name {
    color: #1B8A5A !important;
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 12px;
}

.roll-label {
    color: #FFFFFF !important;
    font-size: 16px;
    font-weight: 600;
}

.roll-number {
    color: #FFD700 !important;
    font-size: 24px;
    font-weight: 700;
    margin-top: 2px;
}
"""

# ==========================
# Gradio Interface
# ==========================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="gray",
        neutral_hue="gray",
    ),
    css=css,
    title="Customer Segmentation System",
) as demo:

    # Clean HTML Render Header (Using gr.HTML instead of gr.Markdown)
    gr.HTML("""
    <div class="main-header">
        <h1>🛡️ Customer Segmentation Analytics Dashboard</h1>
        <h3>Intelligent Customer Cluster Prediction using K-Prototypes</h3>
        <div class="developer-card">
            <div class="dev-label">👨‍💻 Developed By</div>
            <div class="dev-name">Vansh Bareja</div>
            <div class="roll-label">Roll No.</div>
            <div class="roll-number">241047</div>
        </div>
    </div>
    """)

    with gr.Row():
        age = gr.Number(label="Age", value=35)
        income = gr.Number(label="Income Level", value=60000)

    with gr.Row():
        coverage = gr.Number(label="Coverage Amount", value=250000)
        premium = gr.Number(label="Premium Amount", value=15000)

    purchase_year = gr.Number(label="Purchase Year", value=2024)

    gr.Markdown("## 📝 Customer Information")

    with gr.Row():
        gender = gr.Dropdown(["Male", "Female"], value="Male", label="Gender")
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
            label="Geographic Information",
        )

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
        ["Low", "Medium", "High"], value="Medium", label="Behavioral Data"
    )

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

    predict_btn = gr.Button("Predict Customer Segment", variant="primary")

    output = gr.Textbox(label="Prediction Result")

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
            language,
        ],
        outputs=output,
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
                "English",
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
                "Hindi",
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
                "English",
            ],
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
            language,
        ],
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

demo.launch(server_name="0.0.0.0", server_port=7860)
