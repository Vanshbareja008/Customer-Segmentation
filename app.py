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
        # Numerical Features DataFrame
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

        # Categorical Features DataFrame
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
# Modern Custom CSS
# ==========================

css = """
body, .gradio-container {
    background-color: #2F363D !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

/* Dashboard Header */
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

/* Form Container Sections */
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

/* Inputs Styling */
input, select, .single-select {
    border-radius: 8px !important;
    border: 1px solid #D1D5DB !important;
}

/* Submit Button */
button.primary-btn {
    background: #374151 !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    font-size: 17px !weight: 600 !important;
    padding: 12px !important;
    border: 1px solid #4B5563 !important;
    transition: all 0.3s ease;
}

button.primary-btn:hover {
    background: #10B981 !important;
    border-color: #10B981 !important;
}

/* Output Box */
.result-display {
    background: #E6F4EA !important;
    border: 1.5px solid #10B981 !important;
    border-radius: 10px !important;
}

.result-display textarea {
    color: #047857 !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    text-align: left !important;
}
"""

# ==========================
# Gradio Interface Setup
# ==========================

with gr.Blocks(css=css, title="Customer Segmentation Analytics") as demo:

    # Header & Developer Badge
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

    # Group 1: Numerical Values Card
    with gr.Group(elem_classes=["form-card"]):
        with gr.Row():
            age = gr.Number(label="Age", value=35)
            income = gr.Number(label="Income Level", value=60000)
            coverage = gr.Number(label="Coverage Amount", value=250000)
            premium = gr.Number(label="Premium Amount", value=15000)

    # Group 2: Demographics Card
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

    # Group 3: Categorical Preferences Card
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

    # Hidden constant input for model mapping
    purchase_year = gr.Number(value=2024, visible=False)

    # Prediction Trigger
    predict_btn = gr.Button(
        "Predict Customer Segment", elem_classes=["primary-btn"]
    )

    # Output Card
    output = gr.Textbox(
        label="Prediction Result",
        elem_classes=["result-display"],
        interactive=False,
    )

    # Connect Trigger
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

# Launch Dashboard
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
