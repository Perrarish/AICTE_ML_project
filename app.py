from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

app = Flask(__name__)

# Load model and dataset
model_path = r"C:\Desktop\Code\E-waste_predction\random_forest_model.pkl"
data_path = r"C:\Desktop\Code\E-waste_predction\e_waste_india_realistic_2005_2024.csv"

with open(model_path, 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv(data_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        state_input = request.form['state'].strip()
        year_input = int(request.form['year'])

        state_data = df[df['Region'].str.lower() == state_input.lower()]

        if state_data.empty:
            return render_template('index.html', prediction_text="⚠️ State not found in dataset.")

        # Mean of numeric columns for the selected state
        state_mean = state_data.select_dtypes(include=['number']).mean()
        last_year = state_data['Year'].max()
        years_to_predict = max(0, year_input - last_year)

        # ===== Future estimations =====
        pop_future = state_mean['Population_M'] * (1 + 0.013 * years_to_predict)
        gdp_future = state_mean['GDP_per_capita_Lakh'] * (1 + 0.055 * years_to_predict)  # already in Lakhs
        urb_future = min(100, state_mean['Urbanization_Rate'] + 0.6 * years_to_predict)
        dev_future = state_mean['Devices_Per_Person'] * (1 + 0.04 * years_to_predict)
        rec_future = min(98, state_mean['Recycling_Rate'] + 0.45 * years_to_predict)

        # ===== E-waste calculation =====
        total_devices = pop_future * 1_000_000 * dev_future
        obsolete_devices = total_devices * 0.20
        avg_device_weight = 2.0
        predicted_ewaste = (obsolete_devices * avg_device_weight) / 1000  # in tons
        recycled = predicted_ewaste * (rec_future / 100)
        unmanaged = predicted_ewaste - recycled

        # ===== Graph: Line Chart =====
        past = state_data.groupby('Year')['Ewaste_Tons'].mean().reset_index()
        future_df = pd.DataFrame({'Year': [year_input], 'Ewaste_Tons': [predicted_ewaste]})
        combined = pd.concat([past, future_df]).sort_values(by='Year')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=combined['Year'],
            y=combined['Ewaste_Tons'],
            mode='lines+markers',
            line=dict(color='orange', width=4),
            marker=dict(size=8, color='red', line=dict(width=1, color='white')),
            name="E-waste Growth"
        ))

        fig.add_annotation(
            x=year_input, y=predicted_ewaste,
            text=f"{predicted_ewaste:,.0f} tons",
            showarrow=True, arrowhead=2, ay=-40, bgcolor="rgba(255,255,255,0.3)"
        )

        fig.update_layout(
            title=f"E-Waste Evolution in {state_input.title()} ({year_input})",
            xaxis_title="Year",
            yaxis_title="E-waste (tons)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0.8)",
            plot_bgcolor="rgba(0,0,0,0.8)",
            font=dict(color="white"),
            height=500
        )

        line_chart = fig.to_html(full_html=False)

        # ===== Pie Chart: Recycled vs Unmanaged =====
        fig2 = go.Figure(data=[go.Pie(
            labels=['♻️ Recycled', '⚠️ Unmanaged'],
            values=[recycled, unmanaged],
            hole=.4,
            marker=dict(colors=['#00cc96', '#ff6b6b']),
            textinfo='label+percent',
            textfont_size=15
        )])
        fig2.update_layout(
            title=f"E-waste Composition in {year_input}",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0.8)",
            font=dict(color="white"),
            height=400
        )

        pie_chart = fig2.to_html(full_html=False)

        # ===== CLEAN TEXT FOR FRONTEND PARSING =====
        prediction_text = (
            f"Estimated Population: {pop_future:.2f} M\n"
            f"GDP per Capita: ₹{gdp_future:.2f} Lakh\n"
            f"Urbanization Rate: {urb_future:.2f}%\n"
            f"Devices per Person: {dev_future:.2f}\n"
            f"Recycling Rate: {rec_future:.2f}%\n\n"
            f"♻️ Total E-waste Generated: {predicted_ewaste:,.2f} tons/year\n"
            f"✅ Recycled: {recycled:,.2f} tons/year\n"
            f"⚠️ Unmanaged/Dumped: {unmanaged:,.2f} tons/year"
        )

        return render_template(
            'index.html',
            prediction_text=prediction_text,
            line_chart=line_chart,
            pie_chart=pie_chart
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
