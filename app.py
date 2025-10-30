from xgboost import XGBRegressor
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

def downcasting(data: pd.DataFrame, verbose: bool=True) -> pd.DataFrame:
    mem_before = data.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe is {mem_before:.2f} MB")
            
    for col in data.select_dtypes(include=["number"]).columns:
        if pd.api.types.is_integer_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], downcast="integer")
        
        elif pd.api.types.is_float_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], downcast="float")

    mem_after = data.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage after optimization is: {mem_after:.2f} MB")
        print(f"Decreased by {(100 * (mem_before - mem_after) / mem_before):.1f}%\n")

    return data


@st.cache_data
def load_data():
    data = pd.read_csv("playground-series-s5e10/train.csv", index_col="id")
    categorical_columns = ["road_type", "lighting", "weather", "time_of_day"]
    for cc in categorical_columns:
        data[cc] = data[cc].astype("category")
    data = downcasting(data, verbose=False)
    return data


@st.cache_resource
def load_model():
    model = XGBRegressor(enable_categorical=True)
    model.load_model("accident_risk_model.json")
    return model

def main():
    st.title("🚗 Accident Risk Prediction Game")
    st.write("Can you predict accident risk better than the AI model?")
    
    if 'game_started' not in st.session_state:
        st.session_state.game_started = False
    if 'user_score' not in st.session_state:
        st.session_state.user_score = 0
    if 'model_score' not in st.session_state:
        st.session_state.model_score = 0
    if 'current_item' not in st.session_state:
        st.session_state.current_item = None
    if 'revealed' not in st.session_state:
        st.session_state.revealed = False
    if 'user_prediction' not in st.session_state:
        st.session_state.user_prediction = 0.5
    
    data = load_data()
    model = load_model()
    explainer = shap.TreeExplainer(model)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Your Score", st.session_state.user_score)
    with col2:
        st.metric("AI Model Score", st.session_state.model_score)
    with col3:
        total_rounds = st.session_state.user_score + st.session_state.model_score
        st.metric("Total Rounds", total_rounds)
    
    if st.button("🎮 Start New Round" if st.session_state.game_started else "🎮 Start Game"):
        sample_idx = np.random.choice(data.index)
        st.session_state.current_item = data.loc[[sample_idx]]
        st.session_state.game_started = True
        st.session_state.revealed = False
        st.session_state.user_prediction = 0.5
        st.rerun()
    
    if st.session_state.game_started and st.session_state.current_item is not None:
        st.divider()
        
        st.subheader("📊 Current Scenario")
        
        col1, col2 = st.columns(2)
        
        current_item = st.session_state.current_item
        feature_data = current_item.drop('accident_risk', axis=1)
        actual_risk = current_item['accident_risk'].iloc[0]
        
        with col1:
            st.write("**Scenario Details:**")
            for feature, value in feature_data.iloc[0].items():
                st.write(f"• **{feature.replace('_', ' ').title()}:** {value}")
        
        with col2:
            st.write("**Your Prediction:**")
            user_pred = st.slider(
                "What's your accident risk prediction?",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.user_prediction,
                step=0.01,
                key="prediction_slider"
            )
            st.session_state.user_prediction = user_pred
        
        if not st.session_state.revealed:
            if st.button("🔍 Reveal Predictions", type="primary"):
                st.session_state.revealed = True
                st.rerun()
        
        if st.session_state.revealed:
            st.divider()
            st.subheader("📈 Results")
            
            model_pred = model.predict(feature_data)[0]
            explanation = explainer(feature_data)
            user_distance = abs(user_pred - actual_risk)
            model_distance = abs(model_pred - actual_risk)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Your Prediction", 
                    f"{user_pred:.3f}",
                    delta=f"Distance: {user_distance:.3f}"
                )
            
            with col2:
                st.metric(
                    "AI Model Prediction", 
                    f"{model_pred:.3f}",
                    delta=f"Distance: {model_distance:.3f}"
                )
            
            with col3:
                st.metric(
                    "Actual Risk", 
                    f"{actual_risk:.3f}",
                    delta=None
                )
            
            if user_distance < model_distance:
                st.success("🎉 You win this round!")
                st.session_state.user_score += 1
            elif model_distance < user_distance:
                st.error("🤖 AI Model wins this round!")
                st.session_state.model_score += 1
            else:
                st.info("🤝 It's a tie!")
            
            st.subheader("📊 Visual Comparison")
            comparison_df = pd.DataFrame({
                'Prediction Type': ['Your Prediction', 'AI Model', 'Actual Risk'],
                'Value': [user_pred, model_pred, actual_risk],
                'Distance from Actual': [user_distance, model_distance, 0]
            })
            
            st.bar_chart(comparison_df.set_index('Prediction Type')['Value'])

            # Add SHAP waterfall plot
            st.subheader("🔍 AI Model Explanation")
            if st.button("📊 Show SHAP Waterfall Plot"):
                with st.spinner("Generating SHAP explanation..."):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(explanation[0], show=False)
                    plt.tight_layout()

                    st.pyplot(fig)
                    plt.close()


if __name__ == "__main__":
    main()
