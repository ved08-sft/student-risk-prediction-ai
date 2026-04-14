import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# ==============================================================================
# UI CONFIGURATION & STATE INIT
# ==============================================================================
st.set_page_config(
    page_title="AI Student Risk System - Dynamic Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Persistent Data Storage Setup (Session State + CSV Optionally)
DATA_PATH = "data/historical_data.csv"
MODEL_PATH = "models/lstm_model.h5"
UPDATED_MODEL_PATH = "models/lstm_model_updated.keras"

feature_columns = [
    "Stress", "Anxiety", "Mood", "Emotional_Clarity", "Sleep_Hours", 
    "Energy", "Routine", "Procrastination", "Study_Hours", "Task_Completion"
]

def load_data():
    """Load historical data from CSV if exists, otherwise initialize empty DataFrame."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        empty_df = pd.DataFrame(columns=["Week_Index"] + feature_columns)
        empty_df.to_csv(DATA_PATH, index=False)
        return empty_df

if "history_df" not in st.session_state:
    st.session_state.history_df = load_data()

def update_history_csv():
    st.session_state.history_df.to_csv(DATA_PATH, index=False)

# ==============================================================================
# MODEL MANAGEMENT (Loading & Architecture Transfer)
# ==============================================================================
@st.cache_resource
def adapt_model_shape_for_5_weeks(original_model):
    """
    Dynamically transfers weights from an existing (1, 3, 10) LSTM model 
    into a new (1, 5, 10) model to accommodate the 5-week sequence constraint.
    """
    try:
        input_shape = original_model.input_shape
        # If the model already supports 5 timesteps or arbitrary lengths, return it.
        if input_shape[1] == 5 or input_shape[1] is None:
            return original_model
            
        print(f"Dynamically adapting model from {input_shape[1]} to 5 timesteps...")
        
        # Build the new sequential model with the updated shape
        new_model = tf.keras.Sequential()
        for i, layer in enumerate(original_model.layers):
            config = layer.get_config()
            # If this is an input layer or the first layer with batch_input_shape defined
            if 'batch_input_shape' in config:
                config['batch_input_shape'] = (None, 5, input_shape[2])
            
            # Create a fresh instantiation of the same layer
            new_layer = layer.__class__.from_config(config)
            new_model.add(new_layer)
            
        # Compile model
        new_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Copy weights sequentially
        for old_layer, new_layer in zip(original_model.layers, new_model.layers):
             new_layer.set_weights(old_layer.get_weights())
             
        return new_model
    except Exception as e:
        print(f"Failed to rebuild auto-architecture: {e}")
        return original_model

def get_active_model():
    """Loads the updated model if it exists, else base model, and adapts it to (5, 10)."""
    target = UPDATED_MODEL_PATH if os.path.exists(UPDATED_MODEL_PATH) else MODEL_PATH
    if os.path.exists(target):
        try:
            # Setting compile=False bypasses the KerasSaveable bug for metrics like 'mse' on reload
            raw_model = tf.keras.models.load_model(target, compile=False)
            # Reconstruct architecture for 5 weeks if necessary
            adapted_model = adapt_model_shape_for_5_weeks(raw_model)
            # Dummy predict to initialize graph correctly
            adapted_model.predict(np.zeros((1, 5, 10)), verbose=0)
            return adapted_model
        except Exception as e:
            st.sidebar.error(f"Error loading {target}: {e}")
            return None
    return None

model = get_active_model()

# ==============================================================================
# UI COMPONENTS
# ==============================================================================
st.title("🎓 Continuous Learning Student Risk System")
st.markdown("Track behaviors over time, predict academic & emotional risks, and continuously retrain the AI based on ground truths.")

# Layout Tabs
tab_input, tab_history, tab_predict, tab_insights, tab_learning = st.tabs([
    "📥 Data Input", "📋 History Database", "🎯 5-Week Risk Prediction", "💡 Trends & Insights", "🧠 Continuous Learning"
])

# ----------------- 1. INPUT SECTION -----------------
with tab_input:
    st.header("Log This Week's Data")
    col1, col2 = st.columns(2)
    with col1:
        stress = st.slider("Stress Level", 1, 5, 3, help="1=Low, 5=High")
        anxiety = st.slider("Anxiety Level", 1, 5, 3)
        mood = st.slider("Mood", -2, 2, 0, help="-2=Negative, 2=Positive")
        clarity = st.slider("Emotional Clarity", 1, 5, 3)
        energy = st.slider("Energy Level", 1, 5, 3)
    with col2:
        sleep = st.slider("Sleep Hours", 0.0, 10.0, 7.0, 0.5)
        routine = st.slider("Routine Consistency", 1, 5, 3)
        procrast = st.slider("Procrastination", 1, 5, 3)
        study = st.slider("Study Hours", 0.0, 10.0, 4.0, 0.5)
        completion = st.slider("Task Completion Ratio", 0.0, 1.0, 0.5, 0.1)

    if st.button("💾 Add Weekly Data to System", type="primary"):
        # Append to dataframe and session state
        next_index = len(st.session_state.history_df) + 1
        new_row = {
            "Week_Index": next_index,
            "Stress": stress, "Anxiety": anxiety, "Mood": mood, "Emotional_Clarity": clarity,
            "Sleep_Hours": sleep, "Energy": energy, "Routine": routine, 
            "Procrastination": procrast, "Study_Hours": study, "Task_Completion": completion
        }
        st.session_state.history_df = pd.concat([st.session_state.history_df, pd.DataFrame([new_row])], ignore_index=True)
        update_history_csv()
        st.success(f"Week {next_index} successfully added! Moving to predictions...")

# ----------------- 2. DATA HISTORY -----------------
with tab_history:
    st.header("Stored Weekly Behavior Database")
    if len(st.session_state.history_df) == 0:
        st.info("No data added yet. Use the Input tab to log your first week.")
    else:
        st.dataframe(st.session_state.history_df, use_container_width=True)
        st.metric(label="Total Weeks Logged", value=len(st.session_state.history_df))
        
        if st.button("🗑️ Clear Database (Danger)"):
            st.session_state.history_df = pd.DataFrame(columns=["Week_Index"] + feature_columns)
            update_history_csv()
            st.rerun()

# ----------------- 3. PREDICTION (5-Week Sliding Window) -----------------
def generate_5wk_sequence(df):
    """Generates the (1, 5, 10) sequence, padding with the last known data if insufficient."""
    # Convert object arrays back to float32 natively prior to model ingest to prevent TF type errors
    data_values = np.asarray(df[feature_columns].values).astype(np.float32)
    current_len = len(data_values)
    
    if current_len == 0:
        return np.zeros((1, 5, 10))
    elif current_len < 5:
        # Pad with the most recent entry up to 5 weeks
        pad_length = 5 - current_len
        st.warning(f"⚠️ Only {current_len}/5 weeks available. Simulating missing history using current habits to fill the model window.", icon="ℹ️")
        padding = np.tile(data_values[-1], (pad_length, 1))
        sequence = np.vstack((padding, data_values))
        return np.expand_dims(sequence, axis=0) # (1, 5, 10)
    else:
        # Exactly or more than 5 weeks; take the last 5
        sequence = data_values[-5:]
        return np.expand_dims(sequence, axis=0)

def determine_risk(score):
    if score < 0.33: return "green", "Low Risk"
    elif score < 0.66: return "orange", "Moderate Risk"
    else: return "red", "High Risk"

with tab_predict:
    st.header("Real-Time AI Prediction")
    if len(st.session_state.history_df) == 0:
        st.error("Please add data via the Input tab to calculate risks.")
    else:
        # Prepare Data
        lstm_input = generate_5wk_sequence(st.session_state.history_df)
        
        # Predict
        try:
            if model is not None:
                preds = model.predict(lstm_input, verbose=0)[0]
            else:
                st.warning("Running mock prediction mode (lstm_model.h5 not found via active weights).")
                # Mock based on latest entry
                latest = st.session_state.history_df.iloc[-1]
                mock_academic = np.clip(1.0 - (latest["Study_Hours"]/10 * 0.5 + latest["Task_Completion"] * 0.5), 0, 1)
                mock_burnout = np.clip((latest["Stress"]/5 * 0.4 + latest["Anxiety"]/5 * 0.4 + (10 - latest["Sleep_Hours"])/10 * 0.2), 0, 1)
                mock_career = np.clip((latest["Procrastination"]/5 * 0.5 + (1.0 - latest["Routine"]/5) * 0.5), 0, 1)
                preds = [mock_academic, mock_burnout, mock_career]
        except Exception as pred_err:
            st.error("Prediction encountered an internal error. Returning safe zeroes to prevent UI crash.")
            st.write(f"*(Technical context: {pred_err})*")
            preds = [0.0, 0.0, 0.0]

        a_score, b_score, c_score = float(preds[0]), float(preds[1]), float(preds[2])
        
        # Save exact predictions inside session state for smooth insights later
        if "pred_history" not in st.session_state:
            st.session_state.pred_history = []
        
        # We only append history if the UI was just regenerated fully, simplified list tracker:
        if len(st.session_state.pred_history) < len(st.session_state.history_df):
            st.session_state.pred_history.append((a_score, b_score, c_score))

        # Dashboard Visuals
        st.markdown(f"**Analyzing Sequence Array Shape:** `{lstm_input.shape}`")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            color, label = determine_risk(a_score)
            st.markdown(f"### Academic Risk\n<h2 style='color: {color};'>{a_score*100:.1f}%</h2><p>{label}</p>", unsafe_allow_html=True)
            st.progress(a_score)
        with col2:
            color, label = determine_risk(b_score)
            st.markdown(f"### Burnout Risk\n<h2 style='color: {color};'>{b_score*100:.1f}%</h2><p>{label}</p>", unsafe_allow_html=True)
            st.progress(b_score)
        with col3:
            color, label = determine_risk(c_score)
            st.markdown(f"### Career Risk\n<h2 style='color: {color};'>{c_score*100:.1f}%</h2><p>{label}</p>", unsafe_allow_html=True)
            st.progress(c_score)

# ----------------- 4. INSIGHTS & EXPLAINABILITY -----------------
with tab_insights:
    st.header("Behavioral AI Insights")
    if len(st.session_state.history_df) > 0:
        latest = st.session_state.history_df.iloc[-1].to_dict()
        
        # Simplistic Explanation logic: What changed vs the historical average?
        history_mean = st.session_state.history_df[feature_columns].mean()
        diffs = {f: latest[f] - history_mean[f] for f in feature_columns}
        
        st.write("### 🚨 What spiked your risk recently?")
        worst_growth = sorted(diffs.items(), key=lambda x: x[1], reverse=True) # positive diff = grew higher
        
        # We know Stress, Anxiety, Procrast growing is BAD. 
        # But Sleep, Energy, Clarity, Routine, Study growing is GOOD.
        # So lets invert the "good" factors mathematically for the chart.
        inverted_good_factors = ["Mood", "Emotional_Clarity", "Sleep_Hours", "Energy", "Routine", "Study_Hours", "Task_Completion"]
        
        risk_contributions = []
        for feature, diff in diffs.items():
            impact = diff * -1 if feature in inverted_good_factors else diff
            risk_contributions.append((feature, impact))
            
        risk_contributions = sorted(risk_contributions, key=lambda x: x[1], reverse=True)
        
        # Plot Top 5 factors driving Risk UP
        top_risks = risk_contributions[:5]
        
        if top_risks[0][1] <= 0:
             st.success("You are steadily improving across all metrics! No negative drivers right now.")
        else:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.barh([r[0] for r in top_risks][::-1], [r[1] for r in top_risks][::-1], color='salmon')
            ax.set_title("Top Emerging Risk Factors (Compared to your Average)")
            ax.set_xlabel("Relative Worsening Impact")
            st.pyplot(fig)
            st.info(f"💡 It looks like **{top_risks[0][0]}** and **{top_risks[1][0]}** are your primary struggles right now compared to your historical baseline.")
    else:
         st.write("Insufficient data for insights.")

# ----------------- 5. CONTINUOUS LEARNING -----------------
with tab_learning:
    st.header("Incremental System Tuning (Continuous Learning)")
    st.markdown("""
    Here, you supply the True Ground labels for your previous week. 
    The AI uses your actual status to backpropagate and correct its assumptions. 
    This automatically builds a uniquely personalized model tailored just for you.
    """)
    
    if len(st.session_state.history_df) >= 1:
        st.write("### Retrain with Latest Reality")
        st.write("How did you actually feel/perform over the last 5 weeks?")
        
        true_academic = st.slider("True Academic Risk (Actual Outcome)", 0.0, 1.0, 0.5, 0.05)
        true_burnout = st.slider("True Burnout Risk (Actual Outcome)", 0.0, 1.0, 0.5, 0.05)
        true_career = st.slider("True Career Risk (Actual Outcome)", 0.0, 1.0, 0.5, 0.05)
        
        if st.button("🚀 Execute Incremental Retraining"):
            if model is not None:
                 with st.spinner("Compiling sliding window and executing backward pass..."):
                     try:
                         # Get 5wk window
                         x_train = generate_5wk_sequence(st.session_state.history_df)
                         # True labels
                         y_train = np.array([[true_academic, true_burnout, true_career]], dtype=np.float32)
                         
                         # Force re-compile with a fresh optimizer instance to prevent Keras stale variable scoping errors
                         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
                         
                         # Fit model on the current sequence
                         history = model.fit(x_train, y_train, epochs=5, verbose=0)
                         
                         # Save specialized instance
                         model.save(UPDATED_MODEL_PATH)
                         st.success(f"Model retrained incrementally! Loss lowered to: {history.history['loss'][-1]:.4f}")
                         st.info("The system automatically cached the adapted weights and will use them for all future predictions.")
                     except Exception as train_err:
                         st.error(f"Failed to execute training step: {train_err}")
            else:
                 st.error("No core Keras model running to retrain.")
    else:
        st.warning("Need at least 1 week of data to trigger incremental learning updates.")
