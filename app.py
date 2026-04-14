import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import json
import hashlib

# ==============================================================================
# UI CONFIGURATION & AUTH STATE
# ==============================================================================
st.set_page_config(
    page_title="AI Student Risk System - Dynamic Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authentication Session State Initialization
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None

# Ensure basic directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
USERS_DB = "data/users.json"
BASE_MODEL_PATH = "models/lstm_model.keras"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user_db():
    if os.path.exists(USERS_DB):
        with open(USERS_DB, "r") as f:
            return json.load(f)
    return {}

def save_user_db(db):
    with open(USERS_DB, "w") as f:
        json.dump(db, f)

# ==============================================================================
# AUTHENTICATION GATEWAY
# ==============================================================================
if not st.session_state["logged_in"]:
    st.title("Welcome to the AI Student Risk System 🎓")
    st.markdown("Please log in or create an account to securely access your personalized behavioral AI tracking.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("### Access Portal")
        auth_mode = st.radio("Select Action:", ["Login", "Create Account"], horizontal=True)
        
        email_input = st.text_input("Email Address")
        password_input = st.text_input("Password", type="password")
        
        if auth_mode == "Login":
            if st.button("Secure Login"):
                db = get_user_db()
                if email_input in db and db[email_input]["password_hash"] == hash_password(password_input):
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = email_input
                    st.success("Authentication successful! Loading dashboard...")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")
                    
        elif auth_mode == "Create Account":
            if st.button("Register Now", type="primary"):
                if email_input and password_input:
                    db = get_user_db()
                    if email_input in db:
                        st.warning("An account with this email already exists. Please log in.")
                    else:
                        # Simple registration bypassing external verification code explicitly as requested
                        db[email_input] = {"password_hash": hash_password(password_input)}
                        save_user_db(db)
                        st.success("Account successfully created! You can now log in.")
                else:
                    st.error("Please provide both an Email address and a Password.")
                    
    # Halt stream execution so the main app doesn't render behind the login screen
    st.stop()


# ==============================================================================
# DASHBOARD SETUP & PERSONALIZATION PATHS
# ==============================================================================
# Use a safely-escaped version of user email for personal file generation
user_safe_slug = st.session_state["current_user"].replace("@", "_at_").replace(".", "_dot_")
DATA_PATH = f"data/{user_safe_slug}_historical_data.csv"
UPDATED_MODEL_PATH = f"models/{user_safe_slug}_lstm_model_updated.keras"

feature_columns = [
    "Stress", "Anxiety", "Mood", "Emotional_Clarity", "Sleep_Hours", 
    "Energy", "Routine", "Procrastination", "Study_Hours", "Task_Completion"
]

def load_data():
    """Load individualized historical data from CSV; create if neither file nor session vars exist."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        empty_df = pd.DataFrame(columns=["Week_Index"] + feature_columns)
        empty_df.to_csv(DATA_PATH, index=False)
        return empty_df

# Since history is user isolated, we must force-reload it when user changes 
# But Streamlit maintains session state per browser tab, so we just check standard existence:
if "history_df" not in st.session_state:
    st.session_state.history_df = load_data()

def update_history_csv():
    st.session_state.history_df.to_csv(DATA_PATH, index=False)


# ==============================================================================
# MODEL MANAGEMENT
# ==============================================================================
# Keras 3 / TF >= 2.16 Hotfix bypass
original_input_init = tf.keras.layers.InputLayer.__init__
def patched_input_init(self, **kwargs):
    kwargs.pop('batch_shape', None)
    kwargs.pop('optional', None)
    original_input_init(self, **kwargs)
tf.keras.layers.InputLayer.__init__ = patched_input_init

def adapt_model_shape_for_5_weeks(original_model):
    try:
        input_shape = original_model.input_shape
        if input_shape[1] == 5 or input_shape[1] is None:
            return original_model
            
        print(f"Dynamically adapting model from {input_shape[1]} to 5 timesteps...")
        new_model = tf.keras.Sequential()
        for i, layer in enumerate(original_model.layers):
            config = layer.get_config()
            if 'batch_input_shape' in config:
                config['batch_input_shape'] = (None, 5, input_shape[2])
            new_layer = layer.__class__.from_config(config)
            new_model.add(new_layer)
            
        new_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        for old_layer, new_layer in zip(original_model.layers, new_model.layers):
             new_layer.set_weights(old_layer.get_weights())
        return new_model
    except Exception as e:
        print(f"Failed to rebuild auto-architecture: {e}")
        return original_model

# We cache models based specifically on the user's slug so user models don't cross in cache memory!
@st.cache_resource
def get_active_model(user_slug):
    """Loads the personalized updated model if it exists, else base model, adapting mapping cleanly."""
    target = f"models/{user_slug}_lstm_model_updated.keras"
    if not os.path.exists(target):
         # Fallback to absolute base un-personalized model
         target = BASE_MODEL_PATH
         
    if os.path.exists(target):
        try:
            raw_model = tf.keras.models.load_model(target, compile=False)
            adapted_model = adapt_model_shape_for_5_weeks(raw_model)
            adapted_model.predict(np.zeros((1, 5, 10)), verbose=0)
            return adapted_model
        except Exception as e:
            st.sidebar.error(f"Error loading model context ({target}): {e}")
            return None
    return None

model = get_active_model(user_safe_slug)


# ==============================================================================
# MAIN APPLICATION LOGIC
# ==============================================================================
col_title, col_logout = st.columns([8, 1])
with col_title:
    st.title("🎓 Continuous Learning Student Risk System")
with col_logout:
    if st.button("Log out"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.markdown(f"***Logged in securely as:** `{st.session_state['current_user']}`*")

tab_input, tab_history, tab_predict, tab_insights, tab_learning = st.tabs([
    "📥 Data Input", "📋 History Database", "🎯 5-Week Risk Prediction", "💡 Trends & Insights", "🧠 Continuous Learning"
])

with tab_input:
    st.header("Log This Week's Data")
    c1, c2 = st.columns(2)
    with c1:
        stress = st.slider("Stress Level", 1, 5, 3)
        anxiety = st.slider("Anxiety Level", 1, 5, 3)
        mood = st.slider("Mood", -2, 2, 0)
        clarity = st.slider("Emotional Clarity", 1, 5, 3)
        energy = st.slider("Energy Level", 1, 5, 3)
    with c2:
        sleep = st.slider("Sleep Hours", 0.0, 10.0, 7.0, 0.5)
        routine = st.slider("Routine Consistency", 1, 5, 3)
        procrast = st.slider("Procrastination", 1, 5, 3)
        study = st.slider("Study Hours", 0.0, 10.0, 4.0, 0.5)
        completion = st.slider("Task Completion Ratio", 0.0, 1.0, 0.5, 0.1)

    if st.button("💾 Add Weekly Data to Profile", type="primary"):
        next_index = len(st.session_state.history_df) + 1
        new_row = {"Week_Index": next_index, "Stress": stress, "Anxiety": anxiety, "Mood": mood, "Emotional_Clarity": clarity, "Sleep_Hours": sleep, "Energy": energy, "Routine": routine, "Procrastination": procrast, "Study_Hours": study, "Task_Completion": completion}
        st.session_state.history_df = pd.concat([st.session_state.history_df, pd.DataFrame([new_row])], ignore_index=True)
        update_history_csv()
        st.success(f"Week {next_index} saved to your personal history!")

with tab_history:
    st.header("Your Private Behavioral Database")
    if len(st.session_state.history_df) == 0:
         st.info("No data added yet.")
    else:
         st.dataframe(st.session_state.history_df, use_container_width=True)
         ext = st.button("🗑️ Clear Your Database")
         if ext:
             st.session_state.history_df = pd.DataFrame(columns=["Week_Index"] + feature_columns)
             update_history_csv()
             st.rerun()

def generate_5wk_sequence(df):
    data = np.asarray(df[feature_columns].values).astype(np.float32)
    if len(data) == 0: return np.zeros((1, 5, 10))
    if len(data) < 5:
        padding = np.tile(data[-1], (5 - len(data), 1))
        return np.expand_dims(np.vstack((padding, data)), axis=0)
    return np.expand_dims(data[-5:], axis=0)

with tab_predict:
    st.header("Real-Time AI Prediction")
    if len(st.session_state.history_df) == 0:
        st.error("Add data to calculate risks.")
    else:
        lstm_input = generate_5wk_sequence(st.session_state.history_df)
        try:
            if model is not None:
                preds = model.predict(lstm_input, verbose=0)[0]
            else:
                st.warning("Mock prediction mode.")
                latest = st.session_state.history_df.iloc[-1]
                preds = [np.clip(1.0-(latest["Study_Hours"]/10*0.5+latest["Task_Completion"]*0.5),0,1), np.clip((latest["Stress"]/5*0.4+latest["Anxiety"]/5*0.4+(10-latest["Sleep_Hours"])/10*0.2),0,1), np.clip((latest["Procrastination"]/5*0.5+(1.0-latest["Routine"]/5)*0.5),0,1)]
        except Exception as e:
            st.error("Prediction Error.")
            preds = [0,0,0]

        a_s, b_s, c_s = [float(p) for p in preds]
        
        def g_color(s): return "green" if s < 0.33 else "orange" if s < 0.66 else "red"
        def g_lbl(s): return "Low Risk" if s < 0.33 else "Moderate Risk" if s < 0.66 else "High Risk"
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"### Academic Risk\n<h2 style='color:{g_color(a_s)};'>{a_s*100:.1f}%</h2><p>{g_lbl(a_s)}</p>", unsafe_allow_html=True)
            st.progress(a_s)
        with c2:
            st.markdown(f"### Burnout Risk\n<h2 style='color:{g_color(b_s)};'>{b_s*100:.1f}%</h2><p>{g_lbl(b_s)}</p>", unsafe_allow_html=True)
            st.progress(b_s)
        with c3:
            st.markdown(f"### Career Risk\n<h2 style='color:{g_color(c_s)};'>{c_s*100:.1f}%</h2><p>{g_lbl(c_s)}</p>", unsafe_allow_html=True)
            st.progress(c_s)

with tab_insights:
    st.header("Personalized Behavioral Insights")
    if len(st.session_state.history_df) > 0:
        latest = st.session_state.history_df.iloc[-1].to_dict()
        means = st.session_state.history_df[feature_columns].mean()
        diffs = {f: latest[f] - means[f] for f in feature_columns}
        invs = ["Mood", "Emotional_Clarity", "Sleep_Hours", "Energy", "Routine", "Study_Hours", "Task_Completion"]
        contributions = sorted([(f, d * -1 if f in invs else d) for f, d in diffs.items()], key=lambda x: x[1], reverse=True)
        top = contributions[:5]
        
        if top[0][1] <= 0: st.success("Improving steadily across all metrics!")
        else:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.barh([r[0] for r in top][::-1], [r[1] for r in top][::-1], color='salmon')
            ax.set_title("Top Risk Multipliers (Compared to Average)")
            st.pyplot(fig)
            st.info(f"💡 Identify and focus on mitigating **{top[0][0]}** behaviors.")

with tab_learning:
    st.header("Incremental System Tuning")
    st.markdown("Supply Ground Truths to retrain your extremely personalized architecture.")
    if len(st.session_state.history_df) >= 1:
        t_a = st.slider("True Academic Risk (Outcome)", 0.0, 1.0, 0.5)
        t_b = st.slider("True Burnout Risk (Outcome)", 0.0, 1.0, 0.5)
        t_c = st.slider("True Career Risk (Outcome)", 0.0, 1.0, 0.5)
        if st.button("🚀 Execute Personalized Retraining"):
            if model is not None:
                with st.spinner("Retuning AI uniquely for your account..."):
                    try:
                        x = generate_5wk_sequence(st.session_state.history_df)
                        y = np.array([[t_a, t_b, t_c]], dtype=np.float32)
                        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mae'])
                        model.fit(x, y, epochs=5, verbose=0)
                        model.save(UPDATED_MODEL_PATH)
                        st.success(f"Retraining successful! Your weights are safely encoded inside {UPDATED_MODEL_PATH}.")
                    except Exception as e:
                        st.error(f"Training failed: {e}")
    else:
        st.warning("Needs context data to trigger learning.")
