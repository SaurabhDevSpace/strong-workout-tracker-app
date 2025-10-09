import streamlit as st
import pandas as pd
import json
import uuid
import datetime
import pytz
import logging
import os
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, TypedDict

# Google Drive and LLM imports
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IST = pytz.timezone('Asia/Kolkata')
WORKOUT_TYPES = {
    "🫁 Chest": "chest",
    "🪽 Back": "back",
    "🏋🏻 Shoulder": "shoulder",
    "💪 Biceps": "biceps",
    "💪🏿 Triceps": "triceps",
    "🦵 Legs": "legs",
    "🆎 Abs": "abs"
}

# File paths
CONFIG_FILE = 'config.json'
CREDENTIALS_FILE = 'credentials.json'
WORKOUT_DATA_FILE = 'workout_data.csv'

#-----------------------------------------
# Configuration and Data Loading
#-----------------------------------------

def load_config():
    """Load configuration from file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"upload_to_drive_enabled": False}


# No longer needed: load_credentials(). Use st.secrets instead.


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'workout_data' not in st.session_state:
        st.session_state.workout_data = pd.DataFrame()

    if 'data_updated' not in st.session_state:
        st.session_state.data_updated = False

    if 'initial_download_done' not in st.session_state:
        st.session_state.initial_download_done = False

    # Load credentials and config from st.secrets
    st.session_state.api_key = st.secrets.get('GEMINI_API_KEY', '')
    st.session_state.llm_model = st.secrets.get('LLM_MODEL', 'gemini-2.0-flash-lite')
    # For upload_to_drive_enabled, support both bool and string
    upload_to_drive_enabled = st.secrets.get('UPLOAD_TO_DRIVE_ENABLED', False)
    if isinstance(upload_to_drive_enabled, str):
        upload_to_drive_enabled = upload_to_drive_enabled.lower() == 'true'
    st.session_state.upload_to_drive_enabled = upload_to_drive_enabled


#-----------------------------------------
# Google Drive Integration
#-----------------------------------------

def get_drive_service():
    """Initialize Google Drive service."""
    try:
        credentials_dict = json.loads(st.secrets["GOOGLE_DRIVE_CREDENTIALS"])
        credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        logger.error(f"Failed to initialize Google Drive service: {str(e)}")
        return None

def upload_to_drive():
    """Upload workout data to Google Drive."""
    if not st.session_state.upload_to_drive_enabled:
        return

    try:
        service = get_drive_service()
        if not service:
            return

        # Convert DataFrame to CSV
        csv_buffer = BytesIO()
        st.session_state.workout_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        media = MediaIoBaseUpload(csv_buffer, mimetype='text/csv')

        # Search for file by name and mimeType to avoid duplicates
        query = f"name='{WORKOUT_DATA_FILE}' and mimeType='text/csv' and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = results.get('files', [])

        if files:
            # Update existing file (use files().update with correct media_body)
            file_id = files[0]['id']
            service.files().update(fileId=file_id, media_body=media).execute()
            logger.info("Workout data updated on Google Drive")
        else:
            # Create new file
            file_metadata = {'name': WORKOUT_DATA_FILE, 'mimeType': 'text/csv'}
            service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            logger.info("Workout data uploaded to Google Drive")

    except Exception as e:
        logger.error(f"Error uploading to Google Drive: {str(e)}")

def download_from_drive():
    """Download workout data from Google Drive."""
    try:
        service = get_drive_service()
        if not service:
            return

        # Search for file by name and mimeType
        query = f"name='{WORKOUT_DATA_FILE}' and mimeType='text/csv' and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = results.get('files', [])

        if files:
            file_id = files[0]['id']
            request = service.files().get_media(fileId=file_id)

            csv_buffer = BytesIO()
            downloader = MediaIoBaseDownload(csv_buffer, request)

            done = False
            while not done:
                _, done = downloader.next_chunk()

            csv_buffer.seek(0)
            try:
                df = pd.read_csv(csv_buffer)
                st.session_state.workout_data = df
                logger.info("Workout data downloaded from Google Drive")
            except Exception as e:
                logger.error(f"Error reading CSV from Google Drive: {str(e)}")
        else:
            logger.info("No workout data file found on Google Drive")

    except Exception as e:
        logger.error(f"Error downloading from Google Drive: {str(e)}")

#-----------------------------------------
# Data Management
#-----------------------------------------

def load_local_data():
    """Load workout data from local CSV file."""
    try:
        if os.path.exists(WORKOUT_DATA_FILE):
            st.session_state.workout_data = pd.read_csv(WORKOUT_DATA_FILE)
        else:
            # Initialize with empty DataFrame with proper columns
            st.session_state.workout_data = pd.DataFrame(columns=[
                'date', 'workout_types', 'gym_time_minutes', 'cardio_time_minutes',
                'cardio_calories', 'cardio_distance', 'notes', 'id'
            ])
    except Exception as e:
        logger.error(f"Error loading local data: {str(e)}")
        st.session_state.workout_data = pd.DataFrame(columns=[
            'date', 'workout_types', 'gym_time_minutes', 'cardio_time_minutes',
            'cardio_calories', 'cardio_distance', 'notes', 'id'
        ])

def save_local_data():
    """Save workout data to local CSV file."""
    try:
        st.session_state.workout_data.to_csv(WORKOUT_DATA_FILE, index=False)
        logger.info("Workout data saved locally")
    except Exception as e:
        logger.error(f"Error saving local data: {str(e)}")

def load_data():
    """Load workout data from local file and Google Drive."""
    load_local_data()

    if not st.session_state.initial_download_done:
        logger.info("Performing initial download from Google Drive")
        download_from_drive()
        st.session_state.initial_download_done = True

#-----------------------------------------
# LLM Integration for Image Processing
#-----------------------------------------

class AgentState(TypedDict):
    user_input: str
    image_data: Optional[str]
    extracted_data: dict

def extract_cardio_data_from_image(image_data: str) -> dict:
    """Extract cardio data from image using LLM."""
    try:
        prompt = """You are a fitness data extraction assistant. Analyze the cardio machine display image and extract the following information:

1. Time spent (in minutes)
2. Calories burned
3. Distance covered (with unit)

Look for these values on treadmill, elliptical, or other cardio machine displays.

Return the data in this JSON format:
{
  "time_minutes": <number>,
  "calories": <number>, 
  "distance": "<number with unit>",
  "success": true
}

If you cannot extract the data clearly, return:
{
  "success": false,
  "error": "Could not extract cardio data from image"
}

Image to analyze:"""

        model = ChatGoogleGenerativeAI(
            model=st.session_state.llm_model,
            google_api_key=st.session_state.api_key,
            temperature=0.1
        )

        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ])

        response = model.invoke([message])

        # Parse JSON response
        response_text = response.content.strip()
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0].strip()

        return json.loads(response_text)

    except Exception as e:
        logger.error(f"Error extracting cardio data: {str(e)}")
        return {"success": False, "error": str(e)}

#-----------------------------------------
# UI Components
#-----------------------------------------

def header_section():
    """Display app header."""
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🏋🏻‍♂️ StrongSB - The Workout Tracker</h1>
        <p style='font-size: 18px; color: #666;'>Tracks Gym workouts and Cardio sessions</p>
    </div>
    """, unsafe_allow_html=True)

def workout_logging_section():
    """Section for logging daily workouts."""
    st.header("📝 Log Today's Workout")

    # Check if user is admin
    if not st.session_state.is_admin:
        st.warning("🔒 Admin access required to log workouts")
        st.info("Please login as admin in the sidebar to add new workout entries.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        # Date selection
        workout_date = st.date_input(
            "📅 Workout Date",
            value=datetime.date.today(),
            key="workout_date"
        )

        # Workout type selection with emojis
        st.subheader("🏋️ Select Workout Types")
        selected_workouts = []

        cols = st.columns(4)
        for i, (display_name, workout_type) in enumerate(WORKOUT_TYPES.items()):
            with cols[i % 4]:
                if st.checkbox(display_name, key=f"workout_{workout_type}"):
                    selected_workouts.append(workout_type)

        # Time inputs
        col_time1, col_time2 = st.columns(2)
        with col_time1:
            gym_time = st.number_input("⏱️ Gym Time (minutes)", min_value=0, value=60, step=5)

        with col_time2:
            cardio_time = st.number_input("🏃 Cardio Time (minutes)", min_value=0, value=20, step=5)

        # Notes
        notes = st.text_area("📝 Notes", placeholder="Any additional notes about your workout...")

    with col2:
        st.subheader("📸 Upload Cardio Image")
        cardio_image = st.file_uploader(
            "Upload cardio machine display",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of your cardio machine display to automatically extract stats"
        )

        cardio_data = {"time_minutes": 0, "calories": 0, "distance": "0"}

        if cardio_image:
            # Display uploaded image
            image = Image.open(cardio_image)
            st.image(image, caption="Cardio Display", use_column_width=True)

            if st.button("🔍 Extract Data from Image"):
                with st.spinner("Analyzing image..."):
                    # Convert image to base64
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    # Extract data using LLM
                    extracted = extract_cardio_data_from_image(img_str)

                    if extracted.get("success"):
                        cardio_data = {
                            "time_minutes": extracted.get("time_minutes", 0),
                            "calories": extracted.get("calories", 0),
                            "distance": extracted.get("distance", "0")
                        }
                        st.success("✅ Data extracted successfully!")
                        st.json(cardio_data)
                    else:
                        st.error(f"❌ {extracted.get('error', 'Failed to extract data')}")

        # Manual cardio data entry
        st.subheader("📊 Manual Cardio Data")
        manual_calories = st.number_input("🔥 Calories Burned", min_value=0, value=cardio_data["calories"])
        manual_distance = st.text_input("📏 Distance", value=cardio_data["distance"], placeholder="e.g., 5.2 km")

        # Use extracted or manual data
        final_cardio_calories = cardio_data["calories"] if cardio_data["calories"] > 0 else manual_calories
        final_cardio_distance = cardio_data["distance"] if cardio_data["distance"] != "0" else manual_distance

    # Save workout button
    if st.button("💾 Log Workout", type="primary"):
        if selected_workouts:
            # Create new workout entry
            new_workout = {
                'date': workout_date.strftime('%Y-%m-%d'),
                'workout_types': ','.join(selected_workouts),
                'gym_time_minutes': gym_time,
                'cardio_time_minutes': cardio_time,
                'cardio_calories': final_cardio_calories,
                'cardio_distance': final_cardio_distance,
                'notes': notes,
                'id': str(uuid.uuid4())
            }

            # Add to DataFrame
            new_row = pd.DataFrame([new_workout])
            st.session_state.workout_data = pd.concat([st.session_state.workout_data, new_row], ignore_index=True)

            # Save data
            save_local_data()
            upload_to_drive()

            st.success("🎉 Workout logged successfully!")
            st.session_state.data_updated = True
        else:
            st.error("❌ Please select at least one workout type!")

def workout_history_section():
    """Display workout history and statistics."""
    st.header("📊 Workout History & Stats")

    if st.session_state.workout_data.empty:
        st.info("No workout data available. Start logging your workouts!")
        return

    # Recent workouts
    st.subheader("🕐 Recent Workouts")
    recent_workouts = st.session_state.workout_data.sort_values('date', ascending=False).head(10)

    for idx, workout in recent_workouts.iterrows():
        with st.expander(f"📅 {workout['date']} - {workout['workout_types'].replace(',', ', ').title()}"):
            # Show metrics in columns
            col1, col2, col3 = st.columns([2, 2, 2])
            if st.session_state.is_admin:
                # Admin gets 4 columns (including delete button)
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

            with col1:
                st.metric("🏋️ Gym Time", f"{workout['gym_time_minutes']} min")

            with col2:
                st.metric("🏃 Cardio Time", f"{workout['cardio_time_minutes']} min")

            with col3:
                st.metric("🔥 Calories", f"{workout['cardio_calories']}")

            # Only show delete button for admins
            if st.session_state.is_admin:
                with col4:
                    if st.button("🗑️ Delete", key=f"delete_{workout['id']}",
                               help="Delete this workout entry"):
                        delete_workout_entry(workout['id'])

            if workout['cardio_distance'] and workout['cardio_distance'] != "0":
                st.info(f"📏 Distance: {workout['cardio_distance']}")

            if workout['notes']:
                st.write(f"📝 Notes: {workout['notes']}")

def delete_workout_entry(workout_id):
    """Delete a specific workout entry by ID."""
    try:
        # Remove the workout entry by ID
        st.session_state.workout_data = st.session_state.workout_data[
            st.session_state.workout_data['id'] != workout_id
        ]

        # Reset the DataFrame index to avoid gaps
        st.session_state.workout_data = st.session_state.workout_data.reset_index(drop=True)

        # Save to local CSV
        save_local_data()

        # Upload to Google Drive
        upload_to_drive()

        st.success("🗑️ Workout deleted successfully!")
        st.session_state.data_updated = True

        # Force rerun to refresh the UI
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error deleting workout: {str(e)}")
        logger.error(f"Error deleting workout {workout_id}: {str(e)}")


def statistics_section():
    """Display workout statistics and analytics."""
    st.header("📈 Workout Analytics")

    if st.session_state.workout_data.empty:
        st.info("No data available for analytics.")
        return

    df = st.session_state.workout_data

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_workouts = len(df)
        st.metric("🏋️ Total Workouts", total_workouts)

    with col2:
        total_gym_time = df['gym_time_minutes'].sum()
        st.metric("⏱️ Total Gym Time", f"{total_gym_time} min")

    with col3:
        total_cardio_time = df['cardio_time_minutes'].sum()
        st.metric("🏃 Total Cardio Time", f"{total_cardio_time} min")

    with col4:
        total_calories = df['cardio_calories'].sum()
        st.metric("🔥 Total Calories", f"{total_calories}")

    # Workout type frequency
    st.subheader("💪 Workout Type Frequency")

    # Parse workout types
    all_workouts = []
    for workout_types in df['workout_types']:
        all_workouts.extend(workout_types.split(','))

    workout_counts = pd.Series(all_workouts).value_counts()

    if not workout_counts.empty:
        # Create a more readable format
        workout_display = {}
        for workout_type, count in workout_counts.items():
            emoji_name = next((k for k, v in WORKOUT_TYPES.items() if v == workout_type), workout_type.title())
            workout_display[emoji_name] = count

        st.bar_chart(pd.Series(workout_display))

    # Monthly trends
    st.subheader("📅 Monthly Trends")

    if len(df) > 1:
        # Ensure date column is properly converted to datetime
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')

        # Remove any rows with invalid dates
        df_copy = df_copy.dropna(subset=['date'])

        if len(df_copy) > 0:
            # Create month-year labels for better readability
            df_copy['month_year'] = df_copy['date'].dt.strftime('%Y-%m')

            monthly_stats = df_copy.groupby('month_year').agg({
                'gym_time_minutes': 'sum',
                'cardio_time_minutes': 'sum',
                'cardio_calories': 'sum'
            }).reset_index()

            # Set month_year as index for proper chart display
            monthly_stats.set_index('month_year', inplace=True)

            # Sort by date to ensure proper chronological order
            monthly_stats = monthly_stats.sort_index()

            st.line_chart(monthly_stats)
        else:
            st.info("No valid date data available for monthly trends.")


def footer_section():
    """Display app footer."""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <p>💪 Strong App | Keep pushing your limits! 🚀</p>
    </div>
    """, unsafe_allow_html=True)


def admin_authentication():
    """Simplified admin authentication with single admin level"""
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False

    if not st.session_state.is_admin:
        with st.sidebar:
            st.subheader("🔐 Admin Access")
            admin_password = st.text_input("Admin Password", type="password", key="admin_pass")
            if st.button("Login as Admin"):
                if admin_password == os.getenv("ADMIN_PASSWORD", "admin123"):
                    st.session_state.is_admin = True
                    st.success("Admin access granted!")
                    st.rerun()
                else:
                    st.error("Invalid admin password")
    else:
        with st.sidebar:
            st.success("✅ Logged in as Admin")
            if st.button("Logout"):
                st.session_state.is_admin = False
                st.rerun()

def admin_panel_section():
    """Admin-only panel for data management and LLM insights"""
    if not st.session_state.is_admin:
        st.warning("🔒 Admin access required")
        return

    st.header("🛠️ Admin Panel", divider="rainbow")

    tab1, tab2, tab3 = st.tabs(["Workout Management", "AI Insights", "Data Management"])

    with tab1:
        st.subheader("Add New Workout", divider=True)

        col1, col2 = st.columns(2)
        with col1:
            workout_date = st.date_input("Workout Date", value=datetime.date.today())
            workout_type = st.selectbox("Workout Type",
                ["Push", "Pull", "Legs", "Upper", "Lower", "Full Body", "Cardio", "Other"])

        with col2:
            duration = st.number_input("Duration (minutes)", min_value=1, value=60)
            notes = st.text_area("Workout Notes", placeholder="How did the workout feel?")

        # Exercise input
        st.subheader("Exercises", divider=True)
        if 'exercise_inputs' not in st.session_state:
            st.session_state.exercise_inputs = [{"exercise": "", "sets": 1, "reps": "", "weight": ""}]

        for i, exercise in enumerate(st.session_state.exercise_inputs):
            col1, col2, col3, col4, col5 = st.columns([3, 1, 2, 2, 1])
            with col1:
                exercise["exercise"] = st.text_input(f"Exercise {i+1}", value=exercise["exercise"], key=f"ex_{i}")
            with col2:
                exercise["sets"] = st.number_input("Sets", min_value=1, value=exercise["sets"], key=f"sets_{i}")
            with col3:
                exercise["reps"] = st.text_input("Reps", value=exercise["reps"], key=f"reps_{i}")
            with col4:
                exercise["weight"] = st.text_input("Weight", value=exercise["weight"], key=f"weight_{i}")
            with col5:
                if st.button("❌", key=f"del_{i}") and len(st.session_state.exercise_inputs) > 1:
                    st.session_state.exercise_inputs.pop(i)
                    st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add Exercise"):
                st.session_state.exercise_inputs.append({"exercise": "", "sets": 1, "reps": "", "weight": ""})
                st.rerun()

        with col2:
            if st.button("💾 Save Workout", type="primary"):
                save_workout(workout_date, workout_type, duration, notes, st.session_state.exercise_inputs)
                st.success("Workout saved successfully!")
                st.session_state.exercise_inputs = [{"exercise": "", "sets": 1, "reps": "", "weight": ""}]
                st.rerun()

    with tab2:
        generate_ai_insights()

    with tab3:
        data_management_section()

def generate_ai_insights():
    """Generate AI-powered fitness insights and recommendations"""
    st.subheader("🤖 AI Fitness Coach", divider=True)

    if not st.session_state.workouts:
        st.info("Add some workouts first to get AI insights!")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Get personalized insights about your fitness progress and recommendations for pushing your limits!")

    with col2:
        if st.button("🧠 Generate AI Insights", type="primary"):
            with st.spinner("Analyzing your fitness data..."):
                insights = get_fitness_insights()
                if insights:
                    st.session_state.ai_insights = insights
                    st.rerun()

    # Display cached insights
    if hasattr(st.session_state, 'ai_insights') and st.session_state.ai_insights:
        insights = st.session_state.ai_insights

        # Progress Analysis
        st.subheader("📊 Progress Analysis", divider=True)
        st.write(insights.get("progress_analysis", "No progress analysis available."))

        # Strengths & Weaknesses
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💪 Strengths")
            strengths = insights.get("strengths", [])
            for strength in strengths:
                st.success(f"✅ {strength}")

        with col2:
            st.subheader("🎯 Areas for Improvement")
            improvements = insights.get("improvements", [])
            for improvement in improvements:
                st.warning(f"⚠️ {improvement}")

        # Recommendations
        st.subheader("🚀 Push Your Limits - Recommendations", divider=True)
        recommendations = insights.get("recommendations", [])
        for i, rec in enumerate(recommendations, 1):
            st.info(f"**{i}.** {rec}")

        # Weekly Goals
        st.subheader("🎯 This Week's Goals", divider=True)
        goals = insights.get("weekly_goals", [])
        for goal in goals:
            st.checkbox(goal, key=f"goal_{hash(goal)}")

def get_fitness_insights():
    """Generate comprehensive fitness insights using LLM"""
    try:
        if not st.session_state.api_key:
            st.error("Please set your Google AI API key in the sidebar")
            return None

        # Prepare workout data for analysis
        recent_workouts = st.session_state.workouts[-20:]  # Last 20 workouts

        workout_summary = []
        for workout in recent_workouts:
            summary = {
                "date": workout["date"],
                "type": workout["type"],
                "duration": workout["duration"],
                "exercise_count": len(workout["exercises"]),
                "exercises": [f"{ex['exercise']}: {ex['sets']}x{ex['reps']} @ {ex['weight']}"
                            for ex in workout["exercises"] if ex["exercise"]]
            }
            workout_summary.append(summary)

        # Calculate basic stats
        total_workouts = len(st.session_state.workouts)
        avg_duration = sum(w["duration"] for w in st.session_state.workouts) / len(st.session_state.workouts)
        workout_types = {}
        for w in st.session_state.workouts:
            workout_types[w["type"]] = workout_types.get(w["type"], 0) + 1

        prompt = f"""You are an expert AI fitness coach analyzing workout data. Provide comprehensive insights and motivational recommendations.

**Workout Data Analysis:**
- Total Workouts: {total_workouts}
- Average Duration: {avg_duration:.1f} minutes
- Workout Type Distribution: {workout_types}

**Recent Workouts (Last 20):**
{json.dumps(workout_summary, indent=2)}

**Your Task:**
Analyze this fitness data and provide a comprehensive assessment with the following structure:

1. **Progress Analysis**: 2-3 sentences about overall progress, consistency, and trends
2. **Strengths**: List 3-4 specific strengths (e.g., consistency, exercise variety, progressive overload)
3. **Improvements**: List 2-3 areas that need attention (e.g., workout frequency, muscle group balance)
4. **Recommendations**: 4-5 specific, actionable recommendations to push limits and improve performance
5. **Weekly Goals**: 3-4 specific, measurable goals for the upcoming week

**Guidelines:**
- Be motivational and encouraging
- Focus on progressive overload and challenging limits
- Provide specific, actionable advice
- Consider workout balance and recovery
- Keep each point concise but detailed enough to be actionable

**Output Format:**
Return a JSON object with keys: progress_analysis, strengths, improvements, recommendations, weekly_goals
Each should be a string (for progress_analysis) or list of strings (for others).

**Response:**"""

        model = ChatGoogleGenerativeAI(
            model=st.session_state.llm_model,
            google_api_key=st.session_state.api_key,
            temperature=0.3
        )

        message = HumanMessage(content=prompt)
        response = model.invoke([message])

        # Parse response
        response_content = response.content.strip()
        if response_content.startswith("```json"):
            response_content = response_content.split("```json")[1].split("```")[0].strip()

        try:
            insights = json.loads(response_content)
            logger.info("Generated AI fitness insights successfully")
            return insights
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for fitness insights: {str(e)}")
            st.error("Failed to parse AI insights. Please try again.")
            return None

    except Exception as e:
        logger.error(f"Error generating fitness insights: {str(e)}")
        st.error(f"Error generating insights: {str(e)}")
        return None

def data_management_section():
    """Data export/import and settings management"""
    st.subheader("💾 Data Management", divider=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Export Data")
        if st.button("📥 Download Workout Data"):
            workout_df = pd.DataFrame(st.session_state.workouts)
            csv = workout_df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"strong_workouts_{datetime.date.today()}.csv",
                mime="text/csv"
            )

    with col2:
        st.subheader("Settings")

        # Google Drive toggle
        drive_enabled = st.checkbox(
            "Enable Google Drive Backup",
            value=st.session_state.config.get("upload_to_drive_enabled", False),
            help="Automatically backup data to Google Drive"
        )

        if st.button("💾 Save Settings"):
            st.session_state.config["upload_to_drive_enabled"] = drive_enabled
            save_config(st.session_state.config)
            st.success("Settings saved!")

    # Danger zone
    st.subheader("⚠️ Danger Zone", divider=True)
    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.checkbox("I understand this will delete all workout data"):
            if st.button("Confirm Delete", type="primary"):
                st.session_state.workouts = []
                save_workouts()
                st.success("All data cleared!")
                st.rerun()

def save_workout(date, workout_type, duration, notes, exercises):
    """Save a new workout to the data"""
    workout = {
        "id": str(uuid.uuid4()),
        "date": date.strftime("%Y-%m-%d"),
        "type": workout_type,
        "duration": duration,
        "notes": notes,
        "exercises": [ex for ex in exercises if ex["exercise"].strip()],
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if 'workouts' not in st.session_state:
        st.session_state.workouts = []

    st.session_state.workouts.append(workout)
    save_workouts()

    # Upload to Google Drive if enabled
    if st.session_state.config.get("upload_to_drive_enabled", False):
        upload_to_drive(files=["strong_workouts.json"])

def save_workouts():
    """Save workouts to JSON file"""
    try:
        with open("strong_workouts.json", "w") as f:
            json.dump(st.session_state.workouts, f, indent=2)
        logger.info("Workouts saved successfully")
    except Exception as e:
        logger.error(f"Error saving workouts: {str(e)}")


def save_config(config):
    """Save configuration to file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Configuration saved successfully")
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")

def load_workouts():
    """Load workouts from JSON file."""
    if 'workouts' not in st.session_state:
        try:
            if os.path.exists("strong_workouts.json"):
                with open("strong_workouts.json", 'r') as f:
                    st.session_state.workouts = json.load(f)
            else:
                st.session_state.workouts = []
        except Exception as e:
            logger.error(f"Error loading workouts: {str(e)}")
            st.session_state.workouts = []

    # Load config
    if 'config' not in st.session_state:
        st.session_state.config = load_config()

def dashboard_section():
    """Main dashboard with workout logging."""
    initialize_session_state()
    load_data()

    # Display current workout logging section (admin-protected)
    workout_logging_section()

    # Always show workout history (but delete buttons are admin-only)
    workout_history_section()

def analytics_section():
    """Analytics and statistics section."""
    initialize_session_state()
    load_data()

    # Display workout statistics
    statistics_section()

def chatbot_section():
    """AI chatbot for fitness advice with data access and question templates."""
    st.header("💬 AI Fitness Coach Chat")

    if not st.session_state.api_key:
        st.error("❌ Please set your Google AI API key in credentials.json")
        return

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Question templates section (keep existing code)
    st.subheader("🚀 Quick Question Templates")

    # Define question templates
    question_templates = {
        "📊 Data Analysis": [
            "What's my workout consistency over the last month?",
            "Which muscle groups am I training most/least?",
            "How has my workout duration changed over time?",
            "What's my weekly workout frequency?",
            "Show me my cardio vs strength training balance"
        ],
        "💪 Progress & Goals": [
            "How am I progressing compared to last month?",
            "What are my strongest and weakest areas?",
            "Set challenging goals for next week",
            "Am I overtraining or undertraining?",
            "What should I focus on to break plateaus?"
        ],
        "🎯 Workout Planning": [
            "Create a balanced weekly workout plan",
            "Suggest exercises for my weak muscle groups",
            "How can I improve my workout intensity?",
            "What's the optimal rest between my workouts?",
            "Design a progressive overload plan"
        ],
        "🍎 Nutrition & Recovery": [
            "Nutrition recommendations based on my workouts",
            "How much protein should I consume daily?",
            "Best post-workout recovery strategies",
            "Sleep recommendations for my training schedule",
            "Hydration needs for my activity level"
        ]
    }

    # Display templates in columns
    cols = st.columns(2)
    for i, (category, questions) in enumerate(question_templates.items()):
        with cols[i % 2]:
            with st.expander(category):
                for question in questions:
                    if st.button(question, key=f"template_{hash(question)}"):
                        # Add to chat and process
                        st.session_state.chat_history.append({"role": "user", "content": question})
                        process_chat_response(question)
                        st.rerun()

    st.divider()

    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.chat_message("user").write(message['content'])
        else:
            st.chat_message("assistant").write(message['content'])

    # Chat input
    user_input = st.chat_input("Ask me anything about fitness, nutrition, or your workouts...")

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        process_chat_response(user_input)

    # Clear chat button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

def process_chat_response(user_input):
    """Process chat response with full data access."""
    with st.spinner("Analyzing your data and thinking..."):
        try:
            # Prepare comprehensive workout data context
            workout_data_context = prepare_workout_context()

            prompt = f"""You are an expert AI fitness coach with full access to the user's workout data. 
Provide detailed, data-driven insights and recommendations.

User question: {user_input}

{workout_data_context}

Guidelines:
- Use specific data points from their workout history when relevant
- Provide actionable, personalized advice based on their actual performance
- Be motivational and encouraging while being realistic
- Include specific numbers, dates, and trends when discussing their progress
- Suggest concrete next steps and improvements
- If asked about data analysis, provide detailed breakdowns with statistics

Provide a comprehensive, helpful response:"""

            model = ChatGoogleGenerativeAI(
                model=st.session_state.llm_model,
                google_api_key=st.session_state.api_key,
                temperature=0.7
            )

            message = HumanMessage(content=prompt)
            response = model.invoke([message])

            # Add AI response to history
            ai_response = response.content
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            st.chat_message("assistant").write(ai_response)

        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            logger.error(f"Chatbot error: {str(e)}")

def prepare_workout_context():
    """Prepare comprehensive workout data context for the AI."""
    context = "=== USER'S COMPLETE WORKOUT DATA ===\n\n"

    # Check both data sources
    has_csv_data = not st.session_state.workout_data.empty
    has_json_data = hasattr(st.session_state, 'workouts') and st.session_state.workouts

    if has_csv_data:
        df = st.session_state.workout_data
        context += "=== CSV WORKOUT DATA ===\n"
        context += f"Total entries: {len(df)}\n"
        context += f"Date range: {df['date'].min()} to {df['date'].max()}\n\n"

        # Recent workouts (last 10)
        recent_df = df.sort_values('date', ascending=False).head(10)
        context += "Recent 10 workouts:\n"
        for _, row in recent_df.iterrows():
            context += f"- {row['date']}: {row['workout_types']} | Gym: {row['gym_time_minutes']}min | Cardio: {row['cardio_time_minutes']}min | Calories: {row['cardio_calories']} | Distance: {row['cardio_distance']}\n"

        # Statistics
        context += f"\nSTATISTICS:\n"
        context += f"- Average gym time: {df['gym_time_minutes'].mean():.1f} minutes\n"
        context += f"- Average cardio time: {df['cardio_time_minutes'].mean():.1f} minutes\n"
        context += f"- Total calories burned: {df['cardio_calories'].sum()}\n"
        context += f"- Most common workout types: {df['workout_types'].value_counts().head().to_dict()}\n"

        # Weekly/Monthly patterns
        df['date'] = pd.to_datetime(df['date'])
        df['weekday'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.strftime('%Y-%m')
        context += f"- Workout frequency by day: {df['weekday'].value_counts().to_dict()}\n"
        context += f"- Monthly workout counts: {df['month'].value_counts().to_dict()}\n\n"

    if has_json_data:
        workouts = st.session_state.workouts
        context += "=== DETAILED WORKOUT DATA ===\n"
        context += f"Total detailed workouts: {len(workouts)}\n"

        # Recent detailed workouts (last 10)
        recent_workouts = workouts[-10:] if len(workouts) >= 10 else workouts
        context += "Recent detailed workouts:\n"
        for workout in recent_workouts:
            context += f"- {workout['date']}: {workout['type']} ({workout['duration']}min)\n"
            if workout.get('exercises'):
                context += "  Exercises:\n"
                for ex in workout['exercises']:
                    if ex.get('exercise'):
                        context += f"    * {ex['exercise']}: {ex.get('sets', '')}x{ex.get('reps', '')} @ {ex.get('weight', '')}\n"
            if workout.get('notes'):
                context += f"  Notes: {workout['notes']}\n"

        # Workout type analysis
        workout_types = {}
        total_duration = 0
        for w in workouts:
            workout_types[w['type']] = workout_types.get(w['type'], 0) + 1
            total_duration += w.get('duration', 0)

        context += f"\nDETAILED STATISTICS:\n"
        context += f"- Workout types distribution: {workout_types}\n"
        context += f"- Average workout duration: {total_duration/len(workouts):.1f} minutes\n"
        context += f"- Total training time: {total_duration} minutes ({total_duration/60:.1f} hours)\n"

        # Exercise analysis
        all_exercises = []
        for workout in workouts:
            for ex in workout.get('exercises', []):
                if ex.get('exercise'):
                    all_exercises.append(ex['exercise'])

        if all_exercises:
            exercise_counts = {}
            for ex in all_exercises:
                exercise_counts[ex] = exercise_counts.get(ex, 0) + 1
            context += f"- Most performed exercises: {dict(sorted(exercise_counts.items(), key=lambda x: x[1], reverse=True)[:10])}\n"

    if not has_csv_data and not has_json_data:
        context += "No workout data available. User should start logging workouts to get personalized insights.\n"

    # Current date context
    context += f"\n=== CURRENT DATE CONTEXT ===\n"
    context += f"Today: {datetime.date.today()}\n"
    context += f"Current day of week: {datetime.date.today().strftime('%A')}\n"

    return context

#-----------------------------------------
# Main Application
#-----------------------------------------

def main():
    """Main app"""

    # Set page config FIRST, before any other Streamlit commands
    st.set_page_config(
        page_title="Strong - Workout Tracker App",
        page_icon="🏋️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_workouts()
    admin_authentication()
    header_section()

    # Main tabs - add Admin Panel
    if st.session_state.is_admin:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Analytics", "🤖 Admin Panel", "💬 AI Chat"])
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analytics", "💬 AI Chat"])

    with tab1:
        dashboard_section()

    with tab2:
        analytics_section()

    if st.session_state.is_admin:
        with tab3:
            admin_panel_section()
        with tab4:
            chatbot_section()
    else:
        with tab3:
            chatbot_section()

if __name__ == "__main__":
    main()