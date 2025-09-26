# 💪 Strong App - Workout Tracker

A comprehensive fitness tracking application built with Streamlit that helps you log workouts, track progress, and get AI-powered fitness insights.

## ✨ Features

### 🏋️ Workout Logging
- **Multi-type Workout Selection**: Track Chest, Back, Shoulder, Biceps, Triceps, Legs, and Abs workouts
- **Time Tracking**: Log gym time and cardio sessions separately
- **Smart Image Processing**: Upload cardio machine displays and automatically extract stats using AI
- **Manual Data Entry**: Manually input calories burned and distance covered
- **Notes & Comments**: Add personal notes to each workout session

### 📊 Analytics & Statistics
- **Progress Tracking**: Visual charts showing workout frequency and progress over time
- **Monthly Trends**: Track your consistency and improvements month by month
- **Workout Type Analysis**: See which muscle groups you're training most/least
- **Comprehensive Metrics**: Total workouts, gym time, cardio time, and calories burned

### 🤖 AI-Powered Features
- **Fitness Chat Coach**: Ask questions about workouts, nutrition, and fitness goals
- **Smart Image Analysis**: Extract workout data from cardio machine displays
- **Personalized Insights**: Get AI-generated recommendations based on your workout history
- **Question Templates**: Quick-start questions for common fitness topics

### 👥 Multi-User Support
- **Admin Panel**: Secure admin access for workout management
- **Data Protection**: Admin-only access for adding/deleting workout entries
- **User-Friendly Interface**: View-only access for regular users

### ☁️ Cloud Integration
- **Google Drive Sync**: Automatic backup and sync of workout data
- **Cross-Device Access**: Access your data from anywhere
- **Data Persistence**: Never lose your workout history

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Google AI API key for Gemini
- Google Drive service account (optional, for cloud sync)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd strong-workout-tracker
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up configuration files**
   
   Create `credentials.json`:
   ```json
   {
     "gemini_api_key": "your-google-ai-api-key",
     "llm_model": "gemini-2.0-flash-lite"
   }
   ```
   
   Create `config.json`:
   ```json
   {
     "upload_to_drive_enabled": false
   }
   ```

4. **Set environment variables**
   ```bash
   export ADMIN_PASSWORD="your-secure-admin-password"
   ```

5. **Run the application**
   ```bash
   streamlit run strong_app.py
   ```

## 📋 Usage Guide

### For Regular Users
1. **View Dashboard**: See recent workouts and statistics
2. **Explore Analytics**: Check progress charts and workout trends
3. **Use AI Chat**: Ask fitness questions and get personalized advice

### For Admins
1. **Login**: Use the admin password in the sidebar
2. **Log Workouts**: Add new workout entries with detailed information
3. **Upload Images**: Take photos of cardio machines for automatic data extraction
4. **Manage Data**: Edit, delete, or export workout data
5. **Access Admin Panel**: Get AI insights and manage application settings

## 🎯 Key Workflows

### Logging a Workout
1. Select workout date
2. Choose workout types (multiple selection supported)
3. Enter gym time and cardio time
4. Upload cardio machine image (optional) or enter data manually
5. Add notes about the session
6. Save the workout

### Getting AI Insights
1. Navigate to AI Chat or Admin Panel
2. Use quick question templates or ask custom questions
3. Get personalized recommendations based on your workout history
4. Review progress analysis and goal suggestions

### Viewing Progress
1. Check the Analytics tab for comprehensive statistics
2. Review monthly trends and workout frequency
3. Analyze workout type distribution
4. Track total time invested and calories burned

## 🔧 Configuration

### Google Drive Sync (Optional)
- Set up a Google service account
- Enable Google Drive API
- Configure cloud storage for automatic backup

### AI Features
- Requires Google AI (Gemini) API key
- Supports multiple Gemini models
- Used for image analysis and fitness coaching

## 📱 Interface Features

### Wide Layout
- Optimized for desktop and tablet viewing
- Maximum screen real estate utilization
- Clear visual separation of sections

### Responsive Design
- Works on various screen sizes
- Mobile-friendly interface
- Intuitive navigation with tabs

### Visual Elements
- Emoji-enhanced UI for better user experience
- Color-coded metrics and statistics
- Interactive charts and graphs

## 🛡️ Security Features

- **Admin Authentication**: Secure access control
- **Data Protection**: User data is protected from unauthorized access
- **Local Storage**: Option to keep data locally without cloud sync
- **Environment Variables**: Sensitive data stored securely

## 📊 Data Management

### Data Storage
- Local CSV files for workout data
- JSON configuration files
- Optional cloud backup via Google Drive

### Data Export
- Download workout data as CSV
- Full data portability
- Backup and restore capabilities

### Data Privacy
- All data processing happens locally
- Optional cloud sync with user control
- No third-party data sharing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or feature requests:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

## 🎉 Acknowledgments

- Built with Streamlit for the web interface
- Google AI (Gemini) for intelligent features
- Google Drive API for cloud storage
- Pandas for data processing
- PIL for image handling

---

**Start your fitness journey today! 💪 Keep pushing your limits! 🚀**
```

This README provides comprehensive information about the app while keeping sensitive credential information secure. It covers all the major features, setup instructions, usage guidelines, and technical details that users need to get started with the Strong Workout Tracker app.