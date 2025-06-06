# Sprint Report Generator

A web application to generate sprint reports using Jira data, Excel capacity sheets, and LLM analysis.

## Tech Stack

- **Frontend:** React
- **Backend:** Python Flask
- **LLM:** Gemini-2.0-Flash
- **Mail:** Google SMTP

## Setup

### Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with the following variables:
   ```
   # Jira Configuration
   JIRA_URL=your_jira_url
   JIRA_USERNAME=your_jira_username
   JIRA_API_TOKEN=your_jira_api_token

   # Gemini Configuration
   GEMINI_API_KEY=your_gemini_api_key

   # Email Configuration
   SMTP_USERNAME=your_gmail_address
   SMTP_PASSWORD=your_gmail_app_password
   ```

   Note: For Gmail, you'll need to:
   1. Enable 2-factor authentication
   2. Generate an App Password (Google Account → Security → App Passwords)
   3. Use the generated App Password as SMTP_PASSWORD

5. Run the Flask server:
   ```bash
   python app.py
   ```

### Frontend

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm start
   ```

## Usage

1. Open your browser and go to `http://localhost:3000`.
2. Select a board and sprint from the dropdowns.
3. Upload the sprint capacity Excel sheet.
4. Enter the recipient email address.
5. Click "Send Email" to generate and send the sprint report.

## Features

- Fetch sprint data from Jira (stories, comments, changelogs, subtasks).
- Parse Excel data for team metrics.
- Calculate sprint metrics (points committed, delivered, predictability, churn).
- Use LLM (Gemini-2.0-Flash) to analyze data and generate insights.
- Send formatted email reports via Google SMTP.

## Excel File Format

The sprint capacity Excel sheet should contain the following columns:
- Team Size
- Ideal Availability (hrs)
- Committed capacity (hrs)
- Utilized capacity (hrs)
- Variance (%)
- Average Velocity(last 3 months) 