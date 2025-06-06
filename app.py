from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import pandas as pd
from jira import JIRA
from dotenv import load_dotenv
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import json
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import numpy as np
from burndown import (
    get_story_points_at_time,
    get_done_timestamps_in_sprint,
    get_mid_sprint_additions,
    get_story_point_changes,
    get_spillover_issues
)

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Jira API configuration
JIRA_URL = os.getenv('JIRA_URL')
JIRA_USERNAME = os.getenv('JIRA_USERNAME')
JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN')

# Global variable for current sprint ID
current_sprint_id = None

# Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Set up the model configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')

try:
    jira = JIRA(server=JIRA_URL, basic_auth=(JIRA_USERNAME, JIRA_API_TOKEN))
except Exception as e:
    print(f"Error connecting to Jira: {str(e)}")
    jira = None

@app.route('/boards', methods=['GET'])
def get_boards():
    try:
        if not jira:
            return jsonify({'error': 'Jira connection not available'}), 500
        
        boards = jira.boards()
        return jsonify([{'id': board.id, 'name': board.name} for board in boards])
    except Exception as e:
        print(f"Error fetching boards: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/sprints', methods=['GET'])
def get_sprints():
    try:
        board_id = request.args.get('board_id')
        if not board_id:
            return jsonify({'error': 'Board ID is required'}), 400
        
        if not jira:
            return jsonify({'error': 'Jira connection not available'}), 500

        sprints = jira.sprints(board_id)
        return jsonify([{'id': sprint.id, 'name': sprint.name} for sprint in sprints])
    except Exception as e:
        print(f"Error fetching sprints: {str(e)}")
        return jsonify({'error': str(e)}), 500

def parse_jira_datetime(date_str):
    """Parse Jira datetime string to datetime object."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f%z")
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            return None

def calculate_sprint_metrics(sprint_data):
    """Calculate sprint metrics including unassigned stories."""
    # Get sprint dates
    sprint_start = parse_jira_datetime(sprint_data['sprint']['startDate'])
    sprint_end = parse_jira_datetime(sprint_data['sprint']['endDate'])
    
    # Ensure both sprint dates are timezone-aware
    utc = pytz.UTC
    if sprint_start and sprint_start.tzinfo is None:
        sprint_start = utc.localize(sprint_start)
    if sprint_end and sprint_end.tzinfo is None:
        sprint_end = utc.localize(sprint_end)
    
    # Initialize metrics
    metrics = {
        'committed': 0,
        'completed': 0,
        'churned_issues': 0,
        'total_issues': len(sprint_data['issues']),
        'churn_percentage': 0,
        'predictability': 0,
        'variance': 0,
        'committed_capacity': 0,
        'utilized_capacity': 0
    }
    
    # Calculate churn metrics
    for story in sprint_data['issues']:
        was_added_during_sprint = False
        for change in story.get('changelog', []):
            change_date = parse_jira_datetime(change.get('created'))
            if not change_date:
                continue
            if change_date.tzinfo is None:
                change_date = utc.localize(change_date)
            if change.get('field') == 'Sprint' and sprint_start < change_date <= sprint_end:
                was_added_during_sprint = True
                break
        
        if was_added_during_sprint:
            metrics['churned_issues'] += 1
    
    # Calculate churn percentage
    if metrics['total_issues'] > 0:
        metrics['churn_percentage'] = int((metrics['churned_issues'] / metrics['total_issues']) * 100)
    else:
        metrics['churn_percentage'] = 0
    
    # Rest of the existing metrics calculation
    for story in sprint_data['issues']:
        # Determine if story was in sprint at start
        was_in_sprint_at_start = False
        was_completed_during_sprint = False
        # Check changelog to determine story status
        for change in story.get('changelog', []):
            change_date = parse_jira_datetime(change.get('created'))
            if not change_date:
                continue
            if change_date.tzinfo is None:
                change_date = utc.localize(change_date)
            if change.get('field') == 'Sprint':
                if change_date <= sprint_start:
                    was_in_sprint_at_start = True
            if change.get('field') == 'status' and change.get('to') == 'Done':
                if sprint_start <= change_date <= sprint_end:
                    was_completed_during_sprint = True
        if not any(change.get('field') == 'Sprint' for change in story.get('changelog', [])):
            story_created = parse_jira_datetime(story.get('created'))
            if story_created:
                if story_created.tzinfo is None:
                    story_created = utc.localize(story_created)
                if story_created <= sprint_start:
                    was_in_sprint_at_start = True
        # For committed points, use the latest 'Story point estimate' before or at sprint start
        committed_story_points = 0
        if was_in_sprint_at_start:
            # Find all 'Story point estimate' changes before or at sprint start
            sp_changes = [
                (parse_jira_datetime(change.get('created')), change.get('to'))
                for change in story.get('changelog', [])
                if change.get('field') == 'Story point estimate' and change.get('to')
            ]
            sp_changes = [ (dt, val) for dt, val in sp_changes if dt and (dt.tzinfo or utc) and dt <= sprint_start ]
            if sp_changes:
                # Use the value from the latest such change
                latest_change = max(sp_changes, key=lambda x: x[0])
                try:
                    committed_story_points = int(float(latest_change[1]))
                except Exception:
                    committed_story_points = 0
            else:
                committed_story_points = 0
            metrics['committed'] += committed_story_points
        # For completed points, use the latest value (as before)
        story_points = story.get('story_points')
        if story_points is None:
            for change in reversed(story.get('changelog', [])):
                if change.get('field') == 'Story point estimate' and change.get('to'):
                    try:
                        story_points = int(float(change['to']))
                    except Exception:
                        story_points = 0
                    break
        if story_points is None:
            story_points = 0
        if was_completed_during_sprint:
            metrics['completed'] += story_points
    
    # Calculate predictability
    if metrics['committed'] > 0:
        metrics['predictability'] = int((metrics['completed'] / metrics['committed']) * 100)
    else:
        metrics['predictability'] = 0

    # Calculate variance
    if metrics['committed_capacity'] > 0:
        metrics['variance'] = int((metrics['utilized_capacity'] / metrics['committed_capacity']) * 100)
    else:
        metrics['variance'] = 0
        
    return metrics

@app.route('/generate_report', methods=['POST'])
def generate_report():
    global current_sprint_id
    try:
        # Check if request has files
        if 'excel_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        excel_file = request.files['excel_file']
        if not excel_file:
            return jsonify({'error': 'No file selected'}), 400

        # Get other form data
        board_id = request.form.get('board_id')
        sprint_id = request.form.get('sprint_id')
        recipient_email = request.form.get('recipient_email')

        if not all([board_id, sprint_id, recipient_email]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Set the current sprint ID
        current_sprint_id = sprint_id
        print(f"Set current_sprint_id to: {current_sprint_id}")  # Debug log

        if not jira:
            return jsonify({'error': 'Jira connection not available'}), 500

        # Save the uploaded file temporarily
        temp_path = 'temp_excel.xlsx'
        excel_file.save(temp_path)

        try:
            # Fetch sprint data from Jira
            print(f"Fetching sprint data for sprint_id: {sprint_id}")  # Debug log
            sprint = jira.sprint(sprint_id)
            if not sprint:
                return jsonify({'error': 'Sprint not found'}), 404

            # Get issues that were in the sprint at any point
            print(f"Searching for issues in sprint {sprint_id}")  # Debug log
            jql = f'sprint = {sprint_id}'
            print(f"Using JQL: {jql}")  # Debug log
            
            issues = jira.search_issues(
                jql,
                expand='changelog',
                fields='summary,description,comment,customfield_10016,status,created'
            )

            print(f"Found {len(issues)} issues")  # Debug log

            if not issues:
                return jsonify({'error': 'No issues found in sprint'}), 404

            # Prepare sprint data for metrics calculation
            sprint_data = {
                'sprint': {
                    'startDate': sprint.startDate,
                    'endDate': sprint.endDate
                },
                'issues': []
            }

            # Process issues with proper error handling
            print("Processing issues...")  # Debug log
            for issue in issues:
                try:
                    print(f"Processing issue {issue.key}")  # Debug log
                    issue_data = {
                        'key': issue.key,
                        'summary': getattr(issue.fields, 'summary', ''),
                        'description': getattr(issue.fields, 'description', ''),
                        'story_points': getattr(issue.fields, 'customfield_10016', 0),
                        'status': getattr(issue.fields, 'status', {}).name if hasattr(issue.fields, 'status') else 'Unknown',
                        'created': getattr(issue.fields, 'created', None),
                        'changelog': []
                    }

                    # Get changelog if available
                    if hasattr(issue, 'changelog'):
                        print(f"Processing changelog for {issue.key}")  # Debug log
                        for history in issue.changelog.histories:
                            for item in history.items:
                                author_name = 'Unknown'
                                if hasattr(history, 'author') and history.author:
                                    author_name = getattr(history.author, 'displayName', 'Unknown')
                                
                                issue_data['changelog'].append({
                                    'author': author_name,
                                    'field': item.field,
                                    'from': item.fromString,
                                    'to': item.toString,
                                    'created': history.created
                                })

                    sprint_data['issues'].append(issue_data)
                    print(f"Successfully processed issue {issue.key}")  # Debug log
                except Exception as e:
                    print(f"Error processing issue {issue.key}: {str(e)}")
                    continue

            print(f"Total processed issues: {len(sprint_data['issues'])}")  # Debug log

            if not sprint_data['issues']:
                return jsonify({'error': 'No valid issues found in sprint'}), 404

            # Calculate sprint metrics
            print("Calculating sprint metrics...")  # Debug log
            sprint_metrics = calculate_sprint_metrics(sprint_data)
            print(f"Calculated metrics: {sprint_metrics}")  # Debug log

            # Read Excel file
            print("Reading Excel file...")  # Debug log
            df = pd.read_excel(temp_path)
            excel_data = df.to_dict('records')
            print(f"Excel data rows: {len(excel_data)}")  # Debug log

            # Prepare LLM input
            print("Preparing LLM input...")  # Debug log
            llm_input = {
                'sprint_name': sprint.name,
                'sprint_start': sprint.startDate,
                'sprint_end': sprint.endDate,
                'excel_data': excel_data,
                'goals': sprint.goal if hasattr(sprint, 'goal') else [],
                'calculated_metrics': {
                    'points_committed': sprint_metrics['committed'],
                    'points_delivered': sprint_metrics['completed'],
                    'predictability': sprint_metrics['predictability'],
                    'variance': sprint_metrics['variance'],
                    'churn_percentage': sprint_metrics['churn_percentage']
                },
                'issues': sprint_data['issues']
            }

            print("LLM input prepared successfully")  # Debug log
            print(f"Number of issues in LLM input: {len(llm_input['issues'])}")  # Debug log

            # Call LLM
            print("Calling LLM...")  # Debug log
            llm_output = call_llm(llm_input)
            print("LLM call completed")  # Debug log

            # Format and send email
            print("Sending email...")  # Debug log
            send_email(recipient_email, llm_output, sprint.name)
            print("Email sent successfully")  # Debug log

            return jsonify({'message': 'Report generated and email sent successfully'})
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print("Temporary file cleaned up")  # Debug log

    except Exception as e:
        print(f"Error generating report: {str(e)}")
        print(f"Error type: {type(e)}")  # Debug log
        import traceback
        print(f"Traceback: {traceback.format_exc()}")  # Debug log
        return jsonify({'error': str(e)}), 500

def call_llm(data):
    try:
        # Initialize the model with API key
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # First, analyze goals and their related stories
        goals_analysis_prompt = f"""
        Analyze the sprint goals and identify which stories are related to each goal based on their descriptions.
        For each goal, determine if it is met by checking if all related stories are closed.
        
        SPRINT GOALS:
        {data['goals']}

        ISSUES:
        {json.dumps(data['issues'], indent=2)}

        Return a JSON object with this structure:
        {{
            "goals_analysis": [
                {{
                    "goal": "string",
                    "related_stories": ["story_key1", "story_key2"],
                    "is_met": boolean,
                    "completion_percentage": number
                }}
            ],
            "overall_completion": number
        }}

        Rules:
        1. A story is related to a goal if its description or summary contains concepts from the goal
        2. A story is related to a goal if its description or summary work in the context of the goal
        3. Only consider most relevant stories for the goal. Do not assign same story to multiple goals. 
        4. A goal is considered met if ALL its related stories are closed (status = 'Done')
        5. Calculate completion percentage as (number of met goals / total goals) * 100
        6. Make sure the response is valid JSON
        7. Return ONLY the JSON object, no other text
        """

        goals_analysis = model.generate_content(goals_analysis_prompt)
        
        
        
        if not goals_analysis or not hasattr(goals_analysis, 'text') or not goals_analysis.text:
            raise ValueError("Empty or invalid response from Gemini API for goals analysis")
            
        # Clean the goals analysis response
        goals_text = goals_analysis.text.strip()
        
        # Remove markdown code block markers if present
        if goals_text.startswith('```'):
            first_newline = goals_text.find('\n')
            if first_newline != -1:
                goals_text = goals_text[first_newline:].strip()
        if goals_text.endswith('```'):
            goals_text = goals_text[:-3].strip()
            
        # Ensure the response is wrapped in curly braces
        if not goals_text.startswith('{'):
            goals_text = '{' + goals_text
        if not goals_text.endswith('}'):
            goals_text = goals_text + '}'
            
        try:
            goals_data = json.loads(goals_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in goals analysis: {str(e)}")
            print(f"Invalid JSON text: {goals_text}")
            # Fallback to default goals analysis
            goals_data = {
                "goals_analysis": [
                    {
                        "goal": goal,
                        "related_stories": [],
                        "is_met": False,
                        "completion_percentage": 0
                    } for goal in data['goals']
                ],
                "overall_completion": 0
            }

        # Now generate the main report
        prompt = f"""
        You are a sprint report analyzer. Your task is to analyze the sprint data and return a JSON object.
        
        RULES:
        1. Return ONLY a valid JSON object, no other text
        2. Do not include any explanations or markdown
        3. The response must start with {{ and end with }}
        4. All values must be properly quoted
        5. Use null for missing values
        
        REQUIRED JSON STRUCTURE:
        {{
            "sprint_name": "{data['sprint_name']}",
            "sprint_start": "{data['sprint_start']}",
            "sprint_end": "{data['sprint_end']}",
            "metrics": {{
                "Team Size": number,
                "Average Velocity": number,
                "Ideal Availability": number,
                "Committed Capacity": number,
                "Utilized Capcity": number,
                "Variance": number( calculated as Variance= (Utilized Capcity/Committed Capacity)*100 %, round off to integer),
                "Points Committed": {data['calculated_metrics']['points_committed']},
                "Points Delivered": {data['calculated_metrics']['points_delivered']},
                "Predictability": {data['calculated_metrics']['predictability']},
                "churn": {data['calculated_metrics']['churn_percentage']}
            }},
            "goals": [
                {{
                    "text": "string",
                    "is_met": boolean,
                    "related_stories": ["story_key1", "story_key2"]
                }}
            ],
            "key_achievements": [
                "Give detailed description of the key achievements of the sprint",
                "Give detailed description of the key achievements of the sprint"
            ],
            "improvement_areas": [
                {{
                    "story_number": "string",
                    "issue": "string",
                    "impact": "string",
                    "proposed_action": "string",
                    "owner": "string"
                }}
            ]
        }}

        DATA TO ANALYZE:
        EXCEL DATA:
        {json.dumps(data['excel_data'], indent=2)}

        SPRINT INFORMATION:
        - Sprint Name: {data['sprint_name']}
        - Sprint Period: {data['sprint_start']} to {data['sprint_end']}

        SPRINT GOALS ANALYSIS:
        {json.dumps(goals_data, indent=2)}

        ISSUES:
        {json.dumps(data['issues'], indent=2)}

        Remember: Return ONLY the JSON object, nothing else. Do not include markdown code blocks or any other formatting.

        Instructions:
        1. Key achievements:
        You are an Agile Coach. I'm providing sprint data including issues, user stories, subtasks, changelogs, and sprint metrics (velocity, burn down, completion rate, etc.).
        Based on this data, give me a summary of key achievements of the sprint. Focus on:
        -Completed stories and their business value
        -Improvements in delivery metrics (velocity, story points completed, fewer spillovers)
        -Team collaboration or contributions (e.g. resolved blockers, cross-functional work)
        -Any signs of improved planning or execution
        -Do not include blockers, recommendations, or problems.
        -Do not include story numbers in the key achievements.

        2. Improvement areas:
        You are an experienced Agile Coach analyzing the performance of a Scrum team based on sprint data.
        You will be provided with:
        A list of issues/user stories from a sprint (with statuses, story points, assignees, subtasks, and changelogs)
        Sprint-level metrics such as velocity, sprint goal completion, spillover %, churn %, blocked time, bugs reported, QA time, team capacity vs actual effort
        Your task is to identify concrete areas of improvement from this data and output a list of structured observations.
        For each area of improvement, include:
        -story_number: The issue key (e.g., TEAM-123) or Sprint Level if the issue spans multiple stories or is metric-related
        -issue: A concise description of the problem, inefficiency, or gap
        -impact: A brief explanation of how it negatively impacted sprint outcome (delays, reduced velocity, rework, blocked dependencies, etc.)
        -proposed_action: A concrete and practical step the team can take to prevent or mitigate this in future sprints
        -"owner": The most relevant role responsible (e.g., story assignee, Scrum Master, QA Lead, Product Owner, whole team)
        
        -Focus only on problems, inefficiencies, or gaps – skip achievements or completed work
        -Base each improvement area on a specific issue or metric pattern
        -Consider incomplete stories, stories with excessive churn or multiple status changes, unresolved blockers, reassignments, last-minute bug fixes, excessive story point variance, etc.
        -You may include improvement areas that are systemic (e.g., cross-cutting issues affecting QA, planning, communication)
        -Analyze the sprint metrics to identify improvement areas.
        -Example: If the utilized capcity is less than the committed capacity but all the committed points are delivered, then the improvement area should be analysis around story point estimation or workload distribution.
        -Example: If the velocity is less than the ideal velocity, then the improvement area should be analysis around utilisation.
        """

        # Generate content with specific parameters
        response = model.generate_content(prompt)
        
        # Debug logging
        print("Raw response from Gemini:", response)
        print("Response text:", response.text if hasattr(response, 'text') else "No text attribute")
        
        if not response:
            raise ValueError("Empty response from Gemini API")
            
        if not hasattr(response, 'text'):
            raise ValueError("Response has no text attribute")
            
        if not response.text:
            raise ValueError("Empty text in response")
            
        # Clean the response text
        response_text = response.text.strip()
        
        # Remove markdown code block markers if present
        if response_text.startswith('```'):
            # Find the first newline after the opening ```
            first_newline = response_text.find('\n')
            if first_newline != -1:
                # Remove the opening ``` and any language specifier
                response_text = response_text[first_newline:].strip()
        
        # Remove closing ``` if present
        if response_text.endswith('```'):
            response_text = response_text[:-3].strip()
            
        # Ensure the response is wrapped in curly braces
        if not response_text.startswith('{'):
            response_text = '{' + response_text
        if not response_text.endswith('}'):
            response_text = response_text + '}'
            
        # Try to parse the response
        try:
            result = json.loads(response_text)
            # Ensure sprint information is included
            result['sprint_name'] = data['sprint_name']
            result['sprint_start'] = data['sprint_start']
            result['sprint_end'] = data['sprint_end']
            # Add goals completion percentage
            result['goals_completion_percentage'] = goals_data['overall_completion']
            return result
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Invalid JSON text: {response_text}")
            raise ValueError(f"Invalid JSON response from Gemini API: {str(e)}")
            
    except Exception as e:
        print(f"Error in call_llm: {str(e)}")
        raise

def generate_burndown_data(sprint_data):
    """Generate burndown data from sprint issues."""
    print("Starting burndown data generation...")  # Debug log
    
    # Get sprint dates
    sprint_start = parse_jira_datetime(sprint_data['sprint']['startDate'])
    sprint_end = parse_jira_datetime(sprint_data['sprint']['endDate'])
    done_events = []
    
    # Get mid-sprint additions using global sprint ID
    if current_sprint_id:
        # Get mid-sprint additions
        mid_sprint_additions = get_mid_sprint_additions(jira, current_sprint_id, sprint_start, sprint_end)
        for issue_key, timestamp in mid_sprint_additions:
            sp = get_story_points_at_time(jira, issue_key, timestamp)
            done_events.append({
                'timestamp': timestamp, 
                'issue_key': issue_key, 
                'story_points': sp, 
                'type': 'mid_sprint_addition'
            })
        
        # Get story point changes
        story_point_changes = get_story_point_changes(jira, current_sprint_id, sprint_start, sprint_end)
        for change in story_point_changes:
            # Calculate the net change in story points
            point_difference = change['new_points'] - change['old_points']
            done_events.append({
                'timestamp': change['timestamp'],
                'issue_key': change['issue_key'],
                'story_points': abs(point_difference),
                'type': 'story_point_change',
                'is_increase': point_difference > 0
            })
        
        # Get spillover issues
        spillover_issues = get_spillover_issues(jira, current_sprint_id, sprint_start, sprint_end)
        for issue in spillover_issues:
            done_events.append({
                'timestamp': issue['timestamp'],
                'issue_key': issue['issue_key'],
                'story_points': issue['story_points'],
                'type': 'spillover'
            })
    else:
        print("Sprint ID not found in global variable, skipping sprint changes")
    
    # Get done events
    for issue in sprint_data['issues']:
        issue_key = issue['key']
        # get all (timestamp, issue_key) when this issue was marked as done in the sprint
        events = get_done_timestamps_in_sprint(jira, issue_key, sprint_start, sprint_end)
        for ts, key in events:
            sp = get_story_points_at_time(jira, key, ts)
            done_events.append({
                'timestamp': ts, 
                'issue_key': key, 
                'story_points': sp, 
                'type': 'done'
            })
    
    # Order by timestamp
    done_events = sorted(done_events, key=lambda x: x['timestamp'])
    print(f"Done events: {done_events}")
    
    # Calculate sprint metrics to get committed points
    sprint_metrics = calculate_sprint_metrics(sprint_data)
    initial_points = sprint_metrics['committed']
    print(f"Using committed points as initial points: {initial_points}")
    
    return done_events, sprint_start, sprint_end, initial_points

def plot_burndown_from_done_events(done_events, sprint_start, sprint_end, initial_points):
    """
    done_events: list of dicts with 'timestamp' (datetime) and 'story_points' (float)
    sprint_start, sprint_end: datetime objects
    initial_points: float, starting story points
    Returns: BytesIO buffer with PNG image
    """
    # Sort events by timestamp
    done_events = sorted(done_events, key=lambda x: x['timestamp'])
    # Prepare x and y values
    x = [sprint_start]
    y = [initial_points]
    current_points = initial_points
    for event in done_events:
        # Add the event timestamp
        x.append(event['timestamp'])
        # Adjust points based on event type
        if event.get('type') == 'done':
            current_points -= event['story_points']
        elif event.get('type') == 'mid_sprint_addition':
            current_points += event['story_points']
        elif event.get('type') == 'story_point_change':
            if event.get('is_increase', False):
                current_points += event['story_points']
            else:
                current_points -= event['story_points']
        elif event.get('type') == 'spillover':
            current_points -= event['story_points']
        y.append(current_points)

    # Optionally, extend to sprint_end
    if x[-1] < sprint_end:
        x.append(sprint_end)
        y.append(current_points)
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(x, y, where='post', color='#e74c3c', linewidth=2.5, label='Remaining work')
    ax.scatter(x, y, color='#e74c3c', s=50, zorder=5)
    ax.set_xlim(sprint_start, sprint_end)
    ax.set_ylim(0, max(y) * 1.1)  # Adjust y-axis limit to accommodate additions
    ax.set_ylabel('Story points')
    ax.set_xlabel('Time')
    ax.set_title('Burndown Chart')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    fig.autofmt_xdate()
    ax.grid(True, linestyle='--', alpha=0.3, color='#b0b3b8')
    ax.legend(loc='upper right')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)  # Close the figure to free memory
    return buf

def plot_velocity_chart(committed_points, completed_points):
    """Create a bar chart comparing committed vs completed points for 3 sprints."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Set up the data for 3 sprints
    sprints = ['Sprint 1', 'Sprint 2', 'Sprint 3']
    committed = [committed_points, 0, 0]  # Only first sprint has data
    completed = [completed_points, 0, 0]  # Only first sprint has data
    
    # Set the width of the bars
    bar_width = 0.35
    
    # Set the positions of the bars on X axis
    r1 = np.arange(len(sprints))
    r2 = [x + bar_width for x in r1]
    
    # Create the bars
    bars1 = ax.bar(r1, committed, width=bar_width, color='#3498db', label='Committed')
    bars2 = ax.bar(r2, completed, width=bar_width, color='#2ecc71', label='Completed')
    
    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add labels for non-zero values
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
    
    add_labels(bars1)
    add_labels(bars2)
    
    # Customize the chart
    ax.set_ylabel('Story Points')
    ax.set_title('Sprint Velocity')
    ax.set_xticks([r + bar_width/2 for r in range(len(sprints))])
    ax.set_xticklabels(sprints)
    ax.grid(True, linestyle='--', alpha=0.3, color='#b0b3b8')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    plt.close()  # Close the figure to free memory
    return buf

def send_email(recipient, content, sprint_name):
    msg = MIMEMultipart()
    msg['From'] = SMTP_USERNAME
    msg['To'] = recipient
    msg['Subject'] = f"Sprint Report - {sprint_name}"

    # Format dates
    try:
        start_date = datetime.strptime(content['sprint_start'], "%Y-%m-%dT%H:%M:%S.%f%z")
        end_date = datetime.strptime(content['sprint_end'], "%Y-%m-%dT%H:%M:%S.%f%z")
        formatted_start = start_date.strftime("%B %d, %Y")
        formatted_end = end_date.strftime("%B %d, %Y")
    except:
        # Fallback if date parsing fails
        formatted_start = content['sprint_start']
        formatted_end = content['sprint_end']

    # Fetch complete sprint data using global sprint ID
    if current_sprint_id:
        print(f"Fetching complete sprint data for sprint ID: {current_sprint_id}")
        try:
            # Get sprint details
            sprint = jira.sprint(current_sprint_id)
            if not sprint:
                raise ValueError(f"Sprint not found: {current_sprint_id}")

            # Get all issues in the sprint with changelog
            issues = jira.search_issues(
                f'sprint = {current_sprint_id}',
                expand='changelog',
                fields='summary,description,comment,customfield_10016,status,created'
            )

            # Prepare sprint data
            sprint_data = {
                'sprint': {
                    'startDate': sprint.startDate,
                    'endDate': sprint.endDate
                },
                'issues': []
            }

            # Process issues
            for issue in issues:
                try:
                    issue_data = {
                        'key': issue.key,
                        'summary': getattr(issue.fields, 'summary', ''),
                        'description': getattr(issue.fields, 'description', ''),
                        'story_points': getattr(issue.fields, 'customfield_10016', 0),
                        'status': getattr(issue.fields, 'status', {}).name if hasattr(issue.fields, 'status') else 'Unknown',
                        'created': getattr(issue.fields, 'created', None),
                        'changelog': []
                    }

                    # Get changelog if available
                    if hasattr(issue, 'changelog'):
                        for history in issue.changelog.histories:
                            for item in history.items:
                                author_name = 'Unknown'
                                if hasattr(history, 'author') and history.author:
                                    author_name = getattr(history.author, 'displayName', 'Unknown')
                                
                                issue_data['changelog'].append({
                                    'author': author_name,
                                    'field': item.field,
                                    'from': item.fromString,
                                    'to': item.toString,
                                    'created': history.created
                                })

                    sprint_data['issues'].append(issue_data)
                except Exception as e:
                    print(f"Error processing issue {issue.key}: {str(e)}")
                    continue

            print(f"Fetched {len(sprint_data['issues'])} issues for sprint {current_sprint_id}")

            # Calculate sprint metrics
            sprint_metrics = calculate_sprint_metrics(sprint_data)
            
            # Generate burndown chart
            done_events, sprint_start, sprint_end, initial_points = generate_burndown_data(sprint_data)
            burndown_chart = plot_burndown_from_done_events(done_events, sprint_start, sprint_end, initial_points)
            
            # Generate velocity chart
            velocity_chart = plot_velocity_chart(sprint_metrics['committed'], sprint_metrics['completed'])
            
        except Exception as e:
            print(f"Error generating charts: {str(e)}")
            burndown_chart = None
            velocity_chart = None
    else:
        print("No sprint ID available for chart generation")
        burndown_chart = None
        velocity_chart = None

    # Create HTML content with styling
    html_content = f"""
    <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    color: #2c3e50;
                    max-width: 100%;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                h1, h2 {{
                    color: #1a365d;
                    border-bottom: 3px solid #e2e8f0;
                    padding-bottom: 12px;
                    margin-top: 35px;
                    font-weight: 600;
                }}
                h1 {{
                    font-size: 2.2em;
                    text-align: center;
                    margin-bottom: 30px;
                    color: #2d3748;
                }}
                h2 {{
                    font-size: 1.8em;
                    margin-top: 40px;
                }}
                .sprint-header {{
                    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                    padding: 30px;
                    border-radius: 12px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                    border: 1px solid #e2e8f0;
                }}
                .sprint-name {{
                    color: #2d3748;
                    margin: 0 0 15px 0;
                    font-size: 1.8em;
                    font-weight: 700;
                    text-align: center;
                }}
                .sprint-dates {{
                    color: #4a5568;
                    margin: 0;
                    font-size: 1.2em;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 15px;
                }}
                .date-separator {{
                    color: #a0aec0;
                    font-size: 1.4em;
                }}
                .metrics-table {{
                    width: 100%;
                    border-collapse: separate;
                    border-spacing: 0;
                    margin: 30px 0;
                    background: #ffffff;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                    table-layout: fixed;
                }}
                .metrics-table td {{
                    padding: 20px;
                    border: none;
                    vertical-align: middle;
                    position: relative;
                    width: 25%;
                    text-align: center;
                    transition: all 0.3s ease;
                    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                }}
                .metrics-table tr {{
                    transition: background-color 0.3s ease;
                    height: 70px;
                }}
                .metrics-table tr:nth-child(even) td {{
                    background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
                }}
                .metrics-table tr:hover td {{
                    background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                }}
                .metric-label {{
                    font-weight: 600;
                    color: #4a5568;
                    font-size: 0.9em;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    display: block;
                    margin-bottom: 8px;
                    opacity: 0.8;
                }}
                .metric-value {{
                    color: #2b6cb0;
                    font-size: 1.8em;
                    font-weight: 700;
                    display: block;
                    text-shadow: 1px 1px 1px rgba(0,0,0,0.05);
                    margin: 5px 0;
                }}
                .metric-unit {{
                    font-size: 0.75em;
                    color: #718096;
                    margin-left: 4px;
                    font-weight: 500;
                    opacity: 0.8;
                }}
                .metrics-table td:after {{
                    content: '';
                    position: absolute;
                    right: 0;
                    top: 20%;
                    height: 60%;
                    width: 1px;
                    background: linear-gradient(to bottom, transparent, #e2e8f0, transparent);
                }}
                .metrics-table td:last-child:after {{
                    display: none;
                }}
                .bullet-list {{
                    list-style-type: none;
                    padding: 0;
                    background: #ffffff;
                    border-radius: 12px;
                    padding: 25px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                    border: 1px solid #e2e8f0;
                }}
                .bullet-list li {{
                    margin-bottom: 15px;
                    color: #4a5568;
                    padding-left: 25px;
                    position: relative;
                    line-height: 1.6;
                }}
                .bullet-list li:before {{
                    content: "•";
                    color: #4299e1;
                    font-weight: bold;
                    position: absolute;
                    left: 0;
                    font-size: 1.2em;
                }}
                .achievements-section {{
                    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                    border-radius: 12px;
                    padding: 20px;
                    margin: 20px 0;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                    border: 1px solid #e2e8f0;
                }}
                .achievement-item {{
                    display: flex;
                    align-items: flex-start;
                    margin-bottom: 12px;
                    color: #2d3748;
                    line-height: 1.5;
                }}
                .achievement-item:last-child {{
                    margin-bottom: 0;
                }}
                .achievement-icon {{
                    color: #4299e1;
                    font-size: 1.2em;
                    margin-right: 10px;
                    flex-shrink: 0;
                }}
                .achievement-text {{
                    flex: 1;
                }}
                .improvement-table {{
                    width: 100%;
                    border-collapse: separate;
                    border-spacing: 0;
                    margin: 30px 0;
                    background: #ffffff;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                    border: 1px solid #e2e8f0;
                }}
                .improvement-table th {{
                    background: linear-gradient(135deg, #2d3748 0%, #1a365d 100%);
                    color: #ffffff;
                    padding: 18px;
                    text-align: left;
                    font-weight: 600;
                    text-transform: uppercase;
                    font-size: 0.85em;
                    letter-spacing: 0.5px;
                }}
                .improvement-table td {{
                    padding: 18px;
                    border-bottom: 1px solid #e2e8f0;
                    color: #4a5568;
                    line-height: 1.5;
                }}
                .improvement-table tr:last-child td {{
                    border-bottom: none;
                }}
                .improvement-table tr:hover {{
                    background-color: #f8fafc;
                }}
                .metrics-dashboard {{
                    margin-top: 50px;
                    padding-top: 30px;
                    border-top: 3px solid #e2e8f0;
                }}
                .metrics-dashboard h2 {{
                    color: #2d3748;
                    margin-bottom: 35px;
                    font-size: 1.8em;
                    text-align: center;
                }}
                .charts-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 30px;
                    margin: 30px 0;
                }}
                .chart {{
                    flex: 1;
                    min-width: 300px;
                    background: #ffffff;
                    border-radius: 12px;
                    padding: 25px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                    border: 1px solid #e2e8f0;
                }}
                .chart h3 {{
                    color: #2d3748;
                    margin: 0 0 20px 0;
                    font-size: 1.4em;
                    text-align: center;
                }}
                .chart img {{
                    width: 100%;
                    height: auto;
                    border-radius: 8px;
                }}
                @media screen and (max-width: 768px) {{
                    .metrics-table td {{
                        padding: 15px;
                        font-size: 0.9em;
                    }}
                    .metric-label {{
                        font-size: 0.8em;
                    }}
                    .metric-value {{
                        font-size: 1.4em;
                    }}
                    .chart {{
                        min-width: 100%;
                    }}
                    .achievements-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
                .completion-badge {{
                    font-size: 0.7em;
                    background: #4299e1;
                    color: white;
                    padding: 4px 8px;
                    border-radius: 12px;
                    margin-left: 10px;
                    vertical-align: middle;
                }}
                .goal-item {{
                    margin-bottom: 20px;
                    padding: 15px;
                    background: #f8fafc;
                    border-radius: 8px;
                    border: 1px solid #e2e8f0;
                }}
                .goal-content {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 8px;
                }}
                .goal-text {{
                    flex: 1;
                    margin-right: 15px;
                }}
                .goal-status {{
                    font-weight: 600;
                    padding: 4px 8px;
                    border-radius: 6px;
                    font-size: 0.9em;
                }}
                .goal-status.met {{
                    background: #c6f6d5;
                    color: #2f855a;
                }}
                .goal-status.not-met {{
                    background: #fed7d7;
                    color: #c53030;
                }}
                .related-stories {{
                    font-size: 0.9em;
                    color: #4a5568;
                    margin-top: 8px;
                }}
                .stories-label {{
                    font-weight: 600;
                    color: #2d3748;
                }}
            </style>
        </head>
        <body>
            <h1>Sprint Report</h1>
            
            <div class="sprint-header">
                <h3 class="sprint-name">{sprint_name}</h3>
                <p class="sprint-dates">
                    <span>{formatted_start}</span>
                    <span class="date-separator">→</span>
                    <span>{formatted_end}</span>
                </p>
            </div>

            <h2>Sprint Metrics</h2>
            <table class="metrics-table">
                <tr>
                    <td>
                        <span class="metric-label">Team Size</span>
                        <span class="metric-value">{content['metrics']['Team Size']}<span class="metric-unit"></span></span>
                    </td>
                    <td>
                        <span class="metric-label">Ideal Availability</span>
                        <span class="metric-value">{content['metrics']['Ideal Availability']}<span class="metric-unit">hrs</span></span>
                    </td>
                    <td>
                        <span class="metric-label">Average Velocity</span>
                        <span class="metric-value">{content['metrics']['Average Velocity']}<span class="metric-unit">points</span></span>
                    </td>
                    <td>
                        <span class="metric-label">Predictability</span>
                        <span class="metric-value">{content['metrics']['Predictability']}<span class="metric-unit">%</span></span>
                    </td>
                </tr>
                <tr>
                    <td></td>
                    <td>
                        <span class="metric-label">Committed Capacity</span>
                        <span class="metric-value">{content['metrics']['Committed Capacity']}<span class="metric-unit">hrs</span></span>
                    </td>
                    <td>
                        <span class="metric-label">Points Committed</span>
                        <span class="metric-value">{int(content['metrics']['Points Committed'])}<span class="metric-unit">points</span></span>
                    </td>
                    <td>
                        <span class="metric-label">Churn</span>
                        <span class="metric-value">{content['metrics']['churn']}<span class="metric-unit">%</span></span>
                    </td>
                </tr>
                <tr>
                    <td></td>
                    <td>
                        <span class="metric-label">Utilized Capacity</span>
                        <span class="metric-value">{content['metrics']['Utilized Capcity']}<span class="metric-unit">hrs</span></span>
                    </td>
                    <td>
                        <span class="metric-label">Points Delivered</span>
                        <span class="metric-value">{int(content['metrics']['Points Delivered'])}<span class="metric-unit">points</span></span>
                    </td>
                    <td>
                        <span class="metric-label">Variance</span>
                        <span class="metric-value">{content['metrics']['Variance']}<span class="metric-unit">%</span></span>
                    </td>
                </tr>
            </table>

            <h2>Sprint Goals <span class="completion-badge">{content['goals_completion_percentage']}% Complete</span></h2>
            <ul class="bullet-list">
                {''.join(f'''
                <li class="goal-item">
                    <div class="goal-content">
                        <span class="goal-text">{goal["text"]}</span>
                        <span class="goal-status {goal["is_met"] and "met" or "not-met"}">
                            {goal["is_met"] and "✓ Met" or "✗ Not Met"}
                        </span>
                    </div>
                    <div class="related-stories">
                        <span class="stories-label">Related Stories:</span>
                        {', '.join(goal["related_stories"])}
                    </div>
                </li>
                ''' for goal in content['goals'])}
            </ul>

            <h2>Key Achievements</h2>
            <div class="achievements-section">
                {''.join(f'''
                <div class="achievement-item">
                    <span class="achievement-icon">🏆</span>
                    <span class="achievement-text">{achievement}</span>
                </div>
                ''' for achievement in content['key_achievements'])}
            </div>

            <h2>Recommended Improvement Areas</h2>
            <table class="improvement-table">
                <thead>
                    <tr>
                        <th>Story Number</th>
                        <th>Issue</th>
                        <th>Impact</th>
                        <th>Proposed Action</th>
                        <th>Owner</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f'''
                    <tr>
                        <td>{area['story_number']}</td>
                        <td>{area['issue']}</td>
                        <td>{area['impact']}</td>
                        <td>{area['proposed_action']}</td>
                        <td>{area['owner']}</td>
                    </tr>
                    ''' for area in content['improvement_areas'])}
                </tbody>
            </table>

            {f'''
            <div class="metrics-dashboard">
                <h2>Metrics Dashboard</h2>
                <div class="charts-container">
                    <div class="chart">
                        <h3>Burndown Chart</h3>
                        <img src="cid:burndown_chart" alt="Burndown Chart">
                    </div>
                    <div class="chart">
                        <h3>Velocity Chart</h3>
                        <img src="cid:velocity_chart" alt="Velocity Chart">
                    </div>
                </div>
            </div>
            ''' if burndown_chart and velocity_chart else ''}
        </body>
    </html>
    """
    
    msg.attach(MIMEText(html_content, 'html'))

    # Attach charts if available
    if burndown_chart:
        burndown_chart.seek(0)
        image = MIMEImage(burndown_chart.read())
        image.add_header('Content-ID', '<burndown_chart>')
        image.add_header('Content-Disposition', 'attachment', filename=f'burndown_chart_{sprint_name}.png')
        msg.attach(image)

    if velocity_chart:
        velocity_chart.seek(0)
        image = MIMEImage(velocity_chart.read())
        image.add_header('Content-ID', '<velocity_chart>')
        image.add_header('Content-Disposition', 'attachment', filename=f'velocity_chart_{sprint_name}.png')
        msg.attach(image)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        raise

if __name__ == '__main__':
    app.run(debug=True) 