from dateutil.parser import parse as parse_jira_datetime
from datetime import datetime

def get_story_points_at_time(jira, issue_key, timestamp):
    """
    Fetch the story points of a Jira issue as of a given timestamp.
    - jira: an authenticated JIRA client object
    - issue_key: the key of the issue (e.g., 'PROJ-123')
    - timestamp: a datetime object
    """
    issue = jira.issue(issue_key, expand='changelog')
    story_points = None
    # Go through changelog in chronological order
    for history in sorted(issue.changelog.histories, key=lambda h: h.created):
        for item in history.items:
            if item.field == 'Story point estimate':
                # Only consider changes before or at the timestamp
                if parse_jira_datetime(history.created) <= timestamp:
                    try:
                        story_points = float(item.toString)
                    except Exception:
                        continue
    # If no change found before timestamp, fallback to current value
    if story_points is None:
        try:
            story_points = float(getattr(issue.fields, 'customfield_10016', 0))  # Replace with your Story Points field ID
        except Exception:
            story_points = 0
    return story_points

def get_done_timestamps_in_sprint(jira, issue_key, sprint_start, sprint_end, done_status="Done"):
    """
    Return a list of (timestamp, issue_key) tuples when the issue was marked as 'Done' within the sprint.
    - jira: an authenticated JIRA client object
    - issue_key: the key of the issue (e.g., 'PROJ-123')
    - sprint_start, sprint_end: datetime objects for the sprint window
    - done_status: the status name that means 'done' (default: 'Done')
    """
    issue = jira.issue(issue_key, expand='changelog')
    done_events = []
    for history in sorted(issue.changelog.histories, key=lambda h: parse_jira_datetime(h.created)):
        for item in history.items:
            if item.field == 'status' and item.toString == done_status:
                change_time = parse_jira_datetime(history.created)
                if sprint_start <= change_time <= sprint_end:
                    done_events.append((change_time, issue_key))
    return done_events

def get_mid_sprint_additions(jira, sprint_id, sprint_start, sprint_end):
    """
    Identify stories that were added to the sprint after sprint start.
    Returns a list of tuples containing (issue_key, addition_timestamp).
    
    Args:
        jira: an authenticated JIRA client object
        sprint_id: the ID of the sprint
        sprint_start: datetime object for sprint start
        sprint_end: datetime object for sprint end
    
    Returns:
        List of tuples (issue_key, timestamp) where:
        - issue_key is the Jira issue key
        - timestamp is when the issue was added to the sprint
    """
    # Get all issues in the sprint
    issues = jira.search_issues(f'sprint = {sprint_id}', expand='changelog')
    mid_sprint_additions = []
    
    for issue in issues:
        issue_key = issue.key
        was_added_mid_sprint = False
        addition_time = None
        
        # Check changelog for Sprint field changes
        if hasattr(issue, 'changelog') and hasattr(issue.changelog, 'histories'):
            for history in sorted(issue.changelog.histories, key=lambda h: parse_jira_datetime(h.created)):
                for item in history.items:
                    if item.field == 'Sprint':
                        change_time = parse_jira_datetime(history.created)
                        # Check if the issue was added to the sprint after sprint start
                        if sprint_start < change_time <= sprint_end:
                            was_added_mid_sprint = True
                            addition_time = change_time
                            break
                if was_added_mid_sprint:
                    break
        
        if was_added_mid_sprint and addition_time:
            mid_sprint_additions.append((issue_key, addition_time))
    print(f"mid_sprint_additions: {mid_sprint_additions}")
    # Sort by addition timestamp
    return sorted(mid_sprint_additions, key=lambda x: x[1])

def get_story_point_changes(jira, sprint_id, sprint_start, sprint_end):
    """
    Get all stories whose story point estimates changed during the sprint.
    
    Args:
        jira: JIRA client instance
        sprint_id: ID of the sprint
        sprint_start: datetime object for sprint start
        sprint_end: datetime object for sprint end
        
    Returns:
        list of tuples: (issue_key, timestamp, old_points, new_points)
    """
    # Get all issues in the sprint
    issues = jira.search_issues(f'sprint = {sprint_id}', expand='changelog')
    story_point_changes = []
    
    for issue in issues:
        # Get the changelog for the issue
        if hasattr(issue, 'changelog') and hasattr(issue.changelog, 'histories'):
            for history in issue.changelog.histories:
                # Convert history timestamp to datetime
                change_time = parse_jira_datetime(history.created)
                if not change_time:
                    continue
                
                # Check if the change happened during the sprint
                if sprint_start <= change_time <= sprint_end:
                    for item in history.items:
                        # Look for story point estimate changes
                        if item.field == 'Story point estimate':
                            try:
                                old_points = float(item.fromString) if item.fromString else 0
                                new_points = float(item.toString) if item.toString else 0
                                story_point_changes.append({
                                    'issue_key': issue.key,
                                    'timestamp': change_time,
                                    'old_points': old_points,
                                    'new_points': new_points,
                                    'type': 'story_point_change'
                                })
                            except (ValueError, TypeError):
                                # Skip if points can't be converted to float
                                continue
    
    return story_point_changes

def get_spillover_issues(jira, sprint_id, sprint_start, sprint_end):
    """
    Get all issues that were moved out of the sprint during the sprint period.
    
    Args:
        jira: JIRA client instance
        sprint_id: ID of the sprint
        sprint_start: datetime object for sprint start
        sprint_end: datetime object for sprint end
        
    Returns:
        list of dicts containing:
        {
            'issue_key': str,
            'timestamp': datetime,
            'story_points': float,
            'type': 'spillover'
        }
    """
    # First get all issues currently in the sprint
    current_issues = jira.search_issues(f'sprint = {sprint_id}', expand='changelog')
    current_issue_keys = {issue.key for issue in current_issues}
    
    # Then get all issues that were in the sprint at any point
    all_issues = jira.search_issues(f'sprint = {sprint_id}', expand='changelog')
    spillover_issues = []
    
    for issue in all_issues:
        # Skip if the issue is still in the sprint
        if issue.key in current_issue_keys:
            continue
            
        # Get the changelog for the issue
        if hasattr(issue, 'changelog') and hasattr(issue.changelog, 'histories'):
            was_in_sprint = False
            removal_time = None
            
            # Sort histories chronologically
            for history in sorted(issue.changelog.histories, key=lambda h: parse_jira_datetime(h.created)):
                change_time = parse_jira_datetime(history.created)
                if not change_time:
                    continue
                
                for item in history.items:
                    if item.field == 'Sprint':
                        # Check if the issue was in the sprint at any point
                        if item.toString and str(sprint_id) in item.toString:
                            was_in_sprint = True
                        # Check if the issue was removed from the sprint
                        elif was_in_sprint and sprint_start <= change_time <= sprint_end:
                            removal_time = change_time
                            break
                
                if removal_time:
                    break
            
            # If the issue was removed during the sprint, add it to spillovers
            if removal_time:
                # Get story points at the time of removal
                story_points = get_story_points_at_time(jira, issue.key, removal_time)
                spillover_issues.append({
                    'issue_key': issue.key,
                    'timestamp': removal_time,
                    'story_points': story_points,
                    'type': 'spillover'
                })
    
    return sorted(spillover_issues, key=lambda x: x['timestamp'])



